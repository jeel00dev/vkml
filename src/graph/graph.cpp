#include "vkml/graph/graph.h"

#include "vkml/graph/grad_mode.h"
#include "vkml/util/assert.h"

#include <format>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace vkml {

Node::~Node() {
    // Move every outgoing reference into a worklist, then drop them one at a
    // time. Whenever we turn out to hold the last reference to a node, steal
    // its references too before letting it die -- so by the time its
    // destructor runs, it has nothing left to recurse into.
    std::vector<NodePtr> pending;

    auto steal = [&pending](Node& n) {
        for (NodePtr& s : n.src) {
            if (s != nullptr) {
                pending.push_back(std::move(s));
            }
        }
        if (n.view_src != nullptr) {
            pending.push_back(std::move(n.view_src));
        }
        if (n.grad != nullptr) {
            pending.push_back(std::move(n.grad));
        }
    };

    steal(*this);

    while (!pending.empty()) {
        NodePtr node = std::move(pending.back());
        pending.pop_back();
        if (node.use_count() == 1) {
            steal(*node);
        }
        // `node` is released here; its own destructor finds nothing to do.
    }
}

NodePtr make_node(OpKind op, Shape shape, DType dtype, Device device) {
    auto n = std::make_shared<Node>();
    n->op = op;
    n->shape = std::move(shape);
    n->dtype = dtype;
    n->device = device;
    return n;
}

NodePtr make_view(OpKind op, const NodePtr& base, Shape shape, int64_t offset_bytes) {
    VKML_ASSERT(base != nullptr, "view of a null node");
    VKML_ASSERT(is_view_op(op), "make_view called with non-view op '{}'", op_name(op));

    auto n = make_node(op, std::move(shape), base->dtype, base->device);

    // `view_src` and `src[0]` mean different things and must NOT be conflated.
    //
    //   view_src : the STORAGE anchor. Collapsed to the ultimate base so that a
    //              chain of reshape/permute/slice resolves to one buffer plus
    //              one offset, with no linked list for realisation to walk.
    //              This is what ggml's view_src is for.
    //
    //   src[0]   : the DEPENDENCY edge. Kept as the IMMEDIATE base, because
    //              scheduling and autograd both need the real chain.
    //
    // Collapsing src[0] as well is a bug that forward execution cannot see:
    // the view's own Shape already encodes the whole transformation, so values
    // come out right. Backward then breaks, because a rule reads
    // `src[0]->shape` to know what shape to produce a gradient in. For
    // `W.transpose().broadcast_to(...)` it would see W's (out, in) instead of
    // W-transposed's (in, out) and emit a correctly-valued but TRANSPOSED
    // gradient -- which is exactly how this was found, via
    // test_linear_forward_and_backward.
    const NodePtr& root = base->is_view() ? base->view_src : base;
    const int64_t base_offset = base->is_view() ? base->view_offset : 0;

    n->view_src = root;
    n->view_offset = base_offset + offset_bytes;
    n->src[0] = base;
    n->n_src = 1;
    n->requires_grad = grad_enabled() && base->requires_grad;
    return n;
}

std::vector<Node*> topological_order(std::span<const NodePtr> roots) {
    std::vector<Node*> order;
    std::unordered_set<const Node*> done;

    struct Frame {
        Node* node;
        int next_src;
    };

    std::vector<Frame> stack;

    for (const NodePtr& root : roots) {
        if (root == nullptr || done.count(root.get()) != 0) {
            continue;
        }
        stack.push_back({root.get(), 0});

        while (!stack.empty()) {
            Node* node = stack.back().node;

            // Reached again through a second parent, or already emitted.
            if (done.count(node) != 0) {
                stack.pop_back();
                continue;
            }

            // A COMPUTED node is a leaf for scheduling purposes: its value
            // already exists, so nothing behind it needs recomputing.
            //
            // Computed, not bound. A node can hold storage it has not yet
            // written -- an Assign bound to its destination is the case that
            // forced the distinction -- and keying this on binding would skip
            // such a node as though it had already run.
            if (node->is_computed()) {
                done.insert(node);
                stack.pop_back();
                continue;
            }

            if (stack.back().next_src < node->n_src) {
                const int idx = stack.back().next_src;
                // Advance before pushing: push_back may reallocate and
                // invalidate any reference into the stack.
                stack.back().next_src = idx + 1;

                Node* child = node->src[static_cast<size_t>(idx)].get();
                if (child != nullptr && done.count(child) == 0) {
                    stack.push_back({child, 0});
                }
                continue;
            }

            done.insert(node);
            order.push_back(node);
            stack.pop_back();
        }
    }
    return order;
}

std::vector<Node*> topological_order(const NodePtr& root) {
    const std::array<NodePtr, 1> roots{root};
    return topological_order(roots);
}

std::vector<int> compute_use_counts(std::span<Node* const> order) {
    std::unordered_map<const Node*, int> index;
    index.reserve(order.size());
    for (size_t i = 0; i < order.size(); ++i) {
        index.emplace(order[i], static_cast<int>(i));
    }

    std::vector<int> counts(order.size(), 0);
    for (Node* node : order) {
        for (int s = 0; s < node->n_src; ++s) {
            const Node* src = node->src[static_cast<size_t>(s)].get();
            if (src == nullptr) {
                continue;
            }
            if (const auto it = index.find(src); it != index.end()) {
                counts[static_cast<size_t>(it->second)] += 1;
            }
        }
    }
    return counts;
}

std::string to_dot(std::span<const NodePtr> roots) {
    const std::vector<Node*> order = topological_order(roots);

    // topological_order stops at realised nodes, so their sources are absent
    // from `order`. Collect them separately, otherwise the picture would have
    // dangling edges into nodes that were never declared.
    std::unordered_map<const Node*, int> id;
    std::vector<const Node*> all;

    auto intern = [&](const Node* n) {
        const auto [it, inserted] = id.emplace(n, static_cast<int>(all.size()));
        if (inserted) {
            all.push_back(n);
        }
        return it->second;
    };

    for (const Node* node : order) {
        intern(node);
    }
    for (const Node* node : order) {
        for (int s = 0; s < node->n_src; ++s) {
            if (const Node* src = node->src[static_cast<size_t>(s)].get(); src != nullptr) {
                intern(src);
            }
        }
    }

    std::string out = "digraph vkml {\n  rankdir=BT;\n  node [shape=box, fontname=monospace];\n";

    for (const Node* node : all) {
        // Computed nodes are shaded: they are inputs to this evaluation rather
        // than work to be done. Matches what the scheduler treats as a leaf, so
        // the picture and the plan agree.
        const char* style = node->is_computed() ? ", style=filled, fillcolor=lightgrey" : "";
        out += std::format("  n{} [label=\"{}\\n{} {}\"{}];\n", id.at(node), op_name(node->op),
                           node->shape.str(), dtype_name(node->dtype), style);
    }
    for (const Node* node : order) {
        for (int s = 0; s < node->n_src; ++s) {
            if (const Node* src = node->src[static_cast<size_t>(s)].get(); src != nullptr) {
                out += std::format("  n{} -> n{};\n", id.at(src), id.at(node));
            }
        }
    }
    out += "}\n";
    return out;
}

}  // namespace vkml
