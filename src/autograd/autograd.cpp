#include "vkml/autograd/autograd.h"

#include "vkml/api/ops.h"
#include "vkml/graph/node.h"
#include "vkml/util/assert.h"

#include <atomic>
#include <format>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vkml {
namespace {

/// Full traversal from `root`, in dependency order.
///
/// Deliberately NOT graph::topological_order, which stops at realised nodes.
/// Backward needs the whole structure regardless of what has already been
/// computed -- in eager mode every node is realised, and stopping there would
/// find no graph at all.
std::vector<NodePtr> autograd_order(const NodePtr& root) {
    std::vector<NodePtr> order;
    std::unordered_set<const Node*> done;

    struct Frame {
        NodePtr node;
        int next;
    };

    std::vector<Frame> stack;
    stack.push_back({root, 0});

    while (!stack.empty()) {
        Node* raw = stack.back().node.get();
        if (done.count(raw) != 0) {
            stack.pop_back();
            continue;
        }
        if (stack.back().next < raw->n_src) {
            const int idx = stack.back().next;
            stack.back().next = idx + 1;
            const NodePtr& child = raw->src[static_cast<size_t>(idx)];
            // Prune subgraphs that cannot contribute a gradient. This is what
            // keeps backward proportional to the differentiable part of the
            // graph rather than to the whole thing.
            if (child != nullptr && child->requires_grad && done.count(child.get()) == 0) {
                stack.push_back({child, 0});
            }
            continue;
        }
        done.insert(raw);
        order.push_back(stack.back().node);
        stack.pop_back();
    }
    return order;
}

using GradMap = std::unordered_map<Node*, Tensor>;

void accumulate(GradMap& grads, const NodePtr& node, const Tensor& g) {
    if (node == nullptr || !node->requires_grad) {
        return;
    }
    auto it = grads.find(node.get());
    if (it == grads.end()) {
        grads.emplace(node.get(), g);
    } else {
        // A node consumed by several others receives one contribution per
        // consumer; the total derivative is their sum (the multivariate chain
        // rule). This is the only place fan-out is handled, which is why
        // diamond-shaped graphs need no special case.
        it->second = it->second + g;
    }
}

/// Sums `g` down to `target_dims`, undoing a broadcast.
///
/// Broadcasting is an explicit Broadcast node in the forward graph, so its
/// adjoint is an ordinary reduction here rather than logic smeared across every
/// binary op's backward rule.
Tensor reduce_to_shape(const Tensor& g, const std::vector<int64_t>& target_dims) {
    const std::vector<int64_t> gd = g.shape();
    if (gd == target_dims) {
        return g;
    }

    const size_t lead = gd.size() - target_dims.size();
    std::vector<int> axes;

    for (size_t i = 0; i < gd.size(); ++i) {
        if (i < lead) {
            axes.push_back(static_cast<int>(i));  // axis did not exist in the source
        } else if (target_dims[i - lead] == 1 && gd[i] != 1) {
            axes.push_back(static_cast<int>(i));  // axis was stretched from 1
        }
    }

    Tensor r = axes.empty() ? g : sum(g, axes, /*keepdim=*/true);
    return r.reshape(target_dims);
}

[[noreturn]] void not_differentiable(const Node& node) {
    throw NotImplementedError(std::format(
        "no gradient rule for op '{}'; it is either non-differentiable or not yet implemented",
        op_name(node.op)));
}

/// Appends the adjoint computation for one node.
void apply_backward(const NodePtr& np, const Tensor& grad, GradMap& grads) {
    Node& node = *np;
    const NodePtr& a = node.src[0];
    const NodePtr& b = node.src[1];

    auto ta = [&] { return Tensor{a}; };
    auto tb = [&] { return Tensor{b}; };
    auto out = [&] { return Tensor{np}; };

    switch (node.op) {
        // -- structural -----------------------------------------------------
        case OpKind::Contiguous:
        case OpKind::Reshape:
        case OpKind::Squeeze:
        case OpKind::Unsqueeze:
            // Pure relabelling: the adjoint is the gradient reshaped back.
            accumulate(grads, a, grad.reshape(Tensor{a}.shape()));
            return;

        case OpKind::Permute: {
            // Undo the permutation. The forward permutation is recorded in
            // params precisely so the inverse is recoverable -- deriving it
            // from strides would be ambiguous when strides repeat.
            const auto p = node.params.get<PermuteParams>();
            const int nd = node.shape.ndim();
            std::vector<int> inverse(static_cast<size_t>(nd));
            for (int i = 0; i < nd; ++i) {
                inverse[static_cast<size_t>(p.perm[static_cast<size_t>(i)])] = i;
            }
            accumulate(grads, a, grad.permute(inverse));
            return;
        }

        case OpKind::Slice: {
            // Scatter the gradient back into a zero tensor of the original
            // extent. See k_slice_backward for why this needs a kernel.
            const Tensor base = ta();
            auto n = make_node(OpKind::SliceBackward,
                               Shape::contiguous(base.shape(), dtype_size(base.dtype())),
                               base.dtype(), base.device());
            n->src[0] = grad.contiguous().node();
            n->n_src = 1;
            n->params.set(node.params.get<SliceParams>());
            Tensor scattered{std::move(n)};
            scattered.realize();
            accumulate(grads, a, scattered);
            return;
        }

        case OpKind::Broadcast:
            accumulate(grads, a, reduce_to_shape(grad, Tensor{a}.shape()));
            return;

        case OpKind::Cast: accumulate(grads, a, grad.to(a->dtype)); return;

        // -- binary ---------------------------------------------------------
        case OpKind::Add:
            accumulate(grads, a, grad);
            accumulate(grads, b, grad);
            return;

        case OpKind::Sub:
            accumulate(grads, a, grad);
            accumulate(grads, b, neg(grad));
            return;

        case OpKind::Mul:
            accumulate(grads, a, grad * tb());
            accumulate(grads, b, grad * ta());
            return;

        case OpKind::Div:
            accumulate(grads, a, grad / tb());
            // d/db (a/b) = -a/b^2. Written as -(grad * out) / b to reuse the
            // already-computed quotient instead of squaring b.
            accumulate(grads, b, neg(grad * out()) / tb());
            return;

        case OpKind::Pow:
            // d/da a^b = b * a^(b-1). Only the base is differentiated; the
            // exponent's gradient needs log(a), which is undefined for a <= 0
            // and is not needed by anything in scope.
            accumulate(grads, a, grad * tb() * pow(ta(), sub(tb(), 1.0)));
            if (b != nullptr && b->requires_grad) {
                throw NotImplementedError(
                    "gradient with respect to the exponent of pow() is not implemented");
            }
            return;

        case OpKind::Maximum:
            accumulate(grads, a, where(greater_equal(ta(), tb()), grad, zeros_like(grad)));
            accumulate(grads, b, where(less(ta(), tb()), grad, zeros_like(grad)));
            return;

        case OpKind::Minimum:
            accumulate(grads, a, where(less_equal(ta(), tb()), grad, zeros_like(grad)));
            accumulate(grads, b, where(greater(ta(), tb()), grad, zeros_like(grad)));
            return;

        case OpKind::Where:
            // src0 is the condition and is Bool, hence never differentiable.
            accumulate(grads, node.src[1], where(ta(), grad, zeros_like(grad)));
            accumulate(grads, node.src[2], where(ta(), zeros_like(grad), grad));
            return;

        // -- unary ----------------------------------------------------------
        case OpKind::Neg: accumulate(grads, a, neg(grad)); return;

        case OpKind::Abs: accumulate(grads, a, grad * sign(ta())); return;

        case OpKind::Square: accumulate(grads, a, grad * ta() * 2.0); return;

        case OpKind::Sqrt:
            // 1/(2*sqrt(a)) reuses the forward result rather than recomputing
            // the root.
            accumulate(grads, a, grad / (out() * 2.0));
            return;

        case OpKind::Rsqrt:
            // d/da a^-1/2 = -1/2 a^-3/2 = -out^3 / 2
            accumulate(grads, a, neg(grad * out() * out() * out()) / 2.0);
            return;

        case OpKind::Reciprocal: accumulate(grads, a, neg(grad * out() * out())); return;

        case OpKind::Exp: accumulate(grads, a, grad * out()); return;

        case OpKind::Log: accumulate(grads, a, grad / ta()); return;

        case OpKind::Sin: accumulate(grads, a, grad * cos(ta())); return;

        case OpKind::Cos: accumulate(grads, a, neg(grad * sin(ta()))); return;

        case OpKind::Tanh:
            accumulate(grads, a,
                       grad * sub(Tensor::ones(node.shape.dims(), node.dtype, node.device),
                                  out() * out()));
            return;

        case OpKind::Sigmoid:
            accumulate(grads, a,
                       grad * out() *
                           sub(Tensor::ones(node.shape.dims(), node.dtype, node.device), out()));
            return;

        // im2col and col2im are each other's adjoint, exactly as index_select
        // and scatter_add are: extracting windows is linear, and the transpose
        // of an extraction is a sum back into the positions it drew from.
        case OpKind::Im2Col: {
            const auto p = node.params.get<UnfoldParams>();
            accumulate(grads, a,
                       col2im(grad, {p.image_h, p.image_w}, {p.kernel_h, p.kernel_w},
                              {p.stride_h, p.stride_w}, {p.pad_h, p.pad_w},
                              {p.dilation_h, p.dilation_w}));
            return;
        }

        case OpKind::Col2Im: {
            const auto p = node.params.get<UnfoldParams>();
            accumulate(grads, a,
                       im2col(grad, {p.kernel_h, p.kernel_w}, {p.stride_h, p.stride_w},
                              {p.pad_h, p.pad_w}, {p.dilation_h, p.dilation_w}));
            return;
        }

        // index_select and scatter_add are each other's adjoint, which is the
        // whole reason scatter_add exists as an operator: gathering rows is
        // linear, and the transpose of a gather is a scatter-with-accumulation.
        // Repeated indices are exactly why the accumulation cannot be dropped.
        case OpKind::IndexSelect: {
            const int axis = node.params.get<AxisParams>().axis;
            const Tensor index{node.src[1]};
            accumulate(grads, a, scatter_add(grad, axis, index, node.src[0]->shape.dim(axis)));
            return;
        }

        case OpKind::ScatterAdd: {
            const int axis = node.params.get<AxisParams>().axis;
            const Tensor index{node.src[1]};
            accumulate(grads, a, index_select(grad, axis, index));
            return;
        }

        // Concatenation is a permutation of elements, so its adjoint is the
        // inverse permutation: each operand takes back exactly the slice of the
        // gradient it contributed. Slice is a view, so this allocates nothing
        // and needs no new kernel.
        case OpKind::Cat: {
            const int axis = node.params.get<AxisParams>().axis;
            const int64_t split = node.src[0]->shape.dim(axis);
            const int64_t total = node.shape.dim(axis);
            if (node.src[0]->requires_grad) {
                accumulate(grads, node.src[0], grad.slice(axis, 0, split));
            }
            if (node.src[1]->requires_grad) {
                accumulate(grads, node.src[1], grad.slice(axis, split, total));
            }
            return;
        }

        // A triangular mask is linear and idempotent, so its own transpose is
        // itself: whatever the mask zeroed contributed nothing to the output
        // and must receive no gradient. Applying the same mask to `grad` is the
        // whole rule -- no new kernel, per the graph-autograd design.
        case OpKind::Triu:
            accumulate(grads, a, triu(grad, node.params.get<TriParams>().diagonal));
            return;

        case OpKind::Tril:
            accumulate(grads, a, tril(grad, node.params.get<TriParams>().diagonal));
            return;

        case OpKind::Relu:
            accumulate(grads, a, where(greater(ta(), zeros_like(ta())), grad, zeros_like(grad)));
            return;

        case OpKind::Gelu: {
            // d/dx [0.5x(1+erf(x/sqrt2))] = 0.5(1+erf(x/sqrt2)) + x*phi(x)
            // where phi is the standard normal pdf. Expressed with existing ops:
            // the cdf term is out/x, but that is singular at 0, so it is
            // recomputed from erf via the forward gelu of a unit input instead.
            constexpr double kInvSqrt2 = 0.70710678118654752440;
            constexpr double kInvSqrt2Pi = 0.39894228040143267794;
            const Tensor x = ta();
            const Tensor cdf = mul(add(erf(mul(x, kInvSqrt2)), 1.0), 0.5);
            const Tensor pdf = mul(exp(mul(square(x), -0.5)), kInvSqrt2Pi);
            accumulate(grads, a, grad * add(cdf, x * pdf));
            return;
        }

        case OpKind::Silu: {
            // d/dx [x*sigmoid(x)] = sigmoid(x) * (1 + x*(1 - sigmoid(x)))
            const Tensor s = sigmoid(ta());
            const Tensor one = Tensor::ones(node.shape.dims(), node.dtype, node.device);
            accumulate(grads, a, grad * s * add(one, ta() * sub(one, s)));
            return;
        }

        case OpKind::Clamp: {
            const auto p = node.params.get<ClampParams>();
            Tensor mask = Tensor::ones(node.shape.dims(), node.dtype, node.device);
            if (p.has_lo) {
                mask = mask * where(greater_equal(ta(), full_like(ta(), static_cast<double>(p.lo))),
                                    Tensor::ones(node.shape.dims(), node.dtype, node.device),
                                    zeros_like(mask));
            }
            if (p.has_hi) {
                mask = mask * where(less_equal(ta(), full_like(ta(), static_cast<double>(p.hi))),
                                    Tensor::ones(node.shape.dims(), node.dtype, node.device),
                                    zeros_like(mask));
            }
            accumulate(grads, a, grad * mask);
            return;
        }

        // -- reductions -----------------------------------------------------
        case OpKind::Sum: {
            // The adjoint of a sum is a broadcast of the gradient back over the
            // reduced axes.
            const auto p = node.params.get<ReduceParams>();
            const std::vector<int64_t> src_dims = Tensor{a}.shape();
            Tensor g = grad;
            if (!p.keepdim) {
                // Re-insert the collapsed axes so broadcast_to can line up.
                std::vector<int64_t> keep = src_dims;
                for (size_t i = 0; i < keep.size(); ++i) {
                    if ((p.axes_mask & (1U << static_cast<uint32_t>(i))) != 0) {
                        keep[i] = 1;
                    }
                }
                g = g.reshape(keep);
            }
            accumulate(grads, a, g.broadcast_to(src_dims).contiguous());
            return;
        }

        case OpKind::Mean: {
            const auto p = node.params.get<ReduceParams>();
            const std::vector<int64_t> src_dims = Tensor{a}.shape();
            int64_t count = 1;
            for (size_t i = 0; i < src_dims.size(); ++i) {
                if ((p.axes_mask & (1U << static_cast<uint32_t>(i))) != 0) {
                    count *= src_dims[i];
                }
            }
            Tensor g = grad / static_cast<double>(count);
            if (!p.keepdim) {
                std::vector<int64_t> keep = src_dims;
                for (size_t i = 0; i < keep.size(); ++i) {
                    if ((p.axes_mask & (1U << static_cast<uint32_t>(i))) != 0) {
                        keep[i] = 1;
                    }
                }
                g = g.reshape(keep);
            }
            accumulate(grads, a, g.broadcast_to(src_dims).contiguous());
            return;
        }

        case OpKind::Max:
        case OpKind::Min: {
            // Gradient flows only to the extremal elements. Ties split the
            // gradient in PyTorch's `amax`/`amin`; here it is routed to every
            // tied element, which differs when there are ties. Documented as a
            // known divergence rather than silently different.
            const auto p = node.params.get<ReduceParams>();
            const std::vector<int64_t> src_dims = Tensor{a}.shape();
            std::vector<int64_t> keep = src_dims;
            for (size_t i = 0; i < keep.size(); ++i) {
                if ((p.axes_mask & (1U << static_cast<uint32_t>(i))) != 0) {
                    keep[i] = 1;
                }
            }
            const Tensor out_k = p.keepdim ? out() : out().reshape(keep);
            const Tensor g_k = p.keepdim ? grad : grad.reshape(keep);
            const Tensor mask = equal(ta(), out_k.broadcast_to(src_dims));
            accumulate(grads, a,
                       where(mask, g_k.broadcast_to(src_dims).contiguous(), zeros_like(Tensor{a})));
            return;
        }

        // -- composite ------------------------------------------------------
        case OpKind::Softmax: {
            // dL/dx = y * (g - sum(g*y, axis, keepdim))
            const auto p = node.params.get<AxisParams>();
            const std::array<int, 1> axes{p.axis};
            const Tensor y = out();
            const Tensor dot = sum(grad * y, axes, /*keepdim=*/true);
            accumulate(grads, a, y * sub(grad, dot.broadcast_to(Tensor{a}.shape())));
            return;
        }

        case OpKind::LogSoftmax: {
            // dL/dx = g - exp(y) * sum(g, axis, keepdim)
            const auto p = node.params.get<AxisParams>();
            const std::array<int, 1> axes{p.axis};
            const Tensor total = sum(grad, axes, /*keepdim=*/true);
            accumulate(grads, a, sub(grad, exp(out()) * total.broadcast_to(Tensor{a}.shape())));
            return;
        }

        // -- linear algebra -------------------------------------------------
        case OpKind::Matmul:
            // dA = g @ B^T, dB = A^T @ g. Both are matmuls of transposed
            // views -- no new kernel, no copy.
            accumulate(grads, a, matmul(grad, tb().transpose(-2, -1)));
            accumulate(grads, b, matmul(ta().transpose(-2, -1), grad));
            return;

        default: not_differentiable(node);
    }
}

}  // namespace

NoGradGuard::NoGradGuard() noexcept : previous_(grad_enabled()) { set_grad_enabled(false); }

NoGradGuard::~NoGradGuard() { set_grad_enabled(previous_); }

Tensor detach(const Tensor& t) {
    VKML_CHECK(t.defined(), Error, "detach() on an undefined tensor");
    // Share storage, drop history: a fresh leaf pointing at the same buffer.
    const NodePtr& src = t.node();
    auto n = make_node(OpKind::Input, src->shape, src->dtype, src->device);
    n->storage = src->storage;
    n->storage_offset = src->storage_offset;
    n->requires_grad = false;
    if (!src->is_realized()) {
        // Nothing to share yet -- evaluate first so the detached view is real.
        t.realize();
        n->storage = src->storage;
        n->storage_offset = src->storage_offset;
    }
    return Tensor{n};
}

void backward(const Tensor& root, const Tensor& seed) {
    VKML_CHECK(root.defined(), Error, "backward() on an undefined tensor");
    VKML_CHECK(root.requires_grad(), Error,
               "backward() called on a tensor that does not require grad");
    VKML_CHECK(is_differentiable(root.dtype()), DTypeError,
               "backward() requires a floating tensor, got {}", dtype_name(root.dtype()));

    const std::vector<NodePtr> order = autograd_order(root.node());

    GradMap grads;
    grads.emplace(root.node().get(), seed);

    // Reverse dependency order: a node's gradient is complete only once every
    // consumer has contributed, and consumers all appear later in `order`.
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const NodePtr& np = *it;
        const auto found = grads.find(np.get());
        if (found == grads.end()) {
            continue;  // unreachable from the root's gradient
        }
        if (np->is_leaf()) {
            continue;  // leaves receive, they do not propagate
        }
        apply_backward(np, found->second, grads);
    }

    // Deposit into leaves, accumulating rather than replacing (PyTorch's rule,
    // and what makes gradient accumulation across micro-batches work).
    for (const NodePtr& np : order) {
        if (!np->is_leaf() || !np->requires_grad) {
            continue;
        }
        const auto found = grads.find(np.get());
        if (found == grads.end()) {
            continue;
        }
        Tensor total = np->grad ? (Tensor{np->grad} + found->second) : found->second;
        total.realize();
        np->grad = total.node();
    }
}

void backward(const Tensor& root) {
    VKML_CHECK(root.numel() == 1, ShapeError,
               "backward() without an explicit gradient requires a scalar, got shape {}",
               root.node()->shape.str());
    backward(root, Tensor::ones(root.node()->shape.dims(), root.dtype(), root.device()));
}

}  // namespace vkml
