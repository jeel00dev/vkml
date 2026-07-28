#include "vkml/dispatch/executor.h"

#include "vkml/backend/api/backend.h"
#include "vkml/graph/graph.h"
#include "vkml/util/assert.h"
#include "vkml/util/coverage.h"
#include "vkml/util/log.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>

namespace vkml {
namespace {

std::atomic<bool>& eager_flag() noexcept {
    static std::atomic<bool> flag{[] {
        const char* env = std::getenv("VKML_EAGER");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }()};
    return flag;
}

/// Gives one scheduled node its storage.
void bind_storage(Node& node) {
    if (node.is_realized()) {
        return;
    }

    if (node.is_view()) {
        // A view owns no memory. It points into its base, which the topological
        // order guarantees was bound first. Holding a shared_ptr to the same
        // Storage is what keeps the base alive even if the base Tensor is
        // dropped -- the reason ggml tracks view_src at all.
        Node& base = *node.view_src;
        VKML_ASSERT(base.is_realized(), "view bound before its base '{}'", op_name(base.op));
        node.storage = base.storage;
        node.storage_offset = base.storage_offset + node.view_offset;
        return;
    }

    VKML_ASSERT(node.shape.is_contiguous(),
                "computed node '{}' must have a contiguous output layout", op_name(node.op));

    Backend& backend = backend_for(node.device);
    node.storage = backend.allocator().allocate(node.shape.nbytes());
    node.storage_offset = 0;
}

/// Describes one node for the coverage recorder.
///
/// Every node reaching this point is about to be evaluated: `topological_order`
/// treats an already-realised node as a leaf and does not emit it, so there is
/// nothing here that has run before. Views and leaves are emitted but compute
/// nothing, and are labelled as such rather than counted as kernel dispatches.
///
/// The input flags describe the SOURCES, not this node's own layout, because
/// that is where the untested path is: a computed node's output is always
/// contiguous (asserted in bind_storage), while its inputs may be transposed,
/// sliced or broadcast, and a kernel that has only ever seen contiguous inputs
/// has never run its strided indexing.
void record_coverage(const Node& node, std::string_view backend_name) {
    coverage::Dispatch d;
    d.op = op_name(node.op);
    d.backend = (node.is_view() || node.is_leaf()) ? "view" : backend_name;
    d.dtype = dtype_name(node.dtype);
    d.ndim = node.shape.ndim();
    d.numel = node.shape.numel();
    d.work_numel = node.shape.numel();
    d.n_src = node.n_src;
    d.strided_output = !node.shape.is_contiguous();

    for (int i = 0; i < node.n_src; ++i) {
        const Node* src = node.src[static_cast<size_t>(i)].get();
        if (src == nullptr) {
            continue;
        }
        d.broadcast_input |= src->shape.has_broadcast_stride();
        d.strided_input |= !src->shape.is_contiguous();
        d.work_numel = std::max(d.work_numel, src->shape.numel());
    }

    coverage::record(d);
}

}  // namespace

void set_eager(bool enabled) noexcept { eager_flag().store(enabled, std::memory_order_relaxed); }

bool eager() noexcept { return eager_flag().load(std::memory_order_relaxed); }

void realize(std::span<const NodePtr> roots) {
    const std::vector<Node*> order = topological_order(roots);
    if (order.empty()) {
        return;
    }

    // All nodes in one graph share a device in M0; multi-device splitting is
    // the scheduler's job and arrives with the Vulkan backend.
    const Device device = order.front()->device;
    for (Node* n : order) {
        VKML_CHECK(n->device == device, DeviceError,
                   "graph spans devices '{}' and '{}'; cross-device execution is not implemented",
                   device.str(), n->device.str());
    }

    Backend& backend = backend_for(device);

    for (Node* n : order) {
        if (!backend.supports(*n)) {
            throw NotImplementedError(std::format("backend '{}' cannot evaluate op '{}'",
                                                  backend.name(), op_name(n->op)));
        }
        if (coverage::enabled()) {
            record_coverage(*n, backend.name());
        }
        bind_storage(*n);
    }

    VKML_LOG_DEBUG("realize: {} nodes on {}", order.size(), device.str());
    backend.compute(order);
}

void realize(const NodePtr& root) {
    const std::array<NodePtr, 1> roots{root};
    realize(roots);
}

}  // namespace vkml
