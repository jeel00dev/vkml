#pragma once

#include "vkml/core/device.h"
#include "vkml/core/dtype.h"
#include "vkml/core/shape.h"
#include "vkml/core/storage.h"
#include "vkml/graph/op.h"

#include <array>
#include <cstdint>
#include <memory>

namespace vkml {

enum NodeFlags : uint32_t {
    kFlagNone = 0,
    kFlagParam = 1U << 0,   ///< trainable parameter; a leaf that accumulates .grad
    kFlagOutput = 1U << 1,  ///< graph output; the memory planner must not reuse it
    kFlagLoss = 1U << 2,    ///< scalar the backward pass is seeded from
};

/// One vertex of the computation DAG.
///
/// LIFETIME -- note the deviation from docs/ARCHITECTURE.md §4.2
/// ------------------------------------------------------------
/// The architecture sketch shows `std::array<Node*, 4> src` with nodes
/// "arena-allocated", mirroring ggml, where a `ggml_context` owns every node
/// and raw pointers are safe because the context outlives them all.
///
/// vkml uses `shared_ptr` for `src` instead, because it has no equivalent
/// arena. Graphs here are built incrementally from Python, node by node, with
/// no natural scope that owns them; a Python-held Tensor must keep its whole
/// producing subgraph alive on its own. Raw pointers would need either an arena
/// with a lifetime tied to something (there is nothing to tie it to) or manual
/// refcounting, which is what shared_ptr already is.
///
/// This cannot create reference cycles: the DAG is built strictly bottom-up, so
/// a node's sources always predate it and can never point back.
///
/// The measured cost is ~410 ns/node to build and ~460 ns/node to traverse,
/// against ~1000 ns of actual compute for a modest elementwise op -- under 1 %
/// of step time at M0-M4 graph sizes. An arena is 20-64x faster on both counts,
/// but cannot express Python's unpredictable object lifetimes without
/// reintroducing refcounting. The resolution is to lower this graph into a flat
/// arena-backed `ExecGraph` at M5, where planning and execution get the arena's
/// locality and this layer keeps its safety.
///
/// Full analysis, measurements and rejected alternatives:
/// docs/adr/0001-graph-ownership-and-ir.md.
///
/// IMMUTABILITY
/// ------------
/// A Node is immutable once constructed, with exactly one exception: the
/// realisation fields `storage` and `storage_offset`, which are filled in when
/// the node is evaluated.
///
/// This is not a stylistic preference, it is what makes three things sound:
///   - two Tensors may share a node, so a mutation would be action at a distance;
///   - an optimization pass may assume a node never changes under it, and so can
///     rewrite by building new nodes rather than by patching old ones;
///   - lowering to ExecGraph is a pure function of the graph, hence cacheable.
///
/// Pass results therefore live in side-tables keyed by position in the
/// topological order -- never as new mutable fields here. `compute_use_counts`
/// is the pattern to follow.
struct Node {
    OpKind op = OpKind::Input;
    Shape shape;
    DType dtype = DType::F32;
    Device device;
    OpParams params;

    std::array<std::shared_ptr<Node>, kMaxSrc> src;
    int n_src = 0;

    /// For view ops, the node whose storage this one aliases, plus the byte
    /// offset into it. `view_src` is always also `src[0]`, so graph traversal
    /// needs only one dependency mechanism -- the same arrangement ggml uses.
    std::shared_ptr<Node> view_src;
    int64_t view_offset = 0;

    uint32_t flags = kFlagNone;

    // -- filled in by realisation -------------------------------------------
    std::shared_ptr<Storage> storage;
    int64_t storage_offset = 0;

    // -- autograd ------------------------------------------------------------
    bool requires_grad = false;

    /// Accumulated gradient. Only leaves marked kFlagParam retain one, matching
    /// PyTorch, where intermediate .grad is dropped unless explicitly retained.
    std::shared_ptr<Node> grad;

    Node() = default;

    /// Tears the source chain down iteratively.
    ///
    /// Default destruction would recurse: destroying a node releases its
    /// `src` shared_ptrs, which destroys those nodes, which release theirs.
    /// On a deep graph that is a stack overflow -- and "deep graph" is not
    /// hypothetical here, since an unrolled RNN over a long sequence is a
    /// chain 10^5 nodes long. See the deep-chain case in tests/cpp/test_graph.cpp,
    /// which segfaults without this.
    ///
    /// Costs nothing for leaves: the worklist vector never allocates unless
    /// there is actually a source to release.
    ///
    /// Assumes graph teardown is single-threaded, which it is -- the use_count
    /// check below would otherwise be racy.
    ~Node();

    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;
    Node(Node&&) = delete;
    Node& operator=(Node&&) = delete;

    [[nodiscard]] bool is_realized() const noexcept { return storage != nullptr; }

    [[nodiscard]] bool is_leaf() const noexcept { return is_leaf_op(op); }

    [[nodiscard]] bool is_view() const noexcept { return view_src != nullptr; }

    /// Address of element zero. Only valid once realised.
    [[nodiscard]] void* data() noexcept {
        if (storage == nullptr) {
            return nullptr;
        }
        return static_cast<std::byte*>(storage->data()) + storage_offset;
    }

    [[nodiscard]] const void* data() const noexcept {
        if (storage == nullptr) {
            return nullptr;
        }
        return static_cast<const std::byte*>(storage->data()) + storage_offset;
    }
};

using NodePtr = std::shared_ptr<Node>;

/// Creates an unrealised node. Sources are attached separately so that callers
/// can validate shapes first.
[[nodiscard]] NodePtr make_node(OpKind op, Shape shape, DType dtype, Device device);

/// Creates a view node aliasing `base` at `offset_bytes`.
/// Sets both `view_src` and `src[0]`, keeping the two in sync by construction.
[[nodiscard]] NodePtr make_view(OpKind op, const NodePtr& base, Shape shape, int64_t offset_bytes);

}  // namespace vkml
