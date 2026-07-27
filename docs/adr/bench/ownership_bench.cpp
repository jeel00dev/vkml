// Measures the actual cost of the three graph-node ownership models for the
// shapes vkml really sees, so the design review is evidence-based.
//
// Workload model: a training-step graph is ~500-5000 nodes (forward + backward
// + optimizer). Built once per step in the uncompiled path.

#include "vkml/graph/graph.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

using namespace vkml;

static double ms_since(std::chrono::steady_clock::time_point t0) {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// Option A: shared_ptr (current implementation)
// ---------------------------------------------------------------------------
static double bench_shared_ptr_build(int n, int reps) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
        std::vector<int64_t> dims{64, 128};
        auto cur = make_node(OpKind::Input, Shape::contiguous(dims, 4), DType::F32, Device::cpu());
        for (int i = 0; i < n; ++i) {
            auto nx = make_node(OpKind::Relu, cur->shape, cur->dtype, cur->device);
            nx->src[0] = cur;
            nx->n_src = 1;
            cur = nx;
        }
    }
    return ms_since(t0) / reps;
}

static double bench_shared_ptr_traverse(int n, int reps) {
    std::vector<int64_t> dims{64, 128};
    auto cur = make_node(OpKind::Input, Shape::contiguous(dims, 4), DType::F32, Device::cpu());
    for (int i = 0; i < n; ++i) {
        auto nx = make_node(OpKind::Relu, cur->shape, cur->dtype, cur->device);
        nx->src[0] = cur;
        nx->n_src = 1;
        cur = nx;
    }
    const auto t0 = std::chrono::steady_clock::now();
    size_t total = 0;
    for (int r = 0; r < reps; ++r) {
        total += topological_order(cur).size();
    }
    const double t = ms_since(t0) / reps;
    if (total == 0) {
        std::printf("unreachable\n");
    }
    return t;
}

// ---------------------------------------------------------------------------
// Option B: arena + 32-bit handles
// ---------------------------------------------------------------------------
using NodeId = uint32_t;
inline constexpr NodeId kNoNode = 0xFFFFFFFFU;

struct ArenaNode {
    OpKind op;
    DType dtype;
    Device device;
    Shape shape;
    OpParams params;
    std::array<NodeId, kMaxSrc> src{kNoNode, kNoNode, kNoNode, kNoNode};
    int n_src = 0;
    NodeId view_src = kNoNode;
    int64_t view_offset = 0;
    uint32_t flags = 0;
    NodeId grad = kNoNode;
    int32_t storage_index = -1;
    int64_t storage_offset = 0;
    bool requires_grad = false;
};

struct Arena {
    std::vector<ArenaNode> nodes;

    NodeId add(OpKind op, const Shape& s, DType dt, Device dev) {
        ArenaNode n;
        n.op = op;
        n.shape = s;
        n.dtype = dt;
        n.device = dev;
        nodes.push_back(n);
        return static_cast<NodeId>(nodes.size() - 1);
    }
};

static double bench_arena_build(int n, int reps) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
        Arena a;
        a.nodes.reserve(static_cast<size_t>(n) + 1);
        std::vector<int64_t> dims{64, 128};
        NodeId cur = a.add(OpKind::Input, Shape::contiguous(dims, 4), DType::F32, Device::cpu());
        for (int i = 0; i < n; ++i) {
            const Shape s = a.nodes[cur].shape;
            NodeId nx = a.add(OpKind::Relu, s, DType::F32, Device::cpu());
            a.nodes[nx].src[0] = cur;
            a.nodes[nx].n_src = 1;
            cur = nx;
        }
    }
    return ms_since(t0) / reps;
}

static double bench_arena_traverse(int n, int reps) {
    Arena a;
    a.nodes.reserve(static_cast<size_t>(n) + 1);
    std::vector<int64_t> dims{64, 128};
    NodeId cur = a.add(OpKind::Input, Shape::contiguous(dims, 4), DType::F32, Device::cpu());
    for (int i = 0; i < n; ++i) {
        const Shape s = a.nodes[cur].shape;
        NodeId nx = a.add(OpKind::Relu, s, DType::F32, Device::cpu());
        a.nodes[nx].src[0] = cur;
        a.nodes[nx].n_src = 1;
        cur = nx;
    }

    const auto t0 = std::chrono::steady_clock::now();
    size_t total = 0;
    for (int r = 0; r < reps; ++r) {
        // Same algorithm shape as topological_order, but over indices.
        std::vector<uint8_t> done(a.nodes.size(), 0);
        std::vector<NodeId> order;
        order.reserve(a.nodes.size());
        std::vector<std::pair<NodeId, int>> stack;
        stack.emplace_back(cur, 0);
        while (!stack.empty()) {
            auto [id, next] = stack.back();
            if (done[id]) {
                stack.pop_back();
                continue;
            }
            if (next < a.nodes[id].n_src) {
                stack.back().second = next + 1;
                const NodeId child = a.nodes[id].src[static_cast<size_t>(next)];
                if (child != kNoNode && !done[child]) {
                    stack.emplace_back(child, 0);
                }
                continue;
            }
            done[id] = 1;
            order.push_back(id);
            stack.pop_back();
        }
        total += order.size();
    }
    const double t = ms_since(t0) / reps;
    if (total == 0) {
        std::printf("unreachable\n");
    }
    return t;
}

int main() {
    std::printf("sizeof(vkml::Node)        = %zu bytes\n", sizeof(Node));
    std::printf("sizeof(ArenaNode)         = %zu bytes\n", sizeof(ArenaNode));
    std::printf("sizeof(Shape)             = %zu bytes\n", sizeof(Shape));
    std::printf("sizeof(OpParams)          = %zu bytes\n", sizeof(OpParams));
    std::printf("sizeof(shared_ptr<Node>)  = %zu bytes\n", sizeof(std::shared_ptr<Node>));
    std::printf("\n");

    std::printf("%-10s %14s %14s %14s %14s\n", "nodes", "A build(ms)", "B build(ms)",
                "A trav(ms)", "B trav(ms)");
    for (const int n : {500, 5000, 50000}) {
        const int reps = n <= 5000 ? 200 : 20;
        const double ab = bench_shared_ptr_build(n, reps);
        const double bb = bench_arena_build(n, reps);
        const double at = bench_shared_ptr_traverse(n, reps);
        const double bt = bench_arena_traverse(n, reps);
        std::printf("%-10d %14.4f %14.4f %14.4f %14.4f\n", n, ab, bb, at, bt);
    }

    // Put the numbers in context: what does one elementwise op on a realistic
    // tensor actually cost?
    std::printf("\ncontext: one relu over 64x128 f32 = %d elements\n", 64 * 128);
    std::vector<float> x(64 * 128, 0.5F);
    std::vector<float> y(64 * 128);
    const auto t0 = std::chrono::steady_clock::now();
    const int kreps = 20000;
    for (int r = 0; r < kreps; ++r) {
        for (size_t i = 0; i < x.size(); ++i) {
            y[i] = x[i] > 0 ? x[i] : 0.0F;
        }
    }
    std::printf("  measured %.5f ms per op (compute per node)\n", ms_since(t0) / kreps);
    return 0;
}
