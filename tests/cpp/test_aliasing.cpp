// Storage-binding and aliasing assumptions, tested before stage B of
// docs/adr/0006-lazy-assign-and-submission-batching.md is built on them.
//
// Stage B turns assign into a graph node whose destination is an EXISTING
// buffer. That is the first time vkML would schedule a node which does not own
// the memory it writes, and it makes two assumptions that had been read but
// never run. This file runs them.
//
// Both backends, because the answers differ: the CPU backend executes nodes in
// order on one thread and cannot race, while the Vulkan backend records a
// command buffer whose dispatches the GPU may overlap.

#include "doctest.h"
#include "test_support.h"

#include "vkml/api/ops.h"
#include "vkml/api/tensor.h"
#include "vkml/backend/api/backend.h"
#include "vkml/dispatch/executor.h"
#include "vkml/graph/graph.h"
#include "vkml/graph/node.h"

#ifdef VKML_HAS_VULKAN
#    include "vkml/backend/vulkan/vulkan_backend.h"
#endif

#include <algorithm>
#include <numeric>
#include <vector>

using vkml::Backend;
using vkml::Device;
using vkml::DType;
using vkml::Node;
using vkml::NodePtr;
using vkml::OpKind;
using vkml::Shape;
using vkml::Tensor;

namespace {

std::vector<float> host(const Tensor& t) {
    std::vector<float> out(static_cast<size_t>(t.numel()));
    t.to_host(out.data());
    return out;
}

Tensor from(std::vector<int64_t> dims, std::vector<float> values, Device dev = Device::cpu()) {
    return Tensor::from_host(values.data(), dims, DType::F32, dev);
}

/// Schedules `root` and binds storage to every node in it, returning the order
/// so a caller can alias one node's storage before computing.
///
/// This duplicates executor.cpp's `bind_storage`, which is normally a smell.
/// It is the point here: the executor gives every computed node FRESH storage,
/// so the aliasing these tests are about cannot be produced through it. Binding
/// by hand is the only way to build the case stage B would introduce.
///
/// Views are bound to their base, as bind_storage does -- `mul(a, 2.0)` expands
/// to a Full node and a Broadcast view of it, so this is not a corner case but
/// the ordinary shape of a scalar operand.
std::vector<Node*> bind_all(const Tensor& root, Backend& backend) {
    std::vector<Node*> order = vkml::topological_order(root.node());
    for (Node* n : order) {
        if (n->is_bound()) {
            continue;
        }
        if (n->is_view()) {
            n->storage = n->view_src->storage;
            n->storage_offset = n->view_src->storage_offset + n->view_offset;
            continue;
        }
        n->storage = backend.allocator().allocate(n->shape.nbytes());
    }
    return order;
}

std::vector<float> read_back(Backend& backend, const Node& n, size_t count) {
    std::vector<float> out(count);
    backend.copy_to_host(out.data(), *n.storage, n.storage_offset, count * sizeof(float));
    return out;
}

/// Every device the suite can exercise. Vulkan is absent rather than failing
/// when no GPU is present, so the same binary is meaningful on a CI runner.
std::vector<Device> devices() {
    std::vector<Device> out{Device::cpu()};
#ifdef VKML_HAS_VULKAN
    if (vkml::vulkan_available()) {
        (void)vkml::vulkan_backend(0);  // registers it, so backend_for() resolves
        out.push_back(Device::vulkan(0));
    }
#endif
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Assumption 1 -- storage binding
//
// executor.cpp bind_storage() allocates fresh storage for every computed node.
// An Assign node must instead write into its DESTINATION's existing storage.
// The question was whether that can be expressed.
// ---------------------------------------------------------------------------

TEST_CASE("binding storage does not make a node look computed") {
    // The B0 finding, and now its fix. `is_realized()` used to be DEFINED as
    // `storage != nullptr`, and topological_order() treated such a node as a
    // leaf -- so binding an Assign node to its destination's storage made the
    // scheduler skip it: it looked computed before it had run. Measured at the
    // time: order size 3 while unbound, 0 once bound.
    //
    // Splitting the predicate is what removed the blocker
    // (docs/adr/0007-bound-versus-computed.md). This asserts the property stage
    // B depends on, so a regression to one predicate fails here rather than in
    // a silently skipped update.
    const Tensor a = from({4}, {1.0F, 2.0F, 3.0F, 4.0F});
    const Tensor doubled = vkml::mul(a, 2.0);

    const NodePtr node = doubled.node();
    REQUIRE_FALSE(node->is_bound());
    // Three nodes, not one: the scalar becomes a Full plus a Broadcast view,
    // which is the ordinary shape of `mul(t, 2.0)`.
    CHECK(vkml::topological_order(node).size() == 3);

    // Give it storage, as an Assign node bound to its destination would have.
    node->storage = vkml::backend_for(node->device).allocator().allocate(node->shape.nbytes());

    CHECK(node->is_bound());
    CHECK_FALSE(node->is_computed());
    CHECK(vkml::topological_order(node).size() == 3);  // and it STILL runs

    SUBCASE("marking it computed is what removes it from the schedule") {
        node->flags |= vkml::kFlagComputed;
        CHECK(vkml::topological_order(node).empty());
    }
}

// ---------------------------------------------------------------------------
// Assumption 2 -- hazards between dispatches in ONE submission
//
// VulkanBackend::compute() calls Recorder::barrier() between every pair of
// dispatches. The barrier is one global VkMemoryBarrier2 with
//
//   srcStageMask = dstStageMask = COMPUTE_SHADER | TRANSFER
//   srcAccessMask = SHADER_WRITE | TRANSFER_WRITE
//   dstAccessMask = SHADER_READ | SHADER_WRITE | TRANSFER_READ | TRANSFER_WRITE
//
// RAW and WAW need availability and visibility, and the access masks provide
// both. WAR needs only an EXECUTION dependency -- the earlier operation only
// reads, so there is nothing to flush -- and the stage masks provide that.
//
// READ THIS BEFORE TRUSTING THE TESTS BELOW. They pass with the barrier and
// they ALSO pass without it. That was measured, not assumed: `rec.barrier()`
// was commented out of VulkanBackend::compute(), and all of these still went
// green. So they are value-regression guards, NOT evidence that the barrier is
// doing anything.
//
// Vulkan's synchronization validation would be the right instrument, and it
// cannot help here either. It discovers shader accesses through DESCRIPTOR
// bindings, and vkML has none -- `grep vkCmdBindDescriptorSets src/` returns
// nothing, because buffers reach shaders as bufferDeviceAddress values in push
// constants (docs/ARCHITECTURE.md). The layer sees dispatches that touch no
// resources, so it reports no hazards whether or not any exist. Enabling it
// changed nothing in either arm.
//
// What establishes correctness is therefore the mask analysis above, read
// against the spec's dependency rules -- not a green run here.
// ---------------------------------------------------------------------------

TEST_CASE("write-after-read across independent dispatches in one submission") {
    for (const Device dev : devices()) {
        CAPTURE(dev.str());

        // Two computations with NO data dependency, so nothing but the barrier
        // orders them. The second WRITES the buffer the first READS.
        const Tensor a = from({4}, {1.0F, 2.0F, 3.0F, 4.0F}, dev);
        const Tensor b = from({4}, {10.0F, 20.0F, 30.0F, 40.0F}, dev);
        a.realize();
        b.realize();

        const Tensor reader = vkml::mul(a, 2.0);  // reads a
        const Tensor writer = vkml::mul(b, 3.0);  // independent of reader
        Backend& backend = vkml::backend_for(dev);

        std::vector<Node*> order = bind_all(reader, backend);
        const std::vector<Node*> writer_order = bind_all(writer, backend);
        order.insert(order.end(), writer_order.begin(), writer_order.end());

        // Alias the writer's OUTPUT onto `a`, which the reader is reading. That
        // is the hazard, and it cannot be built through the public API today --
        // every computed node gets fresh storage, which is exactly why this had
        // never been exercised.
        Node* wnode = writer.node().get();
        wnode->storage = a.node()->storage;
        wnode->storage_offset = a.node()->storage_offset;

        backend.compute(order);

        // The reader must hold 2*a from BEFORE the writer clobbered a.
        CHECK(read_back(backend, *reader.node(), 4) == std::vector<float>{2.0F, 4.0F, 6.0F, 8.0F});
        // and a must now hold 3*b.
        CHECK(read_back(backend, *a.node(), 4) == std::vector<float>{30.0F, 60.0F, 90.0F, 120.0F});
    }
}

TEST_CASE("read-after-write within one realised graph") {
    // The ordinary case, asserted explicitly rather than relied on implicitly:
    // every chained expression depends on it, but nothing named it.
    for (const Device dev : devices()) {
        CAPTURE(dev.str());
        const Tensor a = from({4}, {1.0F, 2.0F, 3.0F, 4.0F}, dev);
        const Tensor chained = vkml::add(vkml::mul(a, 2.0), 1.0);
        CHECK(host(chained) == std::vector<float>{3.0F, 5.0F, 7.0F, 9.0F});
    }
}

TEST_CASE("a dispatch writing the buffer its own input aliases") {
    // The shape stage B produces: `p.assign_(p - g*lr)` reads p in one dispatch
    // and writes p in a later one, within what would become a single submission.
    for (const Device dev : devices()) {
        CAPTURE(dev.str());
        const Tensor p = from({4}, {1.0F, 2.0F, 3.0F, 4.0F}, dev);
        const Tensor g = from({4}, {0.5F, 0.5F, 0.5F, 0.5F}, dev);
        p.realize();
        g.realize();

        const Tensor updated = vkml::sub(p, vkml::mul(g, 0.1));
        Backend& backend = vkml::backend_for(dev);

        const std::vector<Node*> order = bind_all(updated, backend);
        REQUIRE(order.size() >= 2);

        // Alias the ROOT onto p, so the final dispatch writes the buffer that
        // earlier dispatches in the same submission read.
        Node* root = updated.node().get();
        root->storage = p.node()->storage;
        root->storage_offset = p.node()->storage_offset;

        backend.compute(order);

        const std::vector<float> got = read_back(backend, *p.node(), 4);
        const std::vector<float> want{0.95F, 1.95F, 2.95F, 3.95F};
        for (size_t i = 0; i < got.size(); ++i) {
            CHECK(got[i] == doctest::Approx(want[i]).epsilon(1e-6F));
        }
    }
}

TEST_CASE("aliased execution is deterministic across repeats") {
    // Determinism is part of correctness (P1). A hazard that the barrier failed
    // to order would most likely show up as a result that varies run to run
    // rather than one that is always wrong, so a single-shot check could pass
    // over a real race. Compared as BYTES, not approximately.
    for (const Device dev : devices()) {
        CAPTURE(dev.str());
        std::vector<std::vector<float>> results;

        for (int trial = 0; trial < 20; ++trial) {
            const Tensor a = from({64}, std::vector<float>(64, 1.5F), dev);
            const Tensor b = from({64}, std::vector<float>(64, 2.5F), dev);
            a.realize();
            b.realize();

            const Tensor reader = vkml::mul(a, 2.0);
            const Tensor writer = vkml::mul(b, 3.0);
            Backend& backend = vkml::backend_for(dev);

            std::vector<Node*> order = bind_all(reader, backend);
            const std::vector<Node*> writer_order = bind_all(writer, backend);
            order.insert(order.end(), writer_order.begin(), writer_order.end());

            Node* wnode = writer.node().get();
            wnode->storage = a.node()->storage;
            wnode->storage_offset = a.node()->storage_offset;

            backend.compute(order);
            results.push_back(read_back(backend, *reader.node(), 64));
        }

        for (size_t i = 1; i < results.size(); ++i) {
            CHECK(results[i] == results[0]);
        }
        CHECK(results[0] == std::vector<float>(64, 3.0F));
    }
}
