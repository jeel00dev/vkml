#include "doctest.h"
#include "test_support.h"

#ifdef VKML_HAS_VULKAN

#    include "vk_device.h"
#    include "vkml/spv/fill.h"

#    include "vkml/util/error.h"

#    include <algorithm>
#    include <string>
#    include <utility>
#    include <vector>

using vkml::vk::Context;

namespace {

/// Vulkan tests are skipped rather than failed when no device is present, so
/// the same binary is meaningful on a CI runner without a GPU.
bool have_device() {
    static const bool present = vkml::vk::enumerate_device_count() > 0;
    return present;
}

}  // namespace

TEST_CASE("vulkan device enumeration") {
    if (!have_device()) {
        MESSAGE("no Vulkan device present; skipping");
        return;
    }
    const auto names = vkml::vk::enumerate_device_names();
    CHECK(names.size() == static_cast<size_t>(vkml::vk::enumerate_device_count()));
    for (const std::string& n : names) {
        CHECK(!n.empty());
    }
}

TEST_CASE("vulkan context creation and capability reporting") {
    if (!have_device()) {
        MESSAGE("no Vulkan device present; skipping");
        return;
    }
    const Context ctx(0, /*enable_validation=*/true);
    const auto& info = ctx.info();

    CHECK(!info.name.empty());
    CHECK(ctx.device() != VK_NULL_HANDLE);
    CHECK(ctx.queue() != VK_NULL_HANDLE);

    // Hard requirements: the backend refuses to create a device without these,
    // so reaching this point already proves them, but assert explicitly so a
    // regression names the missing feature.
    CHECK(info.buffer_device_address);
    CHECK(info.scalar_block_layout);
    CHECK(info.timeline_semaphore);

    // Sanity, not exact values: these must hold on any conformant device.
    CHECK(info.max_workgroup_invocations >= 128);
    CHECK(info.max_shared_memory >= 16 * 1024);
    CHECK(info.max_push_constants >= 128);
    CHECK(info.max_allocation_size > 0);
    CHECK(info.device_local_bytes > 0);
    CHECK(info.subgroup_size > 0);

    SUBCASE("capabilities are translated without Vulkan types leaking") {
        const vkml::DeviceCapabilities caps = ctx.capabilities();
        CHECK(caps.buffer_device_address == info.buffer_device_address);
        CHECK(caps.global_float_atomics == info.global_float_atomic_add);
        CHECK(caps.cooperative_matrix == info.cooperative_matrix);
        CHECK(caps.max_shared_memory_bytes == info.max_shared_memory);
        CHECK(caps.total_memory_bytes == info.device_local_bytes);
        CHECK(caps.asynchronous);
        // Device-local memory is not host-mappable, which is what forces every
        // upload through a staging buffer.
        CHECK_FALSE(caps.host_accessible_buffers);
        CHECK(caps.min_buffer_alignment >= 16);
    }
}

TEST_CASE("hard requirements are named one at a time") {
    // Needs no device, which is the point: this is the code path that runs on
    // hardware the backend REFUSES, and no machine the project can test on
    // reaches it. Synthesising the DeviceInfo is the only way to cover it.
    vkml::vk::DeviceInfo info;

    CHECK(vkml::vk::missing_requirement(info) == "bufferDeviceAddress");
    info.buffer_device_address = true;
    CHECK(vkml::vk::missing_requirement(info) == "scalarBlockLayout");
    info.scalar_block_layout = true;
    CHECK(vkml::vk::missing_requirement(info) == "timelineSemaphore");
    info.timeline_semaphore = true;
    CHECK(vkml::vk::missing_requirement(info).empty());

    // A device may lack everything optional and still be usable. If this ever
    // fails, an optional capability has silently become mandatory.
    CHECK(!info.shader_float16);
    CHECK(!info.cooperative_matrix);
    CHECK(!info.subgroup_size_control);
    CHECK(vkml::vk::missing_requirement(info).empty());
}

TEST_CASE("devices can be described without creating one") {
    if (!have_device()) {
        MESSAGE("no Vulkan device present; skipping");
        return;
    }
    const std::vector<vkml::vk::DeviceInfo> infos = vkml::vk::enumerate_device_info();
    const std::vector<std::string> names = vkml::vk::enumerate_device_names();

    REQUIRE(infos.size() == names.size());
    for (size_t i = 0; i < infos.size(); ++i) {
        CHECK(infos[i].name == names[i]);
        CHECK(infos[i].api_version != 0);
        // Reported for every device, usable or not -- that is what makes this
        // safe to run on unknown hardware.
        CHECK(infos[i].max_workgroup_invocations > 0);
    }
}

TEST_CASE("vulkan device index is validated") {
    if (!have_device()) {
        MESSAGE("no Vulkan device present; skipping");
        return;
    }
    const int n = vkml::vk::enumerate_device_count();
    CHECK_THROWS_AS(Context(n + 10, false), vkml::DeviceError);
    CHECK_THROWS_AS(Context(-1, false), vkml::DeviceError);
}

TEST_CASE("vulkan result strings are decoded") {
    CHECK(std::string(vkml::vk::result_string(VK_SUCCESS)) == "VK_SUCCESS");
    CHECK(std::string(vkml::vk::result_string(VK_ERROR_DEVICE_LOST)) == "VK_ERROR_DEVICE_LOST");
    CHECK_NOTHROW(vkml::vk::check(VK_SUCCESS, "noop"));
    CHECK_THROWS_AS(vkml::vk::check(VK_ERROR_DEVICE_LOST, "test"), vkml::DeviceError);
    // Allocation failure maps to the memory-specific exception so Python sees
    // MemoryError rather than a generic RuntimeError.
    CHECK_THROWS_AS(vkml::vk::check(VK_ERROR_OUT_OF_DEVICE_MEMORY, "test"), vkml::OutOfMemoryError);
}

TEST_CASE("embedded SPIR-V is well formed") {
    // The generated header is what proves the whole glslc -> embed -> compile
    // pipeline ran; a stale or truncated blob shows up here rather than as a
    // driver crash at pipeline creation.
#    include "vkml/spv/fill.h"
    CHECK(vkml::spv::fill_size > 0);
    CHECK(vkml::spv::fill_size % 4 == 0);
    CHECK(vkml::spv::fill[0] == 0x07230203);  // SPIR-V magic number
}

#endif  // VKML_HAS_VULKAN

#include "vk_pipeline.h"
#include "vkml/spv/gemv.h"
#include "vkml/spv/probe_private_array.h"

// ---------------------------------------------------------------------------
// Engineering-theory probe (M3-R4).
//
// docs/PERFORMANCE-MODEL.md 5h states that the register allocator spills a
// private array wholesale once it exceeds ~64 floats, and that the threshold is
// the ARRAY SIZE rather than total register pressure. That was derived entirely
// from gemm_reg.comp, where the array sits alongside shared memory, barriers,
// tile loads and accumulators -- any of which could have been the real cause.
//
// probe_private_array.comp isolates the array. These tests check that the law
// still holds when nothing else is present, and pin the threshold, which the
// GEMM kernel could not reach: its stack is RM*RN*STACK_LEVELS and skips every
// value between 64 and 80.
// ---------------------------------------------------------------------------

TEST_CASE("private array spill threshold is a property of the allocator") {
    if (!have_device()) {
        MESSAGE("no Vulkan device present; skipping");
        return;
    }
    Context ctx(0, false);
    if (!ctx.info().pipeline_executable_properties) {
        MESSAGE("driver does not report pipeline statistics; skipping");
        return;
    }
    vkml::vk::PipelineCache pipes(ctx);

    auto probe = [&](uint32_t asize) {
        vkml::vk::KernelConfig cfg;
        cfg.workgroup_size = 256;
        cfg.spec_constants = {256, asize};
        return pipes
            .get("probe_private_array", vkml::spv::probe_private_array,
                 vkml::spv::probe_private_array_size, 16, cfg)
            .stats;
    };

    SUBCASE("arrays at or below the threshold stay in registers") {
        for (const uint32_t n : {16U, 32U, 60U, 63U, 64U}) {
            const auto s = probe(n);
            if (!s.available) {
                MESSAGE("statistics unavailable; skipping");
                return;
            }
            CHECK(s.scratch_bytes == 0);
            CHECK(s.spilled_vgprs == 0);
        }
    }

    SUBCASE("arrays above the threshold spill wholesale") {
        // 65 is the decisive point: one float past the boundary. It is also a
        // size gemm_reg.comp can never produce, since its stack is
        // RM*RN*STACK_LEVELS and skips 65..79 entirely.
        for (const uint32_t n : {65U, 66U, 72U, 80U, 128U}) {
            const auto s = probe(n);
            if (!s.available) {
                return;
            }
            CHECK(s.scratch_bytes > 0);
            // Law 3, derived rather than fitted. Two refinements over the
            // original "scratch = floats x 256":
            //   * the driver reports scratch PER WAVE, so the multiplier is
            //     4 bytes x subgroup_size and NOT the workgroup size -- proven
            //     by doubling the workgroup to 512 with the subgroup held at
            //     64 and seeing no change;
            //   * the per-lane allocation is rounded up to a 4-float granule,
            //     visible only at 65 and 66. Every GEMM stack size is already a
            //     multiple of 4, which is why this stayed hidden until a probe
            //     could ask for 65.
            const uint64_t granule = (n + 3) / 4 * 4;
            CHECK(s.scratch_bytes == granule * 4 * ctx.info().subgroup_size);
        }
    }

    SUBCASE("the threshold is exactly 64 floats") {
        const auto at = probe(64);
        const auto past = probe(65);
        if (!at.available || !past.available) {
            return;
        }
        CHECK(at.scratch_bytes == 0);
        CHECK(past.scratch_bytes > 0);
    }
}

TEST_CASE("private array spill threshold, bracketed") {
    if (!have_device()) {
        return;
    }
    Context ctx(0, false);
    if (!ctx.info().pipeline_executable_properties) {
        return;
    }
    vkml::vk::PipelineCache pipes(ctx);
    MESSAGE("ASIZE  VGPR  scratch  waves   (probe_private_array)");
    for (const uint32_t n :
         {16U, 32U, 48U, 56U, 60U, 62U, 63U, 64U, 65U, 66U, 68U, 72U, 80U, 96U, 128U}) {
        vkml::vk::KernelConfig cfg;
        cfg.workgroup_size = 256;
        cfg.spec_constants = {256, n};
        const auto s = pipes
                           .get("probe_private_array", vkml::spv::probe_private_array,
                                vkml::spv::probe_private_array_size, 16, cfg)
                           .stats;
        if (!s.available) {
            return;
        }
        MESSAGE(n << "  vgpr=" << s.vgprs << "  scratch=" << s.scratch_bytes
                  << "  waves=" << s.max_waves);
    }
}

TEST_CASE("O2: is the private-array budget per array or per kernel?") {
    if (!have_device()) {
        return;
    }
    Context ctx(0, false);
    if (!ctx.info().pipeline_executable_properties) {
        return;
    }
    vkml::vk::PipelineCache pipes(ctx);
    auto probe2 = [&](uint32_t a, uint32_t b) {
        vkml::vk::KernelConfig cfg;
        cfg.workgroup_size = 256;
        cfg.spec_constants = {256, a, b};
        return pipes
            .get("probe_private_array", vkml::spv::probe_private_array,
                 vkml::spv::probe_private_array_size, 16, cfg)
            .stats;
    };
    for (const auto& [a, b] : std::vector<std::pair<uint32_t, uint32_t>>{
             {40, 0}, {40, 24}, {40, 40}, {64, 4}, {32, 32}, {32, 40}}) {
        const auto s = probe2(a, b);
        if (!s.available) {
            return;
        }
        MESSAGE("A=" << a << " B=" << b << " total=" << (a + b) << "  vgpr=" << s.vgprs
                     << "  scratch=" << s.scratch_bytes);
    }
}

TEST_CASE("M3-R5: scope boundaries of the private-array budget") {
    if (!have_device()) {
        return;
    }
    Context ctx(0, false);
    if (!ctx.info().pipeline_executable_properties) {
        return;
    }
    vkml::vk::PipelineCache pipes(ctx);
    // spec: {wg, ASIZE, BSIZE, VSIZE, DYNAMIC}
    auto probe = [&](uint32_t a, uint32_t vs, uint32_t dyn, uint32_t sg) {
        vkml::vk::KernelConfig cfg;
        cfg.workgroup_size = 256;
        cfg.required_subgroup_size = sg;
        cfg.spec_constants = {256, a, 0, vs, dyn};
        return pipes
            .get("probe_private_array", vkml::spv::probe_private_array,
                 vkml::spv::probe_private_array_size, 16, cfg)
            .stats;
    };

    MESSAGE("S1 -- vec4 arrays: budget in BYTES or in ELEMENTS?");
    for (const uint32_t v : {8U, 15U, 16U, 17U, 20U, 32U, 64U}) {
        const auto s = probe(1, v, 1, 0);
        if (!s.available)
            return;
        MESSAGE("  vec4[" << v << "] = " << v * 4 << " floats / " << v * 16
                          << " B   vgpr=" << s.vgprs << "  scratch=" << s.scratch_bytes);
    }

    MESSAGE("S2 -- constant index vs dynamic index, 128 floats:");
    for (const uint32_t dyn : {0U, 1U}) {
        const auto s = probe(128, 0, dyn, 0);
        if (!s.available)
            return;
        MESSAGE("  DYNAMIC=" << dyn << "  vgpr=" << s.vgprs << "  scratch=" << s.scratch_bytes);
    }

    MESSAGE("S3 -- subgroup 32 vs 64 (threshold and scratch scaling):");
    for (const uint32_t sg : {32U, 64U}) {
        for (const uint32_t n : {64U, 65U, 80U}) {
            const auto s = probe(n, 0, 1, sg);
            if (!s.available)
                return;
            MESSAGE("  sg=" << sg << " floats=" << n << "  vgpr=" << s.vgprs
                            << "  scratch=" << s.scratch_bytes);
        }
    }
}

TEST_CASE("M3-R5: the scope boundaries of A1 and A2 are permanent") {
    if (!have_device()) {
        return;
    }
    Context ctx(0, false);
    if (!ctx.info().pipeline_executable_properties) {
        return;
    }
    vkml::vk::PipelineCache pipes(ctx);
    auto probe = [&](uint32_t a, uint32_t vs, uint32_t dyn, uint32_t sg) {
        vkml::vk::KernelConfig cfg;
        cfg.workgroup_size = 256;
        cfg.required_subgroup_size = sg;
        cfg.spec_constants = {256, a, 0, vs, dyn};
        return pipes
            .get("probe_private_array", vkml::spv::probe_private_array,
                 vkml::spv::probe_private_array_size, 16, cfg)
            .stats;
    };

    SUBCASE("S1 -- the budget is 256 BYTES, not 64 array elements") {
        // vec4[16] is 64 floats' worth: same byte count as float[64], and it
        // behaves identically. vec4[17] crosses the boundary at 272 B even
        // though it is only 17 elements. An element-count budget would let
        // vec4[64] (1024 B) through; it does not.
        const auto at = probe(1, 16, 1, 0);
        const auto past = probe(1, 17, 1, 0);
        const auto far = probe(1, 64, 1, 0);
        if (!at.available)
            return;
        CHECK(at.scratch_bytes == 0);
        CHECK(past.scratch_bytes > 0);
        CHECK(far.scratch_bytes > 0);
    }

    SUBCASE("S2 -- A1 applies only to arrays that survive as arrays") {
        // A constant-indexed array is scalarised into SSA values and never
        // becomes an array, so the budget does not apply at any size. This is a
        // SCOPE boundary, not an exception: 128 floats is twice the budget.
        const auto constant_idx = probe(128, 0, 0, 0);
        const auto dynamic_idx = probe(128, 0, 1, 0);
        if (!constant_idx.available)
            return;
        CHECK(constant_idx.scratch_bytes == 0);
        CHECK(dynamic_idx.scratch_bytes > 0);
    }

    SUBCASE("S3 -- the threshold is per LANE; scratch is per WAVE") {
        for (const uint32_t sg : {32U, 64U}) {
            const auto at = probe(64, 0, 1, sg);
            const auto past = probe(65, 0, 1, sg);
            if (!at.available)
                return;
            // Threshold does not move with subgroup size: it is a per-lane budget.
            CHECK(at.scratch_bytes == 0);
            CHECK(past.scratch_bytes > 0);
            // A2, generalised: scratch is rounded up to a 1024-byte granule PER
            // WAVE. The 4-float granule reported in M3-R4 was this same rule
            // seen only at subgroup 64.
            for (const uint32_t n : {65U, 72U, 80U, 128U}) {
                const auto s = probe(n, 0, 1, sg);
                if (!s.available)
                    return;
                const uint64_t raw = static_cast<uint64_t>(n) * 4 * sg;
                CHECK(s.scratch_bytes == (raw + 1023) / 1024 * 1024);
            }
        }
    }
}

TEST_CASE("M4-R2: LDS footprint alone changes GEMV performance (H2 rejected)") {
    if (!have_device()) {
        return;
    }
    // The rejected hypothesis was that the cross-lane reduction's cost explains
    // GEMV's gap. A dummy shared array changes ONLY the LDS footprint -- same
    // instructions, same barriers, same fold -- and the pipeline's reported LDS
    // must move with it. If a future driver elides the padding, this test fails
    // and the M4-R2 discrimination must be re-run rather than assumed.
    Context ctx(0, false);
    if (!ctx.info().pipeline_executable_properties) {
        return;
    }
    vkml::vk::PipelineCache pipes(ctx);
    auto lds_for = [&](uint32_t pad_floats) {
        vkml::vk::KernelConfig cfg;
        cfg.workgroup_size = 64;
        cfg.spec_constants = {64, 8, 1, 64, pad_floats};
        return pipes
            .get("gemv", vkml::spv::gemv, vkml::spv::gemv_size, sizeof(uint64_t) * 3 + 20, cfg)
            .stats;
    };
    const auto none = lds_for(0);
    const auto padded = lds_for(2048);  // 8 KiB
    if (!none.available) {
        return;
    }
    CHECK(padded.lds_bytes > none.lds_bytes);
    CHECK(padded.lds_bytes - none.lds_bytes == 2048 * sizeof(float));
}

TEST_CASE("M4-R3: resident workgroups are an INTEGER function of LDS") {
    if (!have_device()) {
        return;
    }
    Context ctx(0, false);
    if (!ctx.info().pipeline_executable_properties) {
        return;
    }
    vkml::vk::PipelineCache pipes(ctx);
    // P1'' predicts flat plateaus: LDS values that floor() to the same workgroup
    // count must behave alike, and the count must step at the boundaries. This
    // pins the STRUCTURE -- the timing multipliers (1.00/1.28/3.72) are workload
    // specific and are not asserted here.
    const uint64_t budget = ctx.info().max_shared_memory;
    auto lds_of = [&](uint32_t pad_floats) {
        vkml::vk::KernelConfig cfg;
        cfg.workgroup_size = 64;
        cfg.spec_constants = {64, 8, 1, 64, pad_floats};
        return pipes
            .get("gemv", vkml::spv::gemv, vkml::spv::gemv_size, sizeof(uint64_t) * 3 + 20, cfg)
            .stats.lds_bytes;
    };
    const uint64_t base = lds_of(0);
    if (base == 0) {
        return;
    }
    // 0 and 1024 floats (4 KiB) must land in the same plateau; 4096 floats
    // (16 KiB) must drop at least one step.
    CHECK(budget / base == budget / lds_of(1024));
    CHECK(budget / lds_of(4096) < budget / base);
}

#include "vk_command.h"
#include "vkml/spv/probe_mlp.h"

// ---------------------------------------------------------------------------
// M4-R5: why does the latency-hiding crossover move between kernels?
//
// Drives probe_mlp.comp directly through the Recorder, sweeping memory-level
// parallelism against resident waves. Total bytes, arithmetic per byte and
// barrier count are constant across every point, so only MLP and occupancy vary.
// ---------------------------------------------------------------------------

TEST_CASE("M4-R5: MLP vs resident waves" * doctest::skip(false)) {
    if (!have_device()) {
        return;
    }
    Context ctx(0, false);
    if (!ctx.info().pipeline_executable_properties) {
        return;
    }
    vkml::vk::Allocator alloc(ctx);
    vkml::vk::Recorder rec(ctx, alloc);
    vkml::vk::PipelineCache pipes(ctx);
    rec.set_profiling(true);

    constexpr uint32_t kN = 16u * 1024 * 1024;  // 64 MiB
    constexpr uint32_t kWg = 256;
    const auto src = alloc.allocate(uint64_t(kN) * 4, vkml::vk::MemoryKind::DeviceLocal);
    const auto dst = alloc.allocate(uint64_t(kN) * 4, vkml::vk::MemoryKind::DeviceLocal);

    struct Push {
        uint64_t src, dst;
        uint32_t n;
    } push{src.address, dst.address, kN};

    MESSAGE("MLP  padKB    LDS  wg/CU  resident   gpu_ms");
    for (const uint32_t mlp : {1u, 2u, 4u, 8u}) {
        for (const uint32_t pad_kb : {0u, 4u, 8u, 16u, 32u, 48u}) {
            vkml::vk::KernelConfig cfg;
            cfg.workgroup_size = kWg;
            cfg.spec_constants = {kWg, mlp, pad_kb * 256};
            const auto& pipe = pipes.get("probe_mlp", vkml::spv::probe_mlp,
                                         vkml::spv::probe_mlp_size, sizeof(Push), cfg);
            const uint64_t lds = pipe.stats.available ? pipe.stats.lds_bytes : 0;
            // Enough workgroups to fill the GPU many times over.
            const uint64_t groups = 4096;

            double best = 1e9;
            for (int rep = 0; rep < 6; ++rep) {
                rec.begin();
                rec.dispatch_groups(pipe, &push, sizeof(push), groups);
                rec.wait(rec.submit());
                for (const auto& e : rec.profile()) {
                    if (e.label == "submit") {
                        best = std::min(best, e.gpu_ms);
                    }
                }
            }
            const uint64_t wgcu = lds > 0 ? ctx.info().max_shared_memory / lds : 16;
            MESSAGE(mlp << "  " << pad_kb << "  " << lds << "  " << std::min<uint64_t>(wgcu, 16)
                        << "  " << std::min<uint64_t>(wgcu * 4, 64) << "  " << best);
        }
    }
    alloc.free(src);
    alloc.free(dst);
    CHECK(true);
}
