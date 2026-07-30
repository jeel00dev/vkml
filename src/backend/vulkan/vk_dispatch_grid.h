#pragma once

#include "vkml/util/assert.h"  // VKML_CHECK
#include "vkml/util/error.h"   // DeviceError

#include <cstdint>

namespace vkml::vk {

/// The workgroup grid for a flat dispatch.
struct DispatchGrid {
    uint32_t groups_x = 0;
    uint32_t groups_y = 1;
};

/// Chooses the grid covering `element_count` invocations.
///
/// WHY THIS IS A FREE FUNCTION TAKING LIMITS AS PARAMETERS, and not a method
/// reading them from the Context: so that a device reporting the GUARANTEED
/// MINIMUM can be tested on hardware that reports far more. Every machine this
/// project is developed on reports far more -- 2^32-1 for
/// maxComputeWorkGroupCount[x] on the development GPU against a guaranteed
/// 65535 -- so the ceiling could not be reached locally, and issue #20 (every
/// elementwise operation above 64 MiB failing) shipped because the only way to
/// exercise the limit was to own hardware that imposes it.
///
/// Pure integer arithmetic, no Vulkan types, so `tests/cpp/test_dispatch_grid.cpp`
/// compiles and runs in EVERY build configuration, including the CPU-only one
/// three CI jobs use and the Windows job. That is the point: the check now runs
/// where the bug was missed.
///
/// Throws DeviceError rather than returning a status, matching how every other
/// device-limit violation in this backend is reported.
[[nodiscard]] inline DispatchGrid choose_dispatch_grid(uint64_t element_count,
                                                       uint32_t workgroup_size,
                                                       uint64_t max_groups_x,
                                                       uint64_t max_groups_y) {
    VKML_CHECK(workgroup_size > 0, DeviceError, "dispatch needs a non-zero workgroup size");
    VKML_CHECK(max_groups_x > 0, DeviceError, "device reports no workgroups available in x");

    const uint64_t groups = (element_count + workgroup_size - 1) / workgroup_size;
    if (groups == 0) {
        return {};  // nothing to dispatch; the caller returns before recording
    }

    // maxComputeWorkGroupCount[x] is only GUARANTEED to be 65535, which caps a
    // one-dimensional dispatch at 65535 * workgroup_size elements -- 16,776,960
    // at the usual width of 256, or 64 MiB of f32. That is an ordinary tensor:
    // one batch of 256 ImageNet images is over twice it.
    //
    // The excess folds into y. Kernels reconstruct the flat index with
    // global_index() in common.glsl, which is identically gl_GlobalInvocationID.x
    // whenever y holds a single group -- so the common one-dimensional case
    // dispatches and indexes exactly as it did before folding existed.
    const uint64_t groups_x = groups <= max_groups_x ? groups : max_groups_x;
    const uint64_t groups_y = (groups + groups_x - 1) / groups_x;

    VKML_CHECK(groups_y <= max_groups_y, DeviceError,
               "dispatch needs {} workgroups ({} x {}) but the device allows {} x {}", groups,
               groups_x, groups_y, max_groups_x, max_groups_y);

    // global_index() reconstructs the flat index in a 32-bit uint, and the grid
    // covers groups_x * groups_y * workgroup_size invocations -- which rounds UP
    // past element_count. Once that product exceeds 2^32 the reconstruction
    // wraps: an invocation past the end lands on a SMALL index, passes the
    // `i >= n` bounds check every kernel does, and processes an element a second
    // time. That is idempotent for most kernels and silently wrong for
    // scatter_add, which would add twice.
    //
    // Reachable only above 4,294,901,760 elements (16 GiB of f32) on a device
    // reporting the 65535 floor, so no hardware here can produce it -- which is
    // exactly why it is checked rather than reasoned about.
    const uint64_t covered = groups_x * groups_y * static_cast<uint64_t>(workgroup_size);
    VKML_CHECK(covered <= (uint64_t{1} << 32), DeviceError,
               "dispatch of {} elements needs a {} x {} grid covering {} invocations, past the "
               "2^32 flat index space the shaders reconstruct",
               element_count, groups_x, groups_y, covered);

    return {static_cast<uint32_t>(groups_x), static_cast<uint32_t>(groups_y)};
}

}  // namespace vkml::vk
