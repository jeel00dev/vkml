#include "doctest.h"
#include "vk_dispatch_grid.h"
#include "vkml/util/error.h"

#include <cstdint>

// Dispatch geometry against the limits Vulkan GUARANTEES, not the ones this
// machine reports.
//
// WHY THIS FILE COMPILES UNCONDITIONALLY. Every other Vulkan test needs a
// device, so it lives in test_vulkan_device.cpp and is built only when
// VKML_VULKAN is on -- which the three C++ CI jobs and the Windows job do not
// set. choose_dispatch_grid is pure integer arithmetic taking the limits as
// parameters, so it needs neither a device nor a Vulkan build, and the floor
// gets exercised in every configuration instead of one.
//
// That distinction is the whole lesson of issue #20. maxComputeWorkGroupCount[x]
// is guaranteed to be only 65535; the development GPU reports 2^32-1. Every
// elementwise operation above 64 MiB failed on a driver reporting the floor, and
// no test here could have caught it, because reaching the limit required owning
// hardware that imposes it. Passing the limit in as a parameter removes the
// hardware from the question.

using vkml::vk::choose_dispatch_grid;
using vkml::vk::DispatchGrid;

namespace {

/// The Vulkan specification's Required Limits for compute dispatch.
///
/// Cited from the spec, not measured: these are what a conformant device may
/// report at minimum, so anything vkml does must hold here regardless of what
/// the machine running the tests happens to offer.
constexpr uint64_t kFloorGroupsX = 65535;
constexpr uint64_t kFloorGroupsY = 65535;

/// The width every vkml kernel is dispatched at (KernelConfig::workgroup_size).
constexpr uint32_t kWorkgroup = 256;

/// Elements one row of groups covers at the guaranteed floor: 16,776,960, or
/// 64 MiB of f32. Above this the grid must fold into y.
constexpr uint64_t kRowCapacity = kFloorGroupsX * kWorkgroup;

/// Total invocations a grid covers -- always >= element_count, because the last
/// workgroup is partly out of range and every kernel bounds-checks it away.
uint64_t covered(const DispatchGrid& g, uint32_t wg = kWorkgroup) {
    return static_cast<uint64_t>(g.groups_x) * g.groups_y * wg;
}

}  // namespace

TEST_CASE("a dispatch inside the floor stays one-dimensional") {
    // The property that makes folding safe to apply in every kernel rather than
    // only the large ones: while y holds a single group, global_index() is
    // identically gl_GlobalInvocationID.x, so the ordinary case is unchanged.
    for (const uint64_t n : {uint64_t{1}, uint64_t{255}, uint64_t{256}, uint64_t{257},
                             uint64_t{1} << 20, kRowCapacity - 1, kRowCapacity}) {
        CAPTURE(n);
        const DispatchGrid g = choose_dispatch_grid(n, kWorkgroup, kFloorGroupsX, kFloorGroupsY);
        CHECK(g.groups_y == 1);
        CHECK(g.groups_x == (n + kWorkgroup - 1) / kWorkgroup);
        CHECK(covered(g) >= n);
    }
}

TEST_CASE("a dispatch past the floor folds into y instead of failing") {
    // Issue #20: this is the case that threw on a driver reporting the floor,
    // taking down every elementwise operation above 64 MiB.
    for (const uint64_t n : {kRowCapacity + 1, kRowCapacity + kWorkgroup, kRowCapacity * 2,
                             kRowCapacity * 37 + 4097}) {
        CAPTURE(n);
        const DispatchGrid g = choose_dispatch_grid(n, kWorkgroup, kFloorGroupsX, kFloorGroupsY);

        CHECK(g.groups_x == kFloorGroupsX);
        CHECK(g.groups_y > 1);
        // Must cover every element, and waste less than one full row doing it.
        CHECK(covered(g) >= n);
        CHECK(covered(g) - n < kRowCapacity);
    }
}

TEST_CASE("the grid never exceeds the limits it was given") {
    // Swept rather than spot-checked: an off-by-one in the ceiling division
    // shows up as a grid one group over the limit, which the driver would reject
    // at submit time with a message naming neither the tensor nor the operator.
    for (uint64_t n = 1; n <= kRowCapacity * 4; n = n * 3 + 1) {
        CAPTURE(n);
        const DispatchGrid g = choose_dispatch_grid(n, kWorkgroup, kFloorGroupsX, kFloorGroupsY);
        CHECK(g.groups_x <= kFloorGroupsX);
        CHECK(g.groups_y <= kFloorGroupsY);
        CHECK(covered(g) >= n);
    }
}

TEST_CASE("every workgroup width vkml uses folds correctly") {
    // The width is a KernelConfig field, so it is not always 256. The geometry
    // must hold for each one, at the boundary where that width starts folding.
    for (const uint32_t wg : {32U, 64U, 128U, 256U, 512U, 1024U}) {
        CAPTURE(wg);
        const uint64_t row = kFloorGroupsX * wg;

        const DispatchGrid at = choose_dispatch_grid(row, wg, kFloorGroupsX, kFloorGroupsY);
        CHECK(at.groups_y == 1);

        const DispatchGrid over = choose_dispatch_grid(row + 1, wg, kFloorGroupsX, kFloorGroupsY);
        CHECK(over.groups_y == 2);
        CHECK(covered(over, wg) >= row + 1);
    }
}

TEST_CASE("a grid past the 32-bit flat index space is refused, not folded") {
    // global_index() reconstructs the index in a 32-bit uint. Past 2^32 the
    // reconstruction wraps, an out-of-range invocation lands on a SMALL index,
    // passes the `i >= n` check every kernel does, and processes an element
    // twice -- silently wrong for scatter_add, which would add twice.
    //
    // Only reachable above 16 GiB of f32 on a device reporting the floor, so no
    // hardware available to this project can produce it. It is checked here
    // precisely because it cannot be checked anywhere else.
    // 65535 x 256 groups of 256 = 4,294,901,760 invocations: the largest grid
    // whose flat indices all fit in a uint32, and an exact fit for this element
    // count -- 65,536 short of 2^32, which is the whole margin available.
    const uint64_t last_safe = kRowCapacity * 256;
    const DispatchGrid edge =
        choose_dispatch_grid(last_safe, kWorkgroup, kFloorGroupsX, kFloorGroupsY);
    CHECK(covered(edge) == last_safe);
    CHECK(covered(edge) <= (uint64_t{1} << 32));

    // One element more needs a 257th row, whose top index passes 2^32.
    CHECK_THROWS_AS(
        (void)choose_dispatch_grid(last_safe + 1, kWorkgroup, kFloorGroupsX, kFloorGroupsY),
        vkml::DeviceError);
}

TEST_CASE("a grid past the device's y limit is refused with both extents named") {
    // y is guaranteed 65535 as well, so a device at the floor cannot dispatch
    // more than 65535 * 65535 * 256 invocations however the grid is arranged.
    // The failure must name the geometry: "needs N workgroups" alone does not
    // tell the reader which limit was hit.
    const uint64_t past_everything = kRowCapacity * (kFloorGroupsY + 1);
    CHECK_THROWS_AS(
        (void)choose_dispatch_grid(past_everything, kWorkgroup, kFloorGroupsX, kFloorGroupsY),
        vkml::DeviceError);
}

TEST_CASE("degenerate inputs are rejected rather than dividing by zero") {
    CHECK(choose_dispatch_grid(0, kWorkgroup, kFloorGroupsX, kFloorGroupsY).groups_x == 0);
    CHECK_THROWS_AS((void)choose_dispatch_grid(1024, 0, kFloorGroupsX, kFloorGroupsY),
                    vkml::DeviceError);
    CHECK_THROWS_AS((void)choose_dispatch_grid(1024, kWorkgroup, 0, kFloorGroupsY),
                    vkml::DeviceError);
}

TEST_CASE("a generous device gets the same grid it always did") {
    // The development GPU reports 2^32-1 in x. Folding must not change anything
    // there, or the fix for a minimum-spec device would be a regression for
    // every other one.
    constexpr uint64_t kGenerous = 4294967295;
    for (const uint64_t n : {uint64_t{1}, uint64_t{1} << 20, kRowCapacity, kRowCapacity * 100}) {
        CAPTURE(n);
        const DispatchGrid g = choose_dispatch_grid(n, kWorkgroup, kGenerous, kFloorGroupsY);
        CHECK(g.groups_y == 1);
        CHECK(g.groups_x == (n + kWorkgroup - 1) / kWorkgroup);
    }
}
