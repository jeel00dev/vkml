"""Kernels must fit what Vulkan GUARANTEES, not what this GPU happens to report.

Every portability defect this project has had shares one cause: a limit was
asserted against the development machine instead of against the specification's
Required Limits table. Push constants (#2), shared memory (#14), subgroup range
(#6), f16 rounding (#3) and workgroup count (#20) were all the same mistake in a
different field, and each was invisible locally because the development GPU
reports more headroom than the floor.

So these tests are written against the FLOOR. A device with headroom still
exercises them, because the assertions are about what the code requests rather
than what the driver tolerates -- which is what makes them able to fail on the
machine where the bug would otherwise be undetectable.

The floors below are quoted from the Vulkan specification's Required Limits
table, not measured here. A device may report more; none may report less.
"""

from __future__ import annotations

import numpy as np
import pytest

from vkvalidate import VULKAN_DEVICE, gpu_device, requires_vulkan

import vkml as V

pytestmark = requires_vulkan


# Vulkan 1.3 Required Limits. Cited, never measured -- a measured value would be
# this device's, which is the error these tests exist to catch.
GUARANTEED_MAX_PUSH_CONSTANTS = 128
GUARANTEED_MAX_COMPUTE_SHARED_MEMORY = 16384
GUARANTEED_MAX_WORKGROUP_INVOCATIONS = 128
GUARANTEED_MAX_WORKGROUP_COUNT_X = 65535


def device_report():
    """Limits for the device the suite is actually running on.

    Indexed by VULKAN_DEVICE rather than taking the first report: a machine with
    an integrated and a discrete GPU reports very different heaps and workgroup
    counts, and reading one device's limits while dispatching to another silently
    mis-sizes every bound below.
    """
    reports = V.vulkan_device_reports()
    if VULKAN_DEVICE >= len(reports):
        pytest.skip(f"device {VULKAN_DEVICE} is not present")
    return reports[VULKAN_DEVICE]


def test_the_device_meets_the_guaranteed_floors():
    """A conformant device cannot report less than the specification requires.

    Not a test of vkML. It is a guard on every other test in this file: if a
    driver reported below the floor, the floors would be the wrong yardstick and
    the failures here would be misattributed to vkML.
    """
    report = device_report()
    assert report["max_push_constants"] >= GUARANTEED_MAX_PUSH_CONSTANTS
    assert report["max_shared_memory"] >= GUARANTEED_MAX_COMPUTE_SHARED_MEMORY
    assert report["max_workgroup_invocations"] >= GUARANTEED_MAX_WORKGROUP_INVOCATIONS


# ---------------------------------------------------------------------------
# Workgroup count (issue #20)
# ---------------------------------------------------------------------------
#
# maxComputeWorkGroupCount[x] is guaranteed to be only 65535, so a
# one-dimensional dispatch covers at most 65535 * workgroup_size elements. At the
# usual width of 256 that is 16,776,960 -- 64 MiB of f32, which is an ordinary
# tensor rather than an exotic one. The host folds the excess into y and the
# shaders reconstruct the flat index; these check that the reconstruction is
# right, because a wrong one still runs and merely writes to the wrong places.


def require_room_for(n: int) -> None:
    """Skip unless `n` f32 elements fit, with room for an operand and a result.

    Gated on maxMemoryAllocationSize rather than the device-local heap: an
    integrated GPU reports a small device-local heap while allocating happily out
    of system memory, and gating on the heap skipped this test on exactly the
    device whose low workgroup-count limit made it worth running.
    """
    report = device_report()
    if n * 4 > report["max_allocation_size"]:
        pytest.skip(f"{n} f32 elements exceeds maxMemoryAllocationSize")


def elementwise_ceiling() -> int:
    """Largest element count a one-dimensional dispatch could cover here."""
    report = device_report()
    # 256 is KernelConfig::workgroup_size's default and what every elementwise
    # kernel uses; the count, not the width, is what this bounds.
    return report["max_workgroup_count_x"] * 256


@pytest.mark.parametrize("offset", [0, 1, 4097], ids=["at", "one-over", "ragged"])
def test_elementwise_crosses_the_workgroup_count_boundary(offset):
    """Values must stay correct where the dispatch grid gains a second row.

    Parametrised around the boundary rather than at one size: the failure this
    guards is an index reconstruction that is right for a whole number of rows
    and wrong for a partial one, which only a ragged tail exposes.
    """
    n = elementwise_ceiling() + offset
    require_room_for(n)

    # Distinctive per-index values. A constant would pass even if every
    # invocation wrote the same slot, which is precisely the bug in question.
    x = (np.arange(n, dtype=np.float32) % 9973.0) - 4986.0
    gpu = V.tensor(x, device=gpu_device())

    np.testing.assert_array_equal(V.relu(gpu).numpy(), np.maximum(x, 0.0),
                                  err_msg="unary result is misplaced past the 1-D ceiling")
    np.testing.assert_array_equal((gpu + gpu).numpy(), x + x,
                                  err_msg="binary result is misplaced past the 1-D ceiling")


def test_a_strided_operand_survives_the_boundary():
    """operand_offset() indexes from the flat index, so it moves with it.

    A contiguous walk can be right while the strided path is wrong, because only
    the strided path multiplies the index out into per-axis coordinates.
    """
    n = elementwise_ceiling() + 2
    require_room_for(n)

    x = (np.arange(n, dtype=np.float32) % 7919.0) - 3959.0
    rows = n // 2
    gpu = V.tensor(x[: rows * 2], device=gpu_device()).reshape((rows, 2))

    np.testing.assert_array_equal(gpu[:, 0].contiguous().numpy(),
                                  x[: rows * 2].reshape(rows, 2)[:, 0])


# ---------------------------------------------------------------------------
# Shared memory (issue #14)
# ---------------------------------------------------------------------------


def test_no_kernel_requests_more_shared_memory_than_the_floor():
    """Every pipeline the suite built must fit a minimum-spec device.

    `vulkan_pipeline_stats` reports what each created pipeline actually asked
    for. Running the real operator surface first and then auditing the result is
    what makes this catch a kernel nobody thought to check -- the alternative,
    listing kernels by hand, misses exactly the one that was added without
    thinking about the floor.
    """
    device = gpu_device()
    rng = np.random.default_rng(0)

    # A spread wide enough to instantiate the shared-memory users: reductions,
    # GEMM at several shapes (which picks different tile geometries) and GEMV.
    a = V.tensor(rng.standard_normal((128, 128)).astype(np.float32), device=device)
    b = V.tensor(rng.standard_normal((128, 128)).astype(np.float32), device=device)
    V.matmul(a, b).numpy()
    V.matmul(a, V.tensor(rng.standard_normal((128, 1)).astype(np.float32), device=device)).numpy()
    V.sum(a, 1).numpy()
    V.amax(a, 0).numpy()

    over = [
        (p["name"], p["lds_bytes"])
        for p in V.vulkan_pipeline_stats(0)
        if p["available"] and p["lds_bytes"] > GUARANTEED_MAX_COMPUTE_SHARED_MEMORY
    ]
    assert not over, (
        f"these pipelines exceed the {GUARANTEED_MAX_COMPUTE_SHARED_MEMORY}-byte guaranteed "
        f"shared-memory floor and cannot be created on a minimum-spec device: {over}"
    )


# ---------------------------------------------------------------------------
# Subgroup range (issue #6)
# ---------------------------------------------------------------------------


def test_the_reported_subgroup_range_is_self_consistent():
    """min <= size <= max, with a range no code may pin outside of.

    A Renoir iGPU reports a fixed 64..64 while the discrete card in the same
    laptop reports 32..64, and a probe that assumed 32 was always available threw
    out of its test case (#6). The range is a device property; a width is only
    legal inside it.

    The rejection itself is not asserted here. `vulkan_set_subgroup_override`
    records a preference and the range is enforced later, when a pipeline is
    created -- so a test of the refusal belongs where pipelines are built, and
    asserting it at the setter would only pin where the check happens to live.
    """
    report = device_report()
    assert report["min_subgroup_size"] <= report["subgroup_size"] <= report["max_subgroup_size"]
    assert report["min_subgroup_size"] >= 1
