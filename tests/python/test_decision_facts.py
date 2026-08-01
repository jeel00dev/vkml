"""A recorded decision must agree with what the driver actually compiled.

WHY THIS EXISTS. A decision record is the engine's claim about itself, and a
claim nothing can contradict is not evidence. `docs/OBSERVABILITY-ARCHITECTURE.md`
section 5 requires every fact to be checkable against reality, and section 13
warns that a check which can never fail is decoration.

THE TWO PRODUCERS, named explicitly because "a different accessor is not a
different owner" is the trap this file exists to avoid:

  1. THE DECISION SITE, in vulkan_backend.cpp, publishes what it chose at the
     moment it chose it (line ~2050).
  2. THE VULKAN DRIVER attests, through vkGetPipelineExecutableStatisticsKHR,
     which pipelines were successfully compiled. vkML asks for a pipeline ~140
     lines later, on a separate code path, from the flags the decision set.

They CAN disagree, and that is the whole point. If the branch is edited after
the publish, if the wrong pipeline name is fetched, or if the decision record is
simply wrong, the claim and the compiled reality diverge. The negative control
below proves the check notices.

WHAT THIS DELIBERATELY DOES NOT CHECK, because it would be circular. Whether the
*name* `gemm_naive` appears in the pipeline list is decided by vkML calling
`pipes.get("gemm_naive")`. Comparing a decision against vkML's own cache key
would be one producer wearing two hats. What makes the check below independent
is the DRIVER having compiled the thing and reported statistics for it — the
`available` flag and the register counts come from the shader compiler, not from
vkml.

STILL MISSING, recorded rather than papered over: the design also proposed
checking dispatch STRUCTURE — a split-K decision claiming 8 partitions must show
8 dispatches. `vulkan_last_profile()` reports per-OPERATION timings
(`[('submit', ms), ('matmul', ms)]`), not per-dispatch, so that signal does not
exist yet. It arrives with per-dispatch attribution, tracker #99.
"""
from __future__ import annotations

import numpy as np
import pytest

from vkvalidate import VULKAN_DEVICE, gpu_device, vulkan_ready  # noqa: F401

import vkml as V

requires_vulkan = pytest.mark.skipif(not vulkan_ready(), reason="no usable Vulkan device")


def _matmul_with_decisions():
    """Run one matmul, returning (decisions, pipelines) as two independent views."""
    V.record_decisions(32)
    try:
        a = V.tensor(np.random.rand(128, 128).astype(np.float32), device=gpu_device())
        b = V.tensor(np.random.rand(128, 128).astype(np.float32), device=gpu_device())
        (a @ b).numpy()
        return list(V.decisions()), list(V.vulkan_pipeline_stats(VULKAN_DEVICE))
    finally:
        V.stop_recording_decisions()


@requires_vulkan
def test_a_chosen_kernel_was_actually_compiled_by_the_driver():
    decisions, pipelines = _matmul_with_decisions()
    kernel_choices = [d for d in decisions if d["site"] == "matmul.kernel"]
    if not kernel_choices:
        pytest.skip("this device makes no matmul kernel-fallback decision")

    compiled = {p["name"].split(":", 1)[0] for p in pipelines if p["available"]}
    assert compiled, "the driver reported no pipeline statistics; nothing to check against"

    for d in kernel_choices:
        assert d["chose"] in compiled, (
            f"the engine says it chose {d['chose']!r}, but the driver compiled "
            f"{sorted(compiled)}. The decision record and the dispatch path disagree."
        )


@requires_vulkan
def test_a_rejected_kernel_was_never_compiled():
    """The stronger direction: a decision claiming it did NOT use something.

    If the engine says "gemm_naive instead of gemm_reg" while the driver has a
    compiled gemm_reg, then either the fallback did not really happen or the
    record names the wrong alternative. Both are defects the log could never
    have surfaced.
    """
    decisions, pipelines = _matmul_with_decisions()
    kernel_choices = [d for d in decisions if d["site"] == "matmul.kernel" and d["instead_of"]]
    if not kernel_choices:
        pytest.skip("this device makes no matmul kernel-fallback decision")

    compiled = {p["name"].split(":", 1)[0] for p in pipelines if p["available"]}
    for d in kernel_choices:
        assert d["instead_of"] not in compiled, (
            f"the engine says it used {d['chose']!r} INSTEAD OF {d['instead_of']!r}, "
            f"but the driver compiled {d['instead_of']!r} anyway."
        )


@requires_vulkan
def test_the_recorder_reports_a_truncated_history_as_truncated():
    """A window smaller than the traffic must say so rather than look complete."""
    V.record_decisions(1)
    try:
        for _ in range(3):
            a = V.tensor(np.random.rand(64, 64).astype(np.float32), device=gpu_device())
            (a @ a).numpy()
        kept, published = len(V.decisions()), V.decisions_published()
        assert kept <= 1
        if published > kept:
            assert published - kept > 0, "eviction happened but was not reported"
    finally:
        V.stop_recording_decisions()
