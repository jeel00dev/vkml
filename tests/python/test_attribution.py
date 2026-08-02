"""Where a step's time went, and the one arithmetic that has to close.

WHY THIS EXISTS. `vkml.attribution` answers a question the project could not
previously answer at all -- which kernel cost what, across a whole step -- and
its answer is arithmetic over measured intervals. Arithmetic that does not
close is not an answer, and the specific way it fails to close here is already
documented and already cost time: a dispatch's timestamps end at ALL_COMMANDS,
a GLOBAL drain point, so concurrent dispatches each report a window stretching
to the end of the group (docs/MEASUREMENT-AUDIT.md 3, rule 3).

Summing those durations multiply-counts. The tests below pin the difference on
a real device: split-K is the one workload in vkML that dispatches
concurrently, its durations sum to several times its own submission window, and
the interval union does not.

The pure half of this file needs no GPU on purpose. Union and occupancy are
where an off-by-one lives, and a test that only runs where there is a Radeon is
a test most contributors never see fail.
"""
from __future__ import annotations

import numpy as np
import pytest

from vkvalidate import VULKAN_DEVICE, gpu_device, vulkan_ready  # noqa: F401

import vkml as V
from vkml.attribution import (
    Interval,
    attribute,
    capture,
    occupancy_share,
    union_ms,
)

requires_vulkan = pytest.mark.skipif(not vulkan_ready(), reason="no usable Vulkan device")


def iv(start, end, key="k"):
    return Interval(start, end, key)


# ---------------------------------------------------------------------------
# union_ms -- occupancy, not the sum of durations
# ---------------------------------------------------------------------------


def test_union_of_nothing_is_zero():
    assert union_ms([]) == 0.0


def test_disjoint_intervals_sum():
    assert union_ms([iv(0, 1), iv(2, 4)]) == pytest.approx(3.0)


def test_overlapping_intervals_are_counted_once():
    # The defect this whole module exists for, in miniature: the durations sum
    # to 4 and the elapsed time is 3.
    assert union_ms([iv(0, 2), iv(1, 3)]) == pytest.approx(3.0)


def test_a_contained_interval_adds_nothing():
    assert union_ms([iv(0, 10), iv(3, 4)]) == pytest.approx(10.0)


def test_touching_intervals_do_not_double_count_the_boundary():
    assert union_ms([iv(0, 1), iv(1, 2)]) == pytest.approx(2.0)


def test_unsorted_input_gives_the_same_answer():
    ordered = union_ms([iv(0, 2), iv(1, 5), iv(7, 8)])
    assert union_ms([iv(7, 8), iv(1, 5), iv(0, 2)]) == pytest.approx(ordered)


# ---------------------------------------------------------------------------
# occupancy_share -- a model, whose defining property is that it closes
# ---------------------------------------------------------------------------


def test_serial_intervals_are_charged_their_own_duration():
    """Where nothing overlaps the model must not engage at all."""
    shares = occupancy_share([iv(0, 2, "a"), iv(2, 5, "b")])
    assert shares["a"] == pytest.approx(2.0)
    assert shares["b"] == pytest.approx(3.0)


def test_two_concurrent_intervals_split_the_time_they_share():
    # [0,2) and [1,3): a alone for 1, both for 1, b alone for 1.
    shares = occupancy_share([iv(0, 2, "a"), iv(1, 3, "b")])
    assert shares["a"] == pytest.approx(1.5)
    assert shares["b"] == pytest.approx(1.5)


def test_shares_sum_to_the_union_exactly():
    """The property that makes a table built from shares account for the time.

    Without it a row's percentage means nothing, because the denominator and
    the numerators come from different arithmetic.
    """
    intervals = [iv(0, 5, "a"), iv(1, 3, "b"), iv(2, 9, "c"), iv(11, 12, "d")]
    assert sum(occupancy_share(intervals).values()) == pytest.approx(union_ms(intervals))


def test_repeated_keys_accumulate_rather_than_overwrite():
    """Split-K's partitions all carry one key; the last must not win."""
    shares = occupancy_share([iv(0, 1, "m"), iv(2, 3, "m")])
    assert shares["m"] == pytest.approx(2.0)


def test_zero_length_intervals_are_ignored():
    """A dispatch whose timestamps did not advance must not open a live span."""
    assert occupancy_share([iv(1, 1, "a")]) == {}


# ---------------------------------------------------------------------------
# attribute -- the join, on records shaped exactly as the surfaces return them
# ---------------------------------------------------------------------------


def record(label, start, dur, submission=1, dispatch=0):
    return {"label": label, "start_ms": start, "gpu_ms": dur,
            "submission": submission, "dispatch": dispatch}


def test_the_three_totals_partition_the_wall_time():
    records = [
        record("submit", 0.0, 1.0),
        record("matmul", 0.0, 0.6, dispatch=1),
        record("relu", 0.7, 0.2, dispatch=2),
    ]
    report = attribute(records, wall_ms=4.0)
    assert report.gpu_busy_ms == pytest.approx(0.8)
    assert report.gpu_idle_ms == pytest.approx(0.2)
    assert report.host_ms == pytest.approx(3.0)
    assert (report.gpu_busy_ms + report.gpu_idle_ms + report.host_ms) == pytest.approx(4.0)


def test_rows_sum_to_gpu_busy():
    records = [
        record("submit", 0.0, 1.0),
        record("matmul", 0.0, 0.6, dispatch=1),
        record("relu", 0.7, 0.2, dispatch=2),
    ]
    report = attribute(records, wall_ms=4.0)
    assert sum(k.gpu_ms for k in report.kernels) == pytest.approx(report.gpu_busy_ms)


def test_submit_windows_are_summed_across_submissions_and_intervals_are_not():
    """Rule 3's two halves, in one case.

    Two submissions each hold two dispatches that fully overlap. Summing the
    four durations gives 4.0; the answer is 2.0 -- one per submission -- and
    the submit windows do sum, because submissions are serial.
    """
    records = [
        record("submit", 0.0, 1.5, submission=1),
        record("a", 0.0, 1.0, submission=1, dispatch=1),
        record("b", 0.0, 1.0, submission=1, dispatch=2),
        record("submit", 0.0, 1.5, submission=2),
        record("a", 0.0, 1.0, submission=2, dispatch=3),
        record("b", 0.0, 1.0, submission=2, dispatch=4),
    ]
    report = attribute(records, wall_ms=10.0)
    assert report.submit_ms == pytest.approx(3.0)
    assert report.gpu_busy_ms == pytest.approx(2.0)


def test_start_offsets_are_relative_to_their_own_submission():
    """Two submissions both starting at 0 must not be treated as concurrent."""
    records = [
        record("submit", 0.0, 1.0, submission=1),
        record("a", 0.0, 1.0, submission=1, dispatch=1),
        record("submit", 0.0, 1.0, submission=2),
        record("b", 0.0, 1.0, submission=2, dispatch=2),
    ]
    report = attribute(records, wall_ms=5.0)
    assert report.gpu_busy_ms == pytest.approx(2.0)


def test_a_decision_refines_the_row_name_and_a_missing_one_does_not():
    records = [
        record("submit", 0.0, 1.0),
        record("matmul", 0.0, 0.5, dispatch=7),
        record("matmul", 0.5, 0.5, dispatch=8),
    ]
    decisions = [{"dispatch": 7, "chose": "gemm_naive", "site": "matmul.kernel"}]
    report = attribute(records, decisions, wall_ms=2.0)
    names = {k.key for k in report.kernels}
    assert names == {"matmul:gemm_naive", "matmul"}


def test_truncation_is_reported_when_the_window_dropped_a_submission():
    records = [record("submit", 0.0, 1.0, submission=9),
               record("a", 0.0, 1.0, submission=9, dispatch=1)]
    assert attribute(records, submissions_resolved=4).truncated
    assert not attribute(records, submissions_resolved=1).truncated


def test_a_submission_that_timed_nothing_is_not_truncation():
    """A download is a copy with no dispatch, so it is never offered to the
    window. Counting it as a dropped submission would warn on every workload
    that reads a result back."""
    records = [record("submit", 0.0, 1.0, submission=1),
               record("a", 0.0, 1.0, submission=1, dispatch=1)]
    report = attribute(records, submissions_made=8, submissions_resolved=1)
    assert not report.truncated
    assert report.submissions_made == 8


def test_the_table_renders_and_names_its_totals():
    records = [record("submit", 0.0, 1.0), record("matmul", 0.0, 0.9, dispatch=1)]
    text = attribute(records, wall_ms=3.0).table()
    assert "matmul" in text
    assert "GPU busy" in text
    assert "host and driver" in text
    assert "step wall" in text


# ---------------------------------------------------------------------------
# On a device: the intervals have to be real, and the union has to beat the sum
# ---------------------------------------------------------------------------


def profile_of(fn):
    """Run `fn` with profiling on and return the last submission's records.

    LAZY, overriding the suite's eager fixture. Eager mode realises every
    operator on its own, so a submission holds exactly one dispatch and nothing
    here about how several dispatches share a submission can be observed at
    all. The tests below are about that sharing.
    """
    V.vulkan_set_profiling(True, VULKAN_DEVICE)
    try:
        with V.eager_mode(False):
            fn()
        return list(V.vulkan_profile_records(VULKAN_DEVICE))
    finally:
        V.vulkan_set_profiling(False, VULKAN_DEVICE)


def split_records(records):
    submits = [r for r in records if r["label"] == "submit"]
    return submits, [r for r in records if r["label"] != "submit"]


@requires_vulkan
def test_every_dispatch_interval_lies_inside_its_submission_window():
    """The bracket has to bracket. If it does not, every remainder is fiction."""
    records = profile_of(lambda: (
        V.tensor(np.random.rand(256, 256).astype(np.float32), device=gpu_device()) + 1.0
    ).relu().numpy())
    submits, dispatches = split_records(records)
    if not submits or not dispatches:
        pytest.skip("this device produced no usable timestamps")

    window = submits[0]["gpu_ms"]
    for r in dispatches:
        assert r["start_ms"] >= 0.0, f"{r['label']} starts before its own submission"
        assert r["start_ms"] + r["gpu_ms"] <= window + 1e-6, (
            f"{r['label']} ends {r['start_ms'] + r['gpu_ms'] - window:.6f} ms after the "
            f"submission window closed")


def overlap_ratio(dispatches):
    """Summed durations over occupied time: 1.0 when nothing overlaps."""
    union = union_ms([Interval(r["start_ms"], r["start_ms"] + r["gpu_ms"], r["label"])
                      for r in dispatches])
    return sum(r["gpu_ms"] for r in dispatches) / union if union > 0 else 0.0


@requires_vulkan
def test_barrier_separated_dispatches_are_essentially_serial():
    """The control for the split-K test below.

    If BOTH shapes reported heavy overlap the finding would be about the
    instrument rather than about concurrency, and the union would be fixing
    nothing.

    NOT an assertion that the overlap is zero, which it is not: a dispatch's
    bracket opens at TOP_OF_PIPE and its predecessor's closes at ALL_COMMANDS,
    two different pipeline stages, so consecutive brackets abut with a small
    overlap even across a barrier. Measured on RX 5600M / RADV over eight
    dispatches at five sizes: 0.2 to 4 microseconds per boundary, giving ratios
    of 1.002 to 1.25 -- proportionally largest where the dispatches themselves
    are smallest, which is what identifies it as a fixed cost per boundary
    rather than a share of the work.
    """
    records = profile_of(lambda: (
        (V.tensor(np.random.rand(512, 512).astype(np.float32), device=gpu_device()) + 1.0)
        * 2.0).relu().numpy())
    _, dispatches = split_records(records)
    if len(dispatches) < 2:
        pytest.skip("nothing to compare")

    ratio = overlap_ratio(dispatches)
    assert ratio < 2.0, (
        f"{len(dispatches)} barrier-separated dispatches report {ratio:.2f}x more duration "
        f"than elapsed time. Above 2.0 an entire dispatch is being counted twice, which a "
        f"boundary effect cannot explain")


@requires_vulkan
def test_concurrent_dispatches_make_the_naive_sum_exceed_the_window():
    """The measurement error this module was built to survive, on real silicon.

    Split-K dispatches its partitions with no barrier between them -- that
    overlap is the point of the optimisation. Their durations therefore sum to
    far more than the submission they all live in, and the union does not.
    """
    def tall_skinny():
        a = V.tensor(np.random.rand(64, 4096).astype(np.float32), device=gpu_device())
        b = V.tensor(np.random.rand(4096, 64).astype(np.float32), device=gpu_device())
        (a @ b).numpy()

    records = profile_of(tall_skinny)
    submits, dispatches = split_records(records)
    if not submits or len(dispatches) < 4:
        pytest.skip("this device does not split this shape")

    window = submits[0]["gpu_ms"]
    naive = sum(r["gpu_ms"] for r in dispatches)
    union = union_ms([Interval(r["start_ms"], r["start_ms"] + r["gpu_ms"], r["label"])
                      for r in dispatches])

    # 11.8x measured on RX 5600M / RADV with sixteen partitions. 5.0 separates
    # that from the 1.25 ceiling of the serial control above by a wide margin
    # in both directions, so neither threshold is fitted to one run.
    assert naive / union > 5.0, (
        f"expected the durations to multiply-count ({naive:.4f} ms against a {union:.4f} ms "
        f"union); if they no longer do, the ALL_COMMANDS drain point has changed and "
        f"docs/MEASUREMENT-AUDIT.md rule 3 needs revisiting")
    assert union <= window + 1e-6, (
        f"the union ({union:.4f} ms) does not fit its own submission window ({window:.4f} ms)")


@requires_vulkan
def test_capture_accounts_for_a_multi_submission_step():
    """The exit criterion, at the smallest scale that exercises it."""
    gpu = gpu_device()
    a = V.tensor(np.random.rand(256, 256).astype(np.float32), device=gpu)
    b = V.tensor(np.random.rand(256, 256).astype(np.float32), device=gpu)

    with capture(index=VULKAN_DEVICE) as cap:
        for _ in range(4):
            ((a @ b) + 1.0).relu().sum().numpy()
    report = cap.report()

    assert report.submissions_made >= 4, "a four-iteration loop submitted less than four times"
    assert report.submissions_profiled > 1, (
        "profile_history retained one submission; a step is many and this is what "
        "vulkan_profile_records() could not see")
    assert not report.truncated
    assert report.gpu_busy_ms > 0.0
    assert sum(k.gpu_ms for k in report.kernels) == pytest.approx(report.gpu_busy_ms)
    assert report.submit_ms >= report.gpu_busy_ms - 1e-6, (
        "GPU busy time exceeds the submission windows it was measured inside")
    assert report.wall_ms >= report.submit_ms, (
        "wall time is shorter than the GPU time it contains -- the capture stopped its "
        "clock before the device finished")


@requires_vulkan
def test_the_retention_window_drops_the_oldest_and_says_so():
    """A bounded window must be bounded, and must not lie about being full."""
    gpu = gpu_device()
    a = V.tensor(np.random.rand(64, 64).astype(np.float32), device=gpu)

    V.vulkan_set_profiling(True, VULKAN_DEVICE)
    V.vulkan_set_profile_history(2, VULKAN_DEVICE)
    try:
        for _ in range(6):
            (a + 1.0).numpy()
        history = list(V.vulkan_profile_history(VULKAN_DEVICE))
        resolved = V.vulkan_profile_submissions_resolved(VULKAN_DEVICE)
    finally:
        V.vulkan_set_profile_history(0, VULKAN_DEVICE)
        V.vulkan_set_profiling(False, VULKAN_DEVICE)

    kept = {r["submission"] for r in history}
    assert len(kept) == 2, f"asked for 2 submissions, kept {len(kept)}"
    assert resolved >= 6, "the resolved counter did not see every submission it was offered"
    assert attribute(history, submissions_resolved=resolved).truncated


@requires_vulkan
def test_retention_off_by_default_and_released_when_disabled():
    gpu = gpu_device()
    a = V.tensor(np.random.rand(64, 64).astype(np.float32), device=gpu)

    V.vulkan_set_profiling(True, VULKAN_DEVICE)
    try:
        (a + 1.0).numpy()
        assert V.vulkan_profile_history(VULKAN_DEVICE) == [], (
            "retention must cost nothing until it is asked for")
        V.vulkan_set_profile_history(8, VULKAN_DEVICE)
        (a + 1.0).numpy()
        assert V.vulkan_profile_history(VULKAN_DEVICE) != []
        V.vulkan_set_profile_history(0, VULKAN_DEVICE)
        assert V.vulkan_profile_history(VULKAN_DEVICE) == []
    finally:
        V.vulkan_set_profile_history(0, VULKAN_DEVICE)
        V.vulkan_set_profiling(False, VULKAN_DEVICE)
