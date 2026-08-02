"""Where a step's time went — joined from facts, owned by none of them.

WHAT THIS ANSWERS. `EXTENSIBILITY-ROADMAP.md` 4a measures that roughly three
quarters of a training step is overhead rather than arithmetic, and then says
the evidence is all indirect: batch scaling, device substitution, submission
counting. Those locate the problem and cannot close it. What closes it is a
table of where the time actually went, **with the unaccounted remainder shown
explicitly** — the remainder being the overhead itself, not an error term.

WHY IT LIVES HERE AND NOT IN A PRODUCER. Three facts are involved and each has
exactly one owner (`OBSERVABILITY-ARCHITECTURE.md` 4):

    Decision     what kernel was chosen, and instead of what   the backend
    Measurement  what each interval cost                       the profiler
    Identity     which dispatch, which submission              the recorder

Joining them is a fourth thing, and giving it to either producer would make
that producer a second owner of the other's fact. So this is a **consumer**: it
imports the public query surface, reads what is published, and writes nothing
back. Delete this file and both producers still work.

THE ONE MEASUREMENT RULE THAT DECIDES THE DESIGN. A dispatch's timestamps end
at ALL_COMMANDS, a global drain point, so concurrent dispatches each report a
window stretching to the end of the group. Summing durations therefore
multiply-counts: split-K's sixteen partitions sum to 2.3176 ms against a true
submission window of 0.2030 ms, a remainder of **-2.11 ms** (measured,
RX 5600M, RADV). A negative remainder is only the visible symptom — every
per-kernel share is wrong by the same factor.

This is why `ProfileRecord` carries `start_ms`, and why everything below works
on INTERVALS. Taking their union instead of their sum brings the same case to
0.1960 ms against 0.2030 ms — a remainder of +0.0070 ms, which matches the
+0.0072 ms measured on a strictly serial chain and is therefore the fixed cost
of the timestamp bracket rather than attribution error.

    python -m vkml.attribution        # a worked example on this machine
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import _vkml_core as _C

__all__ = [
    "Interval",
    "KernelCost",
    "StepAttribution",
    "capture",
    "occupancy_share",
    "union_ms",
]


@dataclass(frozen=True)
class Interval:
    """A half-open [start, end) span of GPU time within one submission."""

    start: float
    end: float
    key: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def union_ms(intervals: list[Interval]) -> float:
    """Time covered by at least one interval — the GPU's real occupancy.

    The admissible alternative to summing durations when intervals may overlap.
    Serial intervals give the same answer as a sum, which is what makes this
    safe to use unconditionally rather than only when overlap is suspected.
    """
    total = 0.0
    cur_start = cur_end = None
    for iv in sorted(intervals, key=lambda i: (i.start, i.end)):
        if cur_start is None:
            cur_start, cur_end = iv.start, iv.end
        elif iv.start <= cur_end:
            cur_end = max(cur_end, iv.end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = iv.start, iv.end
    if cur_start is not None:
        total += cur_end - cur_start
    return total


def occupancy_share(intervals: list[Interval]) -> dict[str, float]:
    """Split occupancy among the intervals covering each instant.

    A MODEL, not a measurement, and the distinction matters. When k dispatches
    are in flight the hardware does not report how it divided itself between
    them, and nothing here can discover that. This assigns each of them 1/k of
    that instant, which has one property worth the convention: **the shares sum
    to `union_ms` exactly**, so a table built from them accounts for all the
    time and none of it twice.

    Where dispatches are serial — every case in vkML except split-K's
    partitions, because the executor puts a barrier between nodes — k is 1 and
    the share is the measured duration. The model only engages where the
    measurement genuinely cannot distinguish.
    """
    events: list[tuple[float, int, str]] = []
    for iv in intervals:
        if iv.end > iv.start:
            events.append((iv.start, +1, iv.key))
            events.append((iv.end, -1, iv.key))
    events.sort(key=lambda e: e[0])

    shares: dict[str, float] = {}
    live: dict[str, int] = {}
    previous = 0.0
    for position, delta, key in events:
        active = sum(live.values())
        if active > 0 and position > previous:
            slice_ms = (position - previous) / active
            for name, count in live.items():
                if count:
                    shares[name] = shares.get(name, 0.0) + slice_ms * count
        live[key] = live.get(key, 0) + delta
        previous = position
    return shares


@dataclass(frozen=True)
class KernelCost:
    """One row of the table: what a kernel cost across the whole step."""

    key: str
    dispatches: int
    gpu_ms: float


@dataclass
class StepAttribution:
    """Where a step's wall time went, with nothing swept into a residual.

    The four numbers below partition `wall_ms` and are meant to be read
    together — each is the previous one minus what it could not account for.
    """

    kernels: list[KernelCost] = field(default_factory=list)
    #: GPU time inside submissions with at least one dispatch running.
    gpu_busy_ms: float = 0.0
    #: Sum of the whole-submit windows. Summing ACROSS submissions is the one
    #: summation rule 3 permits, because submissions are serial.
    submit_ms: float = 0.0
    #: Wall time the caller measured around the step. 0 when not supplied.
    wall_ms: float = 0.0
    #: Submissions present in the profile.
    submissions_profiled: int = 0
    #: Submissions whose timestamps resolved, including any the window dropped.
    submissions_resolved: int = 0
    #: Every submission the backend made, including those with no dispatch to
    #: time -- a download is a copy. The gap between this and
    #: `submissions_resolved` is submissions that did no compute at all, which
    #: is a fact about dispatch overhead rather than about this report.
    submissions_made: int = 0

    @property
    def gpu_idle_ms(self) -> float:
        """Submitted GPU time with no dispatch in flight: barriers and brackets."""
        return self.submit_ms - self.gpu_busy_ms

    @property
    def host_ms(self) -> float:
        """Wall time outside every submission window. **An upper bound.**

        Submission cost, driver time, host-side graph building, upload and
        download waits. **This is the number `EXTENSIBILITY-ROADMAP.md` 4a P1
        is hunting**, and it is reported rather than inferred from a ratio.

        Why a bound and not a measurement: `wall_ms` is a PROFILED wall clock,
        and profiling adds host-side readback in `resolve_timestamps()` --
        `docs/MEASUREMENT-AUDIT.md` 4 sizes it and rule 4 forbids subtracting
        an unprofiled run to remove it. The GPU rows are timestamps and are
        unaffected (3), so the inflation lands entirely here. Treat this as
        "no more than", and do not quote a delta between two of these unless
        both were profiled the same way.
        """
        return self.wall_ms - self.submit_ms

    @property
    def gpu_fraction(self) -> float:
        """`submit_ms / wall_ms` — rule 1b's admissibility check for this report.

        Below roughly 0.5 the measured operation does not dominate the measured
        window, and a wall-clock comparison is inadmissible whatever the effect
        size (`docs/MEASUREMENT-AUDIT.md` 6b). The report prints this rather
        than leaving the reader to compute it, because the case it guards
        against looked perfectly flat and was a 1.46x effect.
        """
        return self.submit_ms / self.wall_ms if self.wall_ms > 0 else 0.0

    @property
    def truncated(self) -> bool:
        """Whether the retention window dropped submissions it was offered.

        A truncated profile understates every GPU row and overstates `host_ms`
        by exactly what it dropped, so a report must say so rather than print a
        number that looks like a finding.

        Deliberately NOT `submissions_made > submissions_profiled`: submissions
        with no dispatch are never offered to the window, so that comparison
        warns on every workload that reads a result back and the warning stops
        meaning anything.
        """
        return self.submissions_resolved > self.submissions_profiled

    def table(self, limit: int = 12) -> str:
        """The report, as text. Rows sum to `gpu_busy_ms` by construction."""
        lines = []
        rows = sorted(self.kernels, key=lambda k: -k.gpu_ms)
        width = max((len(r.key) for r in rows), default=6)
        width = max(width, 18)

        lines.append(f"  {'kernel':<{width}}  {'count':>6}  {'gpu ms':>9}  {'% step':>7}")
        lines.append(f"  {'-' * width}  {'-' * 6}  {'-' * 9}  {'-' * 7}")

        def pct(ms: float) -> str:
            return f"{100.0 * ms / self.wall_ms:6.1f}%" if self.wall_ms > 0 else "      -"

        for row in rows[:limit]:
            lines.append(
                f"  {row.key:<{width}}  {row.dispatches:>6}  {row.gpu_ms:9.3f}  {pct(row.gpu_ms)}")
        if len(rows) > limit:
            rest = rows[limit:]
            lines.append(f"  {f'({len(rest)} more)':<{width}}  "
                         f"{sum(r.dispatches for r in rest):>6}  "
                         f"{sum(r.gpu_ms for r in rest):9.3f}  "
                         f"{pct(sum(r.gpu_ms for r in rest))}")

        lines.append(f"  {'-' * width}  {'-' * 6}  {'-' * 9}  {'-' * 7}")
        lines.append(f"  {'GPU busy':<{width}}  {'':>6}  {self.gpu_busy_ms:9.3f}  "
                     f"{pct(self.gpu_busy_ms)}")
        lines.append(f"  {'GPU idle in submits':<{width}}  {'':>6}  {self.gpu_idle_ms:9.3f}  "
                     f"{pct(self.gpu_idle_ms)}")
        lines.append(f"  {'host and driver *':<{width}}  {'':>6}  {self.host_ms:9.3f}  "
                     f"{pct(self.host_ms)}")
        lines.append(f"  {'=' * width}  {'=' * 6}  {'=' * 9}  {'=' * 7}")
        lines.append(f"  {'step wall':<{width}}  {'':>6}  {self.wall_ms:9.3f}  {pct(self.wall_ms)}")
        lines.append("")
        lines.append(f"  {self.submissions_made} submissions, "
                     f"{self.submissions_resolved} of them with work to time")
        lines.append("  * upper bound: a profiled wall clock includes the profiler's own "
                     "readback")
        lines.append(f"  GPU / wall = {self.gpu_fraction:.2f}" + (
            "" if self.gpu_fraction >= 0.5 else
            "   BELOW 0.5 -- this wall clock is inadmissible for comparison "
            "(MEASUREMENT-AUDIT 6b)"))

        if self.truncated:
            lines.append("")
            lines.append(f"  WARNING: the window held {self.submissions_profiled} of "
                         f"{self.submissions_resolved} profiled submissions. Every GPU row "
                         f"understates and 'host and driver' absorbs the difference. Raise "
                         f"the `submissions` argument.")
        return "\n".join(lines)


def attribute(records, decisions=(), wall_ms: float = 0.0, submissions_made: int = 0,
              submissions_resolved: int = 0) -> StepAttribution:
    """Join measurement to choice and account for a step's time.

    Pure: it takes the dicts the two query surfaces return and touches no
    device, which is what lets the accounting be tested without a GPU.

    `records` are `vulkan_profile_history()` entries; `decisions` are
    `decisions()` entries. A decision refines a row's name from the operator to
    the kernel that ran; a dispatch with no decision keeps the operator name,
    which is most of them — decisions are published where there is a genuine
    choice, not everywhere.
    """
    chosen: dict[int, str] = {
        d["dispatch"]: d["chose"]
        for d in decisions
        if d.get("dispatch") and d.get("chose")
    }

    submit_ms = 0.0
    per_submission: dict[int, list[Interval]] = {}
    counts: dict[str, int] = {}
    seen: set[int] = set()

    for r in records:
        seen.add(r["submission"])
        if r["label"] == "submit":
            # The one interval that may be summed across submissions.
            submit_ms += r["gpu_ms"]
            continue
        kernel = chosen.get(r["dispatch"])
        key = f"{r['label']}:{kernel}" if kernel else r["label"]
        start = r["start_ms"]
        per_submission.setdefault(r["submission"], []).append(
            Interval(start, start + r["gpu_ms"], key))
        counts[key] = counts.get(key, 0) + 1

    busy = 0.0
    shares: dict[str, float] = {}
    for intervals in per_submission.values():
        busy += union_ms(intervals)
        for key, ms in occupancy_share(intervals).items():
            shares[key] = shares.get(key, 0.0) + ms

    return StepAttribution(
        kernels=[KernelCost(key, counts[key], ms) for key, ms in shares.items()],
        gpu_busy_ms=busy,
        submit_ms=submit_ms,
        wall_ms=wall_ms,
        submissions_profiled=len(seen),
        submissions_resolved=max(submissions_resolved, len(seen)),
        submissions_made=max(submissions_made, len(seen)),
    )


class capture:
    """Records everything one step publishes, and reports where its time went.

        with vkml.attribution.capture() as cap:
            train_one_step()
        print(cap.report().table())

    Turns on profiling, retention and decision recording for its duration and
    turns them off again on exit. It takes the decision recorder over for that
    window: there is one, and a nested capture would fight the outer one.
    """

    def __init__(self, index: int = 0, submissions: int = 512, decisions: int = 4096):
        self._index = index
        self._submissions = submissions
        self._decisions = decisions
        self._records: list = []
        self._decision_list: list = []
        self._wall_ms = 0.0
        self._made = 0
        self._resolved = 0

    def __enter__(self) -> capture:
        _C.vulkan_set_profiling(True, self._index)
        _C.vulkan_set_profile_history(self._submissions, self._index)
        _C.record_decisions(self._decisions)
        self._made = _C.vulkan_stats(self._index)["submissions"]
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        # Every submission must have completed before its timestamps resolve,
        # and before the wall clock stops: a step that has only been SUBMITTED
        # has not been measured.
        _C.vulkan_synchronize(self._index)
        self._wall_ms = (time.perf_counter() - self._start) * 1e3
        self._made = _C.vulkan_stats(self._index)["submissions"] - self._made
        self._resolved = _C.vulkan_profile_submissions_resolved(self._index)
        self._records = list(_C.vulkan_profile_history(self._index))
        self._decision_list = list(_C.decisions())
        _C.vulkan_set_profile_history(0, self._index)
        _C.stop_recording_decisions()
        return False

    def report(self) -> StepAttribution:
        return attribute(self._records, self._decision_list, self._wall_ms, self._made,
                         self._resolved)


def _demo() -> int:
    """A worked example, so the module can demonstrate itself on any machine."""
    import numpy as np

    from . import device, has_vulkan, init_vulkan, tensor, vulkan_available

    if not has_vulkan or not vulkan_available():
        print("no Vulkan device; attribution has nothing to read")
        return 0

    init_vulkan()
    gpu = device("vulkan:0")
    a = tensor(np.random.rand(512, 512).astype(np.float32), device=gpu)
    b = tensor(np.random.rand(512, 512).astype(np.float32), device=gpu)

    with capture() as cap:
        for _ in range(10):
            ((a @ b) + 1.0).relu().sum().numpy()

    print(cap.report().table())
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
