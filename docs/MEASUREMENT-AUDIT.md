# Measurement Audit

Every performance conclusion in this project rests on measurements. This document audits the
measuring instruments themselves, asking of each one: **could this tool change the thing it
is measuring, or misattribute it?**

The motivation is empirical rather than theoretical. Nine times now an apparent implementation
failure has turned out to be a measurement error, and in M3-03 the *profiler built to prevent
that class of error* was itself the cause. An instrument that silently changes its subject is
worse than no instrument, because it produces confident wrong numbers that survive review.

Each section states the question, the experiment, the result, and the resulting rule.

---

## 1. The noise floor

Before any tool can be audited, the measurement noise has to be characterised — otherwise a
null result cannot be distinguished from an undetectable one.

Wall clock, 40 repetitions, after 5 warm-up iterations:

```
shape             min      median     IQR      p95      max
1024x1024x1024   2.736     3.134     0.338    3.929    4.396
256x4096x256     1.177     1.302     0.131    1.890    2.076
2048x2048x2048  11.855    12.983     2.199   24.596   25.717
```

GPU timestamps, 30 repetitions, same shapes, three independent process launches:

```
shape            median spread across trials
1024x1024x1024   1.895 / 1.897 / 1.898     (0.16 %)
256x4096x256     0.725 / 0.731 / 0.732     (0.96 %)
2048x2048x2048   7.267 / 7.316 / 7.347     (1.1 %)
```

> **Fact.** GPU timestamps are roughly **20× more reproducible than wall clock** on this
> machine. Wall-clock p95 exceeds its own minimum by up to 2×; the timestamp median moves by
> about 1 % across independent processes.
>
> **Rule.** Any claimed effect below ~2 % must be supported by GPU timestamps and repeated
> trials. Wall clock is only admissible for effects larger than ~25 %, or where timestamps are
> known to be invalid (§3).

The `max` column also shows why **minimum** is the right statistic for a timing distribution
and mean is not: the tail is contaminated by scheduling, not by the kernel. The minimum
estimates the underlying cost; the mean estimates the cost plus whatever else the machine was
doing.

---

## 2. Does capturing compiler statistics change the compiled code?

**The question.** vkML sets `VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR` on every pipeline.
The Vulkan specification permits an implementation to compile differently when asked to retain
statistics. If it did, every VGPR, occupancy and scratch figure this project has recorded would
describe a pipeline that is not the one shipped — and **the statistics themselves could never
reveal this**, because a pipeline without the flag cannot be queried. It is a blind spot by
construction.

This had never been tested. It is the single largest unexamined assumption in the evidence
chain.

**The experiment.** `VKML_VULKAN_NO_PIPELINE_STATS=1` builds pipelines without the flag.
Compared by GPU timestamp (profiling held constant across both arms), three paired trials:

```
                 1024^3 median      256x4096x256      2048^3 median
stats ON         1.897 1.898 1.895  0.732 0.725 0.731  7.316 7.267 7.347
stats OFF        1.881 1.885 1.882  0.731 0.735 0.729  7.319 7.221 7.268
difference       +0.74 %            none               +0.56 %, inconsistent
```

**Result.** At 1024³ every ON median exceeds every OFF median — a consistent ordering,
about 0.74 %. At 2048³ the direction is inconsistent across trials. At 256×4096×256 the arms
are indistinguishable.

> **Experimentally supported.** Capturing statistics costs **at most ~0.8 %**, near the limit
> of detectability, and may be zero.
>
> **Consequence.** The measurement chain is not meaningfully biased. The smallest effect this
> project has ever claimed is Stage 8's 0.71×, a 29 % effect — two orders of magnitude above
> this. The flag stays on by default.
>
> **Caveat.** This is a *timing* comparison. It cannot rule out a change in generated code that
> happens to be performance-neutral. Ruling that out would need ISA disassembly, which the
> driver does not expose through `VK_KHR_pipeline_executable_properties` on this device.
> Recorded as a known limitation, not as a closed question.

---

## 3. Does timestamp profiling change or misattribute execution?

**The question.** Split-K is the first vkML operation to issue several dispatches with no
barrier between them, so the driver may overlap them. Profiled, it appeared to be a
catastrophic regression.

**Result 1 — the numbers do not add up.**

```
64x16384x64                 profiled       wall clock
  split-K off               2.769 ms        3.127 ms
  split-K, 8 partitions     6.286 ms        1.140 ms
```

A profiled time of 6.286 ms against a wall clock of 1.140 ms is impossible unless the
measurement is wrong about something.

**Result 2 — execution is *not* altered.** Wall clock with profiling on versus off:

```
                      wall (prof OFF)   wall (prof ON)   submit window   sum of dispatches
split-K, 8 parts         1.165 ms          1.761 ms         0.932 ms         7.208 ms
unsplit, 1 dispatch      3.164 ms          3.569 ms         2.720 ms         2.720 ms
```

Profiling adds ~0.4–0.6 ms of host-side readback in **both** cases. If it had serialised the
partitions, profiled wall clock would be ≈7.2 ms; it is 1.761 ms.

> **Correction.** M3-03 originally diagnosed this as profiling *serialising* the dispatches.
> **That was wrong.** Execution is unaffected. The defect is one of **attribution**.

**The mechanism.** `Recorder::end_timestamp()` writes at
`VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT` (`vk_command.cpp`), which is a *global* drain point: a
dispatch's end timestamp fires only once every concurrently running dispatch has also finished.
Each of the eight partitions therefore reports a window stretching to the end of the whole
group — all eight read ~0.84 ms — and summing them counts the same elapsed time eight times.

**Four independent lines of evidence** that this, and not serialisation, is what happens:

1. The submit window (0.932 ms) ≈ unprofiled wall (1.165 ms) minus host overhead.
2. The sum of parts (7.208 ms) ≈ 8 × the submit window — each part spans nearly the whole group.
3. Profiled wall (1.761 ms) ≪ sum of parts (7.208 ms) — nothing was serialised.
4. Control: for a *single* dispatch, submit window and sum agree exactly (2.720 = 2.720),
   showing the outer window is calibrated correctly.

**The fix, and what it does not fix.** `Recorder` now brackets the entire command buffer with
an outer timestamp pair reported as `submit`. That value is valid whether or not dispatches
overlap. The per-dispatch entries are *retained unchanged*, because they remain correct for
dependent dispatches and are how every earlier measurement was taken.

> **Rule.** For a submit containing independent dispatches, use the `submit` entry. **Never sum
> per-dispatch entries** unless a barrier separates every pair. `bench/gpu_bench.py` still sums,
> so it under-reports concurrent work and must be updated before split-K is enabled by default.

**Scope of the damage.** No earlier measurement is affected: every benchmark before M3-03 was a
single dispatch or a dependent chain, where the drain costs nothing and the attribution is
exact.

**Regression-tested.** `tests/python/test_invariants.py::test_submit_window_bounds_concurrent_dispatches`
asserts the window equals the sum for one dispatch and is strictly below it for eight.

---

## 3b. Attributing a submission when its dispatches overlap (2026-08-02)

§3's rule — *use `submit`, never sum the parts* — is correct and, taken alone, leaves the
question it was asked to answer unanswered. `submit` says what the group cost; it cannot say
what a **kernel** cost, which is what `EXTENSIBILITY-ROADMAP.md` §4a P0 needs before anything
can be prioritised.

**A duration cannot carry that.** Two dispatches reporting 0.15 ms each are indistinguishable
whether they ran one after the other or at the same time, so a consumer has no admissible way
to combine them. `ProfileEntry` therefore carries `start_ms` — the interval's offset within its
own submission — and every attribution works on **intervals**, taking their **union**.

Measured, RX 5600M / RADV, a 64×4096×64 matmul split into 16 partitions:

```
                       sum of parts     union of parts     submit window
  split-K, 16 parts      2.3176 ms         0.1960 ms         0.2030 ms
  remainder             -2.1146 ms        +0.0070 ms
```

The union's remainder (+0.0070 ms) matches the +0.0072 ms measured on a strictly serial
elementwise chain in the same session, which is what identifies it as the timestamp bracket's
own cost rather than unattributed work.

> **Rule 3, extended.** To attribute *within* a submission, take the **union of intervals**,
> never the sum of durations. The union equals the sum when nothing overlaps, so it is the
> unconditional form and does not require knowing in advance whether anything did.

### Serial dispatches do not report disjoint intervals, and the reason is structural

A barrier separates every pair of node-level dispatches, so their intervals ought to abut
exactly. They overlap slightly, always, because the two timestamps are written at **different
pipeline stages**: `begin_timestamp` at `TOP_OF_PIPE` and `end_timestamp` at `ALL_COMMANDS`. A
command reaches top-of-pipe before its predecessor's writes have drained.

Measured across eight dispatches at five sizes, sum-over-union:

```
    n=64    1.0715 - 1.0755      overlap 0.0048 - 0.0050 ms
    n=128   1.0320 - 1.0624      overlap 0.0024 - 0.0049 ms
    n=256   1.1531 - 1.2463      overlap 0.0220 - 0.0349 ms
    n=512   1.0038 - 1.0832      overlap 0.0016 - 0.0202 ms
    n=1024  1.0020 - 1.0074      overlap 0.0016 - 0.0060 ms
```

**The overlap is a bounded absolute cost per boundary — 0.2 to 4 µs — not a proportion.** It
looks largest where the dispatches are smallest, which is the signature of a fixed cost and the
reason a *ratio* threshold has to be chosen against the dispatch size. `test_attribution.py`
asserts below 2.0 for barrier-separated work and above 5.0 for split-K, thresholds that sit far
from both measured ranges rather than being fitted to either.

---

## 4. Does the profiler perturb the operation it measures?

Wall clock rises by roughly 0.4–1.5 ms when profiling is on (§3), from query-pool readback and
the host wait in `resolve_timestamps()`. That is host-side cost after submission; the GPU
timestamp delta itself is unaffected, which §3's single-dispatch control confirms.

> **Rule.** Never compare a profiled wall-clock time against an unprofiled one. Compare
> timestamps with timestamps, or wall clock with wall clock.

---

## 5. Other instruments

| Instrument | Can it change what it measures? | Evidence |
|---|---|---|
| `VKML_VULKAN_DEBUG` tracing | Formats nothing unless enabled; one predictable branch per dispatch | Read once into a static; not measured, low risk |
| Validation layers | **Yes, substantially** | Never enabled during benchmarking; enabled only in correctness runs |
| `VkPipelineCache` | Affects compile time, not execution | Pipelines are warmed before timing |
| `.numpy()` download | Dominates wall clock for large outputs | Overhead scales with output size: ~0.45 ms fixed + transfer |
| Bit-identity hashing | Cannot perturb — compares bytes after the fact | The reason it is the preferred acceptance criterion |

The last row is the general lesson. **A correctness criterion that compares bytes cannot be
distorted by the act of measuring**, which is why the split-K design deliberately targeted
bit-identity rather than a tolerance: it converted a numerical question into one no instrument
can get wrong.

---

## 6. Known limitations

Stated so they are not mistaken for settled questions.

1. **No ISA disassembly.** §2 rules out a *performance* difference from statistics capture, not
   a code difference. The driver exposes statistics but not disassembly here.
2. **`bench/gpu_bench.py` sums per-dispatch entries.** Correct today, wrong for any future
   multi-dispatch operation. Must be moved to the `submit` window.
3. **Compute-unit count is vendor-specific.** `shader_core_count` comes from
   `VK_AMD_shader_core_properties2`; it is 0 on non-AMD hardware, and every consumer must treat
   0 as a reason to decline an occupancy decision rather than guess.
4. **`max_waves` is a per-pipeline theoretical figure**, not achieved occupancy. It cannot see
   how many *independent* workgroups a CU can interleave, which is what
   `PERFORMANCE-MODEL.md` §5e is about. No counter for achieved occupancy is available.
5. **Wall clock includes host overhead** that scales with output size, so wall-clock
   comparisons are only valid between runs of the *same* shape.
6. **One machine, one driver.** Every number here is RADV on Navi 10. None of it is portable
   evidence about other drivers.

---

## 6b. Wall clock fails when transfer dominates, at ANY effect size (M4-R4)

Rule 1 says wall clock is admissible for effects above ~25 %. That is **not sufficient**. A
softmax padding sweep measured by wall clock showed a perfectly flat 1.00x across a 16x change
in resident waves. Measured by GPU timestamp, the same sweep showed a real **1.46x**:

```
                wall clock    GPU (submit window)
 resident 64      70.373m            4.970m
 resident  4      70.420m            7.278m   <- 1.46x, invisible to wall clock
```

The kernel was 7 % of the wall time; a 64 MiB device-to-host download was the other 93 %. A
46 % effect on 7 % of the measurement is a 3 % effect on the total -- below the noise floor.

> **Rule 1, corrected.** Wall clock is admissible only when the measured operation dominates
> the measured window. Check `GPU time / wall time` before trusting any wall-clock comparison;
> below roughly 0.5, use timestamps regardless of how large the expected effect is.

This nearly produced a false refutation of P1'' recorded as a cross-kernel failure.

---

## 6c. Rule 5 is violated by DOING NOTHING, and was (2026-08-02)

Rule 5 says *never benchmark with validation layers enabled*. It is written as a prohibition,
which quietly assumes that enabling them is an act. **In vkML it is the default.**
`vulkan_backend()` reads `VKML_VULKAN_VALIDATION` with a fallback of `true`, deliberately —
`web/content/reference_env.py` states it as a correctness-first choice, and CI depends on it
(`macos-moltenvk.yml` reasons about what validation reports without ever setting the switch).

So a measurement obeys rule 5 only by *acting*, and `docs/SMALL-STEP-LATENCY.md`'s original
decomposition did not. Measured, RX 5600M / RADV, minimum of 200:

```
                                    layers ON   layers OFF   inflation
  one-node realise                    98.0 us      60.0 us       1.63x
  MNIST MLP 784-128-10 step, b=64     1720 us      1189 us       1.45x
```

**A third of that document's headline number was the instrument.**

The knowledge was already in the repository and did not reach the measurement.
`bench/gpu_bench.py` records `validation_layers` in every baseline and its comment gives the
effect on the regression count — but it reads `os.environ`, which says what was *asked for*
rather than what the library *read*. A process that sets the variable after import records an
arm it did not run.

> **Rule 5, extended.** A benchmark must (a) set the switch itself, before the backend is
> created, and (b) report the value **as observed by the library** — `vkml.configuration()`,
> which is populated at the point of reading and cannot disagree with the code — not as read
> back from the environment.
>
> `bench/latency_bench.py` does both and is the pattern to copy.

The general lesson is the one §5's table ends on, in a new form: **a default is part of the
instrument.** An instrument audit that only examines the tools a measurement *chooses* misses
every tool it merely *inherits*.

---

## 7. Rules, collected

1. Effects below ~2 % require GPU timestamps and repeated independent trials.
1b. Wall clock is admissible only when the operation dominates the window -- check
    GPU/wall > 0.5 first, whatever the effect size (6b).
2. Report the **minimum** of a timing distribution, never the mean.
3. Never sum per-dispatch timestamps across independent dispatches — use `submit`.
3b. To attribute cost *within* a submission, take the **union of intervals**. It equals the
    sum when nothing overlaps, so it needs no prior knowledge of whether anything did (3b).
4. Never compare profiled against unprofiled timings.
5. Never benchmark with validation layers enabled — and note that they are ON BY DEFAULT, so
   this rule is broken by omission. Set the switch before the backend exists and report what
   `vkml.configuration()` observed, not what the environment says (6c).
6. Warm pipelines before timing; compilation is setup, not measurement.
7. An A/B toggle is only valid if the A arm is the **frozen, unmodified** baseline
   (Stage 6.5's spurious 1.4×).
8. Prefer acceptance criteria that compare bytes; they cannot be perturbed by measurement.
9. When two independent calculations agree, that is not confirmation — it is a warning that the
   experiment cannot distinguish them (`PERFORMANCE-MODEL.md` §5e, corrected in M3-03).
10. Check every correctness gate for **vacuity** before trusting a pass.

## Warm-up is a precondition of each measurement, not a phase

This machine parks at 400 MHz of a possible 1500 and raises clocks in response
to load, so an unwarmed benchmark measures the clock policy rather than the
code. On `dispatch 1 element`, the same binary in one process:

    cold                            0.01052 ms
    after sixty 1024-cubed matmuls  0.00572 ms

A factor of 1.8 from clock state alone. That is not a correction to the signal,
it IS the signal for anything small.

Warming once at the start of a run does not work, and the failed attempt is the
useful part -- a warm-up was added to the top of gpu_bench.py and the number did
not move. The order test says why:

    warm -> measure                 0.00612 ms
    warm -> transfers -> measure    0.00932    the transfers undo it
    transfers -> warm -> measure    0.00560

The suite runs 1/4/16 MiB uploads and downloads before reaching the dispatch
case. Those move memory without raising the compute clock and take long enough
for it to fall back, so anything measured after them is measured cold whatever
happened at the top of the run.

`measure()` now warms immediately before timing, with the operands cached per
device so allocating them does not become the thing being measured. Across the
suite, run-to-run spread fell from a median of 50.2% to 10.2%, and the number of
benchmarks whose own noise exceeded the 15% regression threshold fell from 22 of
34 to 7.

**What this cost before it was found.** An apparent 38.6% regression on
`dispatch 1 element`, tracked, investigated, and entirely inside the clock range.
The baseline recorded no clock or driver state, so nothing could rule it out --
which is why a baseline without its measurement conditions is not a baseline.
