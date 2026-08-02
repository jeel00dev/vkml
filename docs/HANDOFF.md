# Session handoff

Written when a session's working context ran out mid-stream. Delete it once the
next session has absorbed it — it is a note, not a document.

## Exact state

`main` is **32 commits ahead of `origin/main`** and has not been pushed.
Working tree clean.

Green: 123 C++ cases · 1553 Python tests · **the same suite re-run in lazy
mode** · CPU-only · `VKML_MIN_SPEC=1` · all 17 gates, 11 with an automated
control · the site at 61 pages, every link resolving.

---

## What this session did

**A realistic MNIST step went 1603 µs → 1045 µs, 1.53×, against the frozen
baseline at `d399fdd`.** No kernel changed.

### 1. The measurement record was wrong, in two ways

`SMALL-STEP-LATENCY.md` was written from an ad-hoc session with nothing
committed, and neither conclusion survived being re-run.

- **The validation layers were on.** They are on **by default** — a measurement
  takes them unless it *acts*, and `MEASUREMENT-AUDIT` rule 5 is phrased as a
  prohibition, which assumes enabling them is an act. Cost: 39% of a one-node
  realise, 31% of a step. Rule 5 now says so, as §6c.
- **The per-node cost was attributed to the host and it is the GPU.** The
  experiment subtracted a graph of *views* from a graph of *compute nodes*; a
  view graph issues no dispatches, so the subtraction cancelled no GPU time.
  Host cost per dispatch is **0.24 µs**, measured directly. The roadmap item
  asking it to fall "from ~13 µs to ~1–2 µs" was aimed at nothing and is
  withdrawn.

**The default itself stays.** It is deliberate and recorded, and CI depends on
it *without setting it* — `macos-moltenvk.yml` reasons about what validation
reports and never exports the switch, so flipping the default would silently
disable validation in three jobs. That was checked before deciding, and it is
the reason the answer is "fix the measurement path", not "fix the default".

`bench/latency_bench.py` is the tool that was missing.

### 2. Asynchronous submission — ADR 0012

Two thirds of a submission was the host asleep in `vkWaitSemaphores`. `Recorder`
now holds a **ring of four command buffers** and `compute()` submits without
waiting.

Four is measured: 59.3 µs blocking, 36.3 with a ring of two, 17.7 with four,
17.4 with eight. **Command-buffer replay was prototyped in the same run and
rejected with its number** — 15.2 µs, only 1.14× better than the ring, because
recording is ~2 µs against a ~17 µs submission.

```
                             before      after
  per submission (fixed)   66.32 us   19.52 us    3.4x
  per node (marginal)       4.83 us    2.40 us
  MNIST step (no upload)     1189 us     883 us   1.35x
```

The projection written down *before* the work was 1.4×.

### 3. An upload was draining the queue — a regression this session caused

`StagingBuffer::upload` waited on its copy's ticket **after** submitting. Once
submissions stopped blocking, that drained every earlier submission too, because
the timeline is monotonic. Moved to immediately *before* the next memcpy, which
is what it actually protects: 477 µs → 40 µs behind queued work.

Found by attributing the step, not by anything failing.

### 4. Selective barriers: prototyped, measured, deleted

llama.cpp does **not** barrier between every dispatch — `overlaps_unsynced`
compares buffer ranges and barriers only on a real overlap. `vk_command.h` cited
llama.cpp as authority for the opposite policy for months.

Implemented behind a switch, **bit-identical** on both workloads, and worth
**4% on an MLP step, 3% on a CNN**. Deleted: a training graph is mostly a
dependency chain, only a quarter of its barriers are elidable, and — decisively —
**it cannot be verified here** (below).

---

## Findings worth not re-deriving

- **The extension in `site-packages` is not the one `cmake --build` wrote.**
  CLAUDE.md documents this and it still cost an hour: every Python measurement
  and *every test run* for a stretch of this session exercised the pre-change
  binary, including a negative control that "passed" because the code it broke
  was not the code under test. `md5sum` both before believing any Python result.
- **Cross-submission synchronisation cannot be verified on this machine.** The
  leading barrier was removed on purpose and all 1550 tests passed; a probe
  built to provoke the hazard found zero corruption in 20 attempts with *both*
  protections removed. RADV serialises submissions to one queue in practice. The
  barrier stays because the specification requires it, not because anything here
  demonstrates it.
- **The validation suite runs eager, and eager mode now synchronises**, so a
  green run says nothing about the asynchronous path. Run it lazily by
  overriding the autouse fixture — that is a real second suite, and it is how
  this change was actually checked.
- **`vulkan_last_profile()` was implicitly valid after `realize()`** because
  `realize()` waited. It is not any more. This bit `latency_bench.py` during the
  very change that caused it and understated GPU time by 3×.
- **Do not subtract a profiled GPU total from an unprofiled wall clock.** Rule 4
  forbids comparing them and the derived "host = wall − GPU" is the same
  mistake wearing a hat; it put the GPU at 69% of a step. The clean form needs
  no profiler: time `realize()`, then time `realize()` followed by a drain, and
  subtract. Host 408–428 µs against GPU 416–429 µs — now within a few percent of
  each other, where before they were strictly additive.
- **An under-warmed script can be wrong by 3.4×**, not the 1.8× the audit
  records. Two throwaway scripts measuring the same CNN step disagreed 5817 vs
  1711 µs, and the difference was warm-up alone.
- A negative control that passes is not a control. Three of the four written
  this session fail correctly; the one that did not was the stale-binary case.

## The next concrete step

The step is now **balanced** — host ~420 µs, GPU ~420 µs — so neither side is
the obvious target and the order has changed:

1. **`loss.item()`, 371 µs of 863.** Not a regression: it is where the whole
   step's GPU work now lands, because nothing before it blocks. The remaining
   win on this workload is not reading the loss every step, which is a user-level
   change and belongs in the examples.
2. **The optimiser, 295 µs**, and **backward, 149 µs** — both host-side graph
   construction, and neither has been attributed further. That is the next
   measurement.
3. **`vkQueueSubmit2` at ~16.5 µs, six per step.** The only lever is fewer
   submissions. ADR 0006 already took this 39 → 8.
4. `matmul` remains `M3_ROADMAP`'s subject for CIFAR-100, unchanged by any of
   this.

## The one open decision, unchanged

**#114.** Nothing defines "releasable". `PHASE2-MANIFESTO.md` is authoritative
and defines scope, but gives objective criteria only for P1. Three tasks (#108,
#110, #112) have "a scope decision" as their only closing condition and cannot be
resolved from the repository.
