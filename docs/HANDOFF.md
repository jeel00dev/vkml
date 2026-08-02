# Session handoff

Written when a session's working context ran out mid-stream. Delete it once the
next session has absorbed it — it is a note, not a document.

## Exact state

`main` is **36 commits ahead of `origin/main`** and has not been pushed.
Working tree clean.

Green: 123 C++ cases · 1576 Python tests · **the same suite re-run in lazy
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

## Session 2 — the optimiser, and what the dispatch census showed

**A realistic MNIST step went 1037 µs → 953 µs**, and the step is now **49
dispatches across 4 submissions**, down from 65 across 6.

### The census nobody had taken

The dispatch *count* was known; the *composition* was not. Taking it answered
three questions at once:

- **The optimiser issued 24 dispatches for two lines of arithmetic**, a third of
  them only to materialise a number. `scalar_like` wraps a Python float as a
  rank-0 tensor, so `velocity * 0.9` is a `full` node *and* a `mul` node.
  Measured directly: `a * b` is one dispatch, `a * 2.0` is two.
- **Twelve "matmuls" for a two-Linear MLP** turned out to be split-K working —
  one node, seven partitions and a reduce. See below.
- The rest is short elementwise chains interleaved with reductions, where
  general fusion would buy much less than it does in the optimiser.

### `scaled_add` — ADR 0013

`a*alpha + b*beta` as one operator. Four arms were prototyped against the
Recorder on the real parameter shapes:

```
  A  composed, as vkML built it   24 dispatches   126.0 us
  B  scaled_add                    8 dispatches    47.2 us   2.7x
  C  sgd_step, the whole update    4 dispatches    31.7 us   4.0x
  C2 the same, no barriers         4 dispatches    24.2 us   5.2x
```

**C was faster and was rejected**: two outputs mean it cannot be a graph node at
all, the arithmetic would exist in three places instead of one, and every
optimiser would need its own kernel. B is generic — every momentum optimiser is
this shape — and captures over half the win. The prototype was deleted with its
number recorded.

Optimiser 458 µs → 309 µs. Bit-identity checked byte for byte on both backends
against the composed form *and* an independent f32 reference.

### Two defects it exposed, both worth more than the speedup

- **Coefficients were applied after the node was realised.** `binary()` ends in
  `finish()`, which realises in eager mode, so every eager result used the
  defaults of 1 and 1. The lazy path hid it completely — and so did the GPU
  bit-identity check, which runs lazy. The eager suite caught it at once.
- **The backward rule was dead code.** `coverage_matrix.py` reported it as a
  rule that never fired: the optimiser calls `scaled_add` on detached tensors,
  so no gradient flows through it and the whole suite passed without running it.

### Split-K: a premise disproven, and a comment that lied

`SplitKMode`'s comment said AUTO was "deliberately identical to OFF". It has not
been since the profitability rule landed:

```
  auto      8 dispatches    44.8 us      MLP step 635 us
  off       1 dispatch     106.0 us      MLP step 871 us
  forced    8 dispatches    45.0 us      MLP step 619 us
```

Split-K is **on by default and worth 2.4× on that GEMM**. A reader trusting the
comment would have believed the opposite and might have "fixed" the dispatch
count. The adjacent `GemvMode` claim was checked and is still true.

## Session 3 — cost attribution, no code changed

`docs/OPTIMISER-COST-ATTRIBUTION.md` is the full item-by-item report. Three
things in it change what to do next.

**The gap being attributed did not exist.** ADR 0013 §5 put "189 µs real
against 47 µs recorder" side by side. 189 was `drained − submit_only` — a
blocking wall-clock wait over three submissions, unprofiled — and 47 was a GPU
timestamp window over one submission, profiled. Rule 4 forbids the comparison,
and 189 carries a ~43 µs wake-up that is not GPU work. ADR 0013 now carries the
correction. Real figures: **120 µs host, 149 µs GPU, 295 µs drained.**

**Every weight gradient is a transposed view**, and two of the optimiser's
eight dispatches take the strided path because of it:

```
  param 0 (128, 784)  grad strides [4, 512] bytes   contiguous = False
  param 2 (10, 128)   grad strides [4,  40] bytes   contiguous = False
```

In situ, the same shape costs **78.8 µs strided against 42.3 µs contiguous** —
25% of the step's GPU time. Isolated at 1M elements the penalty is **11×**, and
the split between causes is measured: the `offset_from` integer arithmetic is
1.26×, the access pattern is the rest.

**Copy-only submissions emit no `submit` window**, so the `assign`'s **36.9 µs**
of GPU was missing from every total that summed windows. True GPU per step is
~186 µs, not 149.

### What is NOT worth touching, with numbers

Command buffer recording (**1.0 µs for all eight dispatches**), pipeline lookup
(2.9 µs), allocation and free (0.3 µs), command buffer resets (0.01 µs) — 4.2 µs
together, 1.4% of the step. Descriptor allocation and memory mapping are
**structurally zero**: there is no descriptor machinery anywhere in the tree
(`setLayoutCount = 0` on every pipeline layout) and mapping is per-block.

`vkQueueSubmit2` is **13.7 µs and independent of what is in the buffer** —
thirteen times the recording cost, three times per step.

### Three synthetic benchmarks got this wrong before the profile got it right

Reproducing the optimiser's exact dispatch mix in C++ measured the strided
penalty at 1%, at working sets of 1.2, 4.7 and 18.6 MiB. Fresh allocation and
idle host gaps were tested and disproven too. **All three kept the data
L2-resident**; the real optimiser runs after a forward and backward have flushed
a 4 MiB L2. The same contiguous dispatch is 13.0 µs in a tight loop and 42.3 µs
in situ. The per-dispatch profile of the real workload answered in one run what
the benchmarks got wrong.

## The next concrete step

Phases now, each drained so nothing is misattributed:

```
  fwd+bwd    528 us      optimiser  294 us      item  111 us
```

Ranked in `OPTIMISER-COST-ATTRIBUTION.md` §6. In short:

1. **R3 first — merge the two compute submissions.** ~5%, low complexity, low
   risk, and it needs nobody's decision. The passes are separate only because
   `detach()` on an uncomputed node forces a realise, so `finish()` cannot run
   before the velocity pass is submitted; restructuring `Optimizer.step` merges
   them without touching the executor.
2. **R1 — contiguous weight gradients.** ~27% of the step, and it compounds:
   every elementwise op consuming a weight gradient takes the strided path, not
   only the optimiser. The fix belongs in the matmul backward rule.
   `.contiguous()` at the optimiser was measured and is **worse** (177 µs against
   151) because the copy costs more than the strided access saves at this size.
3. **R2 — in-place parameter update.** ~19%, but high risk: it needs the
   executor to bind a computed node's output to an existing buffer, which is the
   M5 planner's territory, and aliasing is exactly the class of bug the previous
   session proved **cannot be verified on this driver**.

2. **`fwd+bwd` at 528 µs is now the largest phase** and has not been attributed
   beyond its dispatch census. 12 of its dispatches are split-K partitions doing
   real work; the rest are short elementwise chains.

3. `matmul` remains `M3_ROADMAP`'s subject for CIFAR-100, unchanged by any of
   this.

## The one open decision, unchanged

**#114.** Nothing defines "releasable". `PHASE2-MANIFESTO.md` is authoritative
and defines scope, but gives objective criteria only for P1. Three tasks (#108,
#110, #112) have "a scope decision" as their only closing condition and cannot be
resolved from the repository.
