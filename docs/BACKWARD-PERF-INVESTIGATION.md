# Backward-pass performance investigation

**Status:** CLOSED. Root cause found (H11) and fixed; CIFAR-100 compute is
**5.6x** faster, 92.0 -> 16.5 ms/step. Ten hypotheses were eliminated first, and
the reason it took eleven is itself recorded, in 5.
**Date:** 2026-07-29
**Hardware:** AMD RX 5600M (RDNA1), RADV, Vulkan 1.4.354. All figures from this machine.

This records an investigation that did not finish, so the next person does not repeat
the eliminated half. Every number below was produced by running something; where a
claim is inference rather than measurement it says so.

---

## 1. The symptom

Training a CIFAR-100 CNN (`examples/cifar100/train.py`, batch 64) spends 99% of each
step in "compute". Splitting that stage by stage:

```
                submissions   dispatches      wall
forward              1            42         3.51 ms
backward            11            46        82.48 ms
optimiser           32            48         4.41 ms
```

Backward issues roughly the same number of dispatches as forward and takes **23x**
longer. That ratio is the subject of this document.

**After the H11 fix, re-measured the same way** (batch 64, minimum of 12 steps,
profiling on throughout, so this table is internally comparable but must NOT be
compared against the one above -- rule 4):

```
                submissions   dispatches      wall     gpu   gpu/wall
forward              1            42         2.52    2.32     0.92
backward            11            42         6.12    4.84     0.79
optimiser           32            48         3.30    0.31     0.09
                                            -----   -----
step                                        11.94    7.47     0.63
```

Backward is **13.5x** faster and the ratio to forward is now **2.4x**, against
1.6x on the CPU backend -- normal. Its dispatch count fell 46 -> 42, which is
exactly the four degenerate reductions counted in the CIFAR graph, and is an
independent confirmation of the cause.

**The bottleneck has moved to the optimiser**, and 6 records what that means.

---

## 2. What was eliminated, and by what evidence

### H1 — host-side graph construction. REJECTED.

Varying batch size while holding graph structure identical:

```
batch   forward     backward   ratio
    1    0.83 ms    10.19 ms   12.2x
    8    1.00 ms    18.13 ms   18.1x
   64    3.67 ms    84.67 ms   23.1x
```

Fits `backward ≈ 9.0 ms + 1.18 ms/sample` (the batch-8 prediction is 18.5 against
18.13 measured). Node count is identical across all three, so a purely host-side
cost would be flat. It is not.

### H2 — submission/synchronisation overhead. REJECTED for backward, CONFIRMED for the optimiser.

Per-submission submit+wait latency, minimum of 200 trivial realises: **0.106 ms**.
(Re-measured later at **0.0729 ms** over 400 realises. The table below is left at
the value it was computed with; use the newer figure for new work -- see 6.)

```
             submissions   predicted   measured
optimiser        32         3.40 ms     4.41 ms   <- essentially all overhead
backward         11         1.17 ms     9.01 ms   <- explains 13% of the fixed part
```

This is what makes task #19 (batch the optimiser's realise calls) a correctly
diagnosed but SMALL win, and what rules submissions out as backward's problem.

### H3 — the backward rules do too much work (algorithmic). REJECTED.

Same model on the CPU backend, batch 8:

```
backend   forward     backward    ratio
CPU       153.06 ms   245.21 ms   1.6x   <- textbook
Vulkan      1.00 ms    18.13 ms  18.1x
```

The work itself is normal. The cost is specific to the Vulkan path.

### H4 — tall-skinny GEMM (small K, the weight-gradient shape). REJECTED.

```
  M=1024 K=1024 N=1024   1740 GFLOP/s   (square, for reference)
  M=2048 K=2048 N=2048   2561 GFLOP/s
  M=4096 K=64   N=4096   1606 GFLOP/s   <- dW shape, healthy
```

### H5 — skinny-M GEMM (the dx shape). REJECTED.

```
  M=64   K=4096 N=4096   1201 GFLOP/s   <- dx shape, healthy
  M=256  K=4096 N=4096   2432 GFLOP/s
  M=1024 K=4096 N=4096   2539 GFLOP/s
```

Both GEMM shapes a Linear backward issues run at full speed standalone.

### H6 — transposed / non-contiguous operand hits a slow path. REJECTED.

Identical shape and FLOPs, contiguous operand versus a transposed view:

```
  contiguous (4096,64) @ (64,4096)   1.24 ms   1735 GFLOP/s
  transposed view  gradT.T @ x       1.64 ms   1308 GFLOP/s
```

A 25% cost, not the ~200x being looked for.

### H7 — memory landing in the wrong heap (BAR window). REJECTED, and the reasoning was bad.

The arithmetic never supported it: host-visible memory is roughly 25x slower than
VRAM, which cannot produce a 200x gap even if it were happening. Recorded because
the *reasoning* error is worth not repeating, not just the conclusion.

### H8 — a memory leak in the backward path. REJECTED under control.

An uncontrolled measurement showed backward growing `in_use` by 1 GB over five
iterations with 15 fresh device allocations, against zero for a standalone GEMM.
That was an artefact of the test: gradients were never cleared, so `backward()`
accumulated (`total = grad + new_grad`), which legitimately retains memory.

Controlled, with `optimiser.zero_grad()` each iteration:

```
                              time/iter   in_use/iter   dev_allocs/iter
(a) standalone GEMM             1.20 ms      +0.0 MB         +0.0
(b) backward, grads CLEARED   167.44 ms      +0.2 MB         +0.0
(c) backward, ACCUMULATING    506.36 ms    +205.5 MB         +3.0
```

**There is no leak.** With gradients cleared, allocation behaviour is identical to
the standalone path.

### H9 — a different pipeline variant is compiled (spec constants, spilling, vectorisation). REJECTED, and it kills the whole class.

`vulkan_pipeline_stats()` from two separate processes -- separate, because the
pipeline cache accumulates and one process would blur the sets together:

```
standalone (contiguous)  gemm_reg:wg256_sg0_lv4_..._2_2_4_1_1_0_0   vgpr=33 spill=0 lds=8192 waves=20
backward                 gemm_reg:wg256_sg0_lv1_..._2_2_4_0_1_0_0   vgpr=33 spill=0 lds=8192 waves=20
```

`lv` is `load_vector_width` (`vk_pipeline.cpp:14`), so backward does compile
SCALAR tile loads where the contiguous path gets vec4 -- the same mechanism as
task #32. That looked like the answer. It is not:

```
standalone, TRANSPOSED view   gemm_reg:wg256_sg0_lv1_..._2_2_4_0_1_0_0    1.40 ms
backward                      gemm_reg:wg256_sg0_lv1_..._2_2_4_0_1_0_0   167    ms
```

**Byte-identical pipeline key, 120x apart.** Same variant, same scalar loads, same
registers, no spilling, same LDS, same occupancy. Whatever is wrong is not in the
compiled code, which eliminates shader compilation, specialisation constants,
register allocation and vectorisation together.

### Synchronisation: what the code actually does (read, not measured)

`Recorder::dispatch()` emits **no barrier**. It binds the pipeline, pushes
constants, brackets the dispatch in timestamps, and returns. Barriers come from a
separate `Recorder::barrier()`, called from exactly two sites in
`vulkan_backend.cpp` (lines 2074 and 2123 at the time of writing).

That barrier is one GLOBAL memory barrier -- `srcStageMask` and `dstStageMask`
both `COMPUTE_SHADER | TRANSFER` -- and `vk_command.h` records the reason: a
per-buffer hazard tracker costs more CPU than the barrier costs GPU, for a graph
where almost every node depends on its predecessor. It is deliberate, documented,
and marked as the thing an M5 planner would make selective.

**This does not explain the symptom.** Barriers serialise dispatches; they do not
make a single dispatch 120x slower, and the measurement in question is one
dispatch's own window. Recorded so the next investigator does not re-read the same
code hoping for a different answer.

---

## 3. What the evidence positively establishes

**Per-dispatch attribution is trustworthy for this workload.** The guard from
`docs/MEASUREMENT-AUDIT.md` §3 -- compare `sum(parts)` against the `submit` window --
passes to three decimals, so these dispatches do not overlap and per-dispatch numbers
may be believed here:

```
Linear 4096->4096 backward   submit 508.707 ms   sum(parts) 508.699 ms   guard OK
      339.921 ms   66.8%
      167.197 ms   32.9%
      (six dispatches below 1.5 ms)
```

Combined with the controlled table above, this decomposes exactly:

* **339.9 ms** is the gradient-accumulation add, present only in arm (c).
  A 67 MB elementwise add costing 340 ms is roughly 0.6 GB/s against the ~44 GB/s
  the transfer benchmark reaches. This is a second, separate anomaly.
* **167.2 ms** is backward proper, and it survives every control.

---

## 4. The question, stated precisely -- and answered in H11 below

> With gradients cleared, no allocation growth, no device allocations, correct GEMM
> shapes and healthy standalone throughput, a Linear 4096->4096 backward takes
> **167 ms** while the two GEMMs it contains take **~3 ms** run standalone.

Roughly 55x, unexplained. Whatever it is, it is present when an operation runs inside
an autograd-produced graph and absent when the same operation runs on freshly created
tensors.

Since H9, the question is sharper still: **the two paths compile the same pipeline
and run it 120x apart.** The cause is not in the shader, and it is not in the
barrier code, which does not touch `dispatch()` at all.

### H10 — the backward path launches more work (dispatch geometry). REJECTED.

`VKML_VULKAN_DEBUG=1` already logs `M/N/K`, `grid` and `ktiles` per GEMM, so this
needed no rebuild:

```
standalone  gemm M=4096 N=4096 K=64   grid=128x128  ktiles=2
backward    gemm M=64   N=4096 K=4096 grid=2x128    ktiles=128
            gemm M=4096 N=4096 K=64   grid=128x128  ktiles=2
```

Backward's two GEMMs carry exactly the geometries measured standalone at 1.34 ms
and 1.67 ms. Same grid, same tiles, same ktiles, same workgroup count. Nothing is
launching more work.

**This reframes the symptom, and the reframing is the most useful thing in this
section.** The 167 ms figure was assumed to be a GEMM. It cannot be: both GEMMs
have normal geometry AND a pipeline key that runs in ~1.5 ms standalone. The
Linear backward issues eight dispatches, so **the 167 ms belongs to one of the
other six** -- an elementwise operation, not a matrix multiply. Every hypothesis
from H4 onwards was aimed at the wrong dispatch.

### H11 — a degenerate reduction: `sum` over axes of extent 1. CONFIRMED.

Acting on the reframing -- name the dispatch before theorising about it -- one
run with profiling and `VKML_VULKAN_DEBUG=1` ends the investigation:

```
reduce n_out=16777216 n_red=1 axes_mask=0x3 src=(1, 1, 4096, 4096)
dispatch op=sum kernel=reduce groups=16777216x1x1 wg=256 shape=(1, 1, 4096, 4096)

  timing submit      169.9037 ms
  timing full          0.0048 ms
  timing contiguous    0.0512 ms
  timing matmul        2.3930 ms
  timing sum         167.4477 ms   <- the whole gap, in one dispatch
```

The parts sum to 169.897 against a 169.904 submit window, so this decomposition
is exact (rule 3's guard).

**`n_red=1`.** The reduce kernel launches **one workgroup of 256 threads per
output element**, and each workgroup here reduces exactly one value. That is
16,777,216 workgroups -- 4.29 billion invocations -- to copy 16.7 M floats.
The reduction reduces nothing.

The healthy `sum` in the very next submission is the control: `n_out=4096
n_red=64`, a real reduction over the batch, **0.1625 ms**.

**Cause, in `src/autograd/autograd.cpp`, `reduce_to_shape()`:**

```cpp
if (i < lead) {
    axes.push_back(static_cast<int>(i));   // axis did not exist in the source
} else if (target_dims[i - lead] == 1 && gd[i] != 1) {
    axes.push_back(static_cast<int>(i));   // axis was stretched from 1
}
```

The trailing branch correctly ignores an axis that was not actually stretched
(`gd[i] != 1`). The leading branch has no such guard: it reduces every leading
axis unconditionally, whatever its extent. Matmul promotes its operands to 4-D
batched form, so gradients flowing back through it carry leading dims of extent
1 (`axes_mask=0x3` -- axes 0 and 1, both extent 1), and each one becomes a
full-size dispatch that computes an identity.

This fits every earlier observation, which is what makes it the cause rather
than another candidate: it is not a GEMM (H4-H6, H10); its pipeline is
irrelevant because H9 compared GEMM pipelines; it allocates nothing (H8); it
scales with tensor size, matching H1's `9.0 ms + 1.18 ms/sample`; and it exists
only inside an autograd-produced graph, which is exactly the boundary 4 drew.

### The fix, and what it was worth

Guarding both branches of the axis selection with `gd[i] != 1`. An axis of
extent 1 holds one value, so summing it is the identity and the trailing
`reshape(target_dims)` removes it just as well; when every axis drops out,
`axes.empty()` already returns a pure reshape and no dispatch is issued at all.

Numerically identical by construction -- a one-element sum has nothing to
reassociate -- so the numerical contract needed no re-derivation.

Microbenchmark, Linear 4096->4096 backward, same process and method:

```
             before      after
submit    169.9037 ms   1.6959 ms
  sum     167.4477 ms   (no dispatch)
  matmul    2.3930 ms   1.6556 ms
```

Whole workload, CIFAR-100 CNN, 20 steps at batch 64, fix reverted and rebuilt
for the control arm:

```
            compute   per step
before       1.84 s    92.0 ms
after        0.33 s    16.5 ms     5.6x
```

75.5 ms/step recovered, against **74 ms predicted** before the fix was written
from the degenerate workgroup count and a rate of 9.98 ns/workgroup. Both runs
report identical accuracy and a test loss of 4.6066, which is the end-to-end
check that nothing numerical moved.

---

## 5. Method notes, for whoever picks this up

**The instrument was broken, and that is why this took eleven hypotheses.**
`compute()` already logged a per-dispatch timing correlation under
`VKML_VULKAN_DEBUG=1`. It paired `profile[i]` with `nodes[i]` -- but
`Recorder::submit()` inserts the whole-submit window at the FRONT of the
profile, so slot 0 is not a dispatch. Every operation was therefore reported
with its predecessor's time, the last dispatch was dropped entirely, and the
submit total appeared as the first op's cost. Run before the fix, this workload
reported `timing op=full gpu=169.8561ms` -- attributing the entire submission to
a scalar fill.

The correlation was deleted rather than corrected. `Recorder::set_label()`
already existed for exactly this purpose and had never been called, so the
backend now labels each node's dispatches and the profile is printed straight
out, carrying its own names. That also fixes a second latent error: the pairing
assumed one dispatch per node, which is false for split-K GEMM and multi-level
reductions.

**The lesson is not "check for off-by-one".** A diagnostic that is wrong in a
plausible-looking way is worse than no diagnostic, because it is trusted. This
one produced a self-consistent story for ten hypotheses. What would have caught
it immediately: the reported per-dispatch numbers did not add up to the submit
window, and 3's own guard was the tool to notice that -- it was applied to the
aggregate profile but never to the labelled report.

**One instrumentation attempt was invalid and was deleted.** A `profile_stages()`
helper in the CIFAR-100 example measured forward/backward/optimiser by realising each
stage and reading the submit window afterwards. It reported "backward 1.2% of the
step", which is impossible. Cause: `loss.backward()` realises internally despite
`is_eager() == False`, so by the time the gradients were explicitly realised the work
was already done and the measurement read empty submissions. **Do not measure backward
by realising gradients after calling it.**

**Rules that constrained this investigation** (`docs/MEASUREMENT-AUDIT.md` §7):
rule 1b -- wall clock is admissible only when GPU/wall > 0.5, which several conv
measurements failed (42-64%); rule 2 -- report the minimum, never the mean, used
throughout; rule 3 -- never sum per-dispatch timestamps, which is why the guard exists;
rule 4 -- never compare profiled against unprofiled timings, which is why none of the
numbers here are compared against the 331 s training baseline; rule 6 -- warm pipelines
first, done in every loop above.

**`vkml.vulkan_submit_ms()`** was added during this work so the rule-3 logic lives in
one place rather than being copied between `bench/gpu_bench.py` and ad-hoc scripts.
`vulkan_last_profile()` is a footgun without it.

---

## 6. Consequences for the task list

* **#19 (batch optimiser realise calls) is now the largest remaining win**, and
  the re-measurement is what changed its priority -- not its cost, which never
  moved. It was ~5% of a 90 ms step; it is **27.6% of an 11.94 ms step**.

  The optimiser's `gpu/wall` is **0.09**: 3.30 ms of wall against 0.31 ms of GPU.
  Rule 1b would refuse wall clock for a claim about kernel speed, and no such
  claim is made -- a ratio that low is precisely the evidence that the time is
  NOT in the kernels. Submit+wait latency re-measured at **0.0729 ms** (minimum
  of 400 trivial realises; the 0.106 ms recorded earlier in 2 is stale and should
  not be reused), so 32 submissions account for 2.33 ms of the 2.99 ms non-GPU
  time, and the rest is host-side graph work.

  Collapsing the step to one submission removes 31 of them: **~2.3 ms saved on an
  11.94 ms step**, a floor rather than an estimate, since per-realise host work
  falls too.
* **A new item is implied but not yet filed:** the gradient-accumulation add running at
  ~0.6 GB/s. It only bites callers who accumulate across micro-batches, which nothing
  in the repository does yet, so it is recorded here rather than raised as a task.
* **The fix is made**, pinned by `test_backward_emits_no_degenerate_reductions`
  in `tests/python/test_invariants.py` 10, which asserts a dispatch BUDGET.
  Red-verified: it fails with the fix reverted. Recorded there, because it cost
  an attempt: a test that the dispatch count is independent of tensor size does
  NOT catch this and passed against the bug -- the degenerate reduction adds one
  dispatch at any size, and what scaled was its cost, not the count.
* **The 339.9 ms gradient-accumulation add is the only anomaly left open here.**
  It appears on the same shapes, so re-measure it before opening a separate
  investigation -- H8's arm (c) was measured with the degenerate reduction still
  present, so that figure is no longer trustworthy.
