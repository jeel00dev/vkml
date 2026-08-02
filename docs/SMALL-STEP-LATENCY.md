# Why an MNIST step costs 1.2 ms when its arithmetic is 6 µs

**Date:** 2026-08-02, corrected the same day · **Hardware:** AMD RX 5600M (RDNA1), RADV
**Status:** measured. Reproduce with `python bench/latency_bench.py`.

The objective this answers: *make the GPU 15–20× faster than PyTorch CPU on workloads where
the GPU should dominate*, with MNIST named as the case that is currently slower and treated
as a bug until proven otherwise.

**It is not a bug in any kernel, and no kernel work can fix it.** This document says what it
is instead, with the measurement for each claim, and what the target would actually require.

> ### Two corrections to the first version of this document
>
> Both were methodology rather than arithmetic, which is why they survived being written down.
>
> **1. It was measured with the validation layers enabled.** They are **on by default** —
> `vulkan_backend()` reads `VKML_VULKAN_VALIDATION` defaulting to true — so a measurement
> takes them unless it acts, and this one did not. `MEASUREMENT-AUDIT.md` rule 5 says never
> to. Measured cost: **39% of a one-node realise, 31% of an MNIST step.** Every number in the
> first version was inflated by roughly a third.
>
> **2. It attributed the marginal per-node cost to the host, and it is the GPU.** The
> experiment subtracted a graph of *views* from a graph of *compute nodes*. A view graph
> issues **no dispatches**, so that subtraction does not cancel GPU time — there was none to
> cancel — and the whole of it landed in a line labelled "allocate + record + barrier +
> dispatch", which are host activities. Re-measured by regressing against node count *within*
> one arm, so the fixed cost cancels in the slope: **the marginal cost is 5.2 µs/node and
> essentially all of it is GPU.** Host cost per dispatch is **0.24 µs**, measured directly.
>
> The practical consequence is that §4's second requirement — *"per-dispatch host cost must
> fall from ~13 µs to ~1–2 µs"* — was aimed at a cost that does not exist. It is withdrawn.
>
> Both corrections come from `bench/latency_bench.py`, which exists because the original
> decomposition was done by hand with nothing committed and so could not be re-run.

---

## 1. The floor, and the gap

MNIST MLP 784-128-10 at batch 64. Forward is `64 × (784×128 + 128×10) × 2` = 13.0 MFLOP;
with backward, ~39 MFLOP. Weights and gradients are ~1.2 MB of traffic.

| | |
|---|---|
| Arithmetic floor at 6.9 TFLOP/s | **5.7 µs** |
| Bandwidth floor at 288 GB/s | **4.2 µs** |
| Measured step (best of 200, warm, validation off) | **1189 µs** |
| The same step with validation layers on | 1720 µs |

**200× above the floor.** PyTorch CPU does the same step in ~640 µs, so vkML's GPU is
1.9× slower on this workload while being 3.4× *faster* on CIFAR-100. The difference between
those two facts is the whole subject here.

## 2. The two constants that explain it

Measured with `bench/latency_bench.py`, which realises a dependent chain of *N* trivial
kernels as one submission and regresses wall time against *N*. The fixed cost cancels in the
slope, which is what makes the two separable at all:

```
  nodes    wall us     GPU us   GPU/wall
      1      64.91       7.96       0.12
      2      71.08      12.84       0.18
      4      89.38      23.96       0.27
      8     111.99      47.24       0.42
     16     147.80      94.24       0.64
     32     217.07     174.04       0.80

  per submission (fixed)   66.32 us wall,  3.31 us GPU
  per node (marginal)       4.83 us wall,  5.40 us GPU
```

> **A submission costs ~66 µs. A node costs ~5 µs and the GPU owns it.**
>
> These are the *before* numbers, kept because everything below is reasoning from them. §6 has
> what they became.

An MNIST step makes **seven** submissions — two uploads, one backward, three optimiser, two
for `.item()` — so roughly 460 µs of the 1189 is submission overhead before any arithmetic.

### The submission cost is the block, not the call

Timing `begin()` … `submit()` separately from the wait that follows, with no timestamps
anywhere, so nothing the profiler costs can leak in:

```
  dispatches   record + submit        block
           1         16.60 us      40.72 us
           2         16.98 us      46.27 us
           4         17.31 us      57.30 us
           8         18.32 us      62.50 us
```

```
  host per dispatch (record + barrier)     0.24 us
  host per submission (vkQueueSubmit2)    ~16.5 us
  blocking on completion                  ~40.5 us   of which ~8 us is real GPU work
```

`vulkan_synchronize()` on an already-idle device costs **1.95 µs**, so the wait *path* is
cheap. What is expensive is a wait that genuinely blocks: the host thread sleeps in
`vkWaitSemaphores`, the GPU finishes, an interrupt wakes it. **That round trip is ~32 µs and
is a property of the OS scheduler, not of vkML or of Vulkan.**

`Recorder::begin()` compounds it: there is **one command buffer**, so it must wait for the
previous submission to complete before it can be reset. Consecutive realises serialise
completely — host and GPU never overlap.

### The per-node cost is the barrier

The executor emits one global memory barrier after *every* node, because the graph gives it no
aliasing information. Sixteen dispatches writing **disjoint** buffers, one submission, GPU
`submit` window:

```
  elements   with barriers    no barriers   ratio
        64        79.20 us       40.56 us   1.95x
      1024        69.24 us       41.16 us   1.68x
    16384        71.52 us       40.96 us   1.75x
   262144       134.08 us       75.16 us   1.78x
```

**A barrier costs ~2.4 µs and roughly doubles the GPU time of small independent work.** Of
the 5.4 µs a node costs, about 2.4 µs is the barrier and about 2.5 µs is the dispatch.

`vk_command.h`'s strategy note says the always-barrier policy "is what llama.cpp does". It is
not. `ggml-vulkan.cpp` keeps `unsynced_nodes_written` / `unsynced_nodes_read`, compares
buffer ranges in `overlaps_unsynced`, and emits its global barrier **only when a real overlap
is found**. The barrier *form* vkML copied is llama.cpp's; the *policy* is the opposite of it.

### And the per-dispatch cost is not the pipeline key

Graph construction is **0.10 µs/node** in C++ and 0.34 µs through Python. `topological_order`
on a one-node graph is **0.27 µs** — not the 3.83 µs the first version attributed to "the
graph walk", which was that measurement's share of a submission it could not see.

`PipelineCache::get` looked like the culprit: it builds `name + ":" + config.key()` — a dozen
`std::format` calls and three heap allocations — on **every dispatch**. Replacing it with a
POD key and a hand-written hash was implemented and measured at **0.3% — indistinguishable,
ranges fully overlapping** — and reverted. Recorded so the next reader does not spend the same
afternoon on the same plausible suspect. With host cost per dispatch now measured at 0.24 µs,
that null result is exactly what should have been expected.

## 3. What 15–20× would require, arithmetically

PyTorch CPU is 640 µs/step. The target is **32–43 µs/step**.

> **One submission costs ~66 µs today and ~17 µs at best. The target is below the cost of a
> single blocking submission.**

So the target is unreachable by any amount of kernel work or shape dispatch. It requires:

1. **The host must stop blocking per step.** Submit and continue; block only when a value is
   read. Then the step costs `max(host, GPU)` rather than `host + GPU`. This needs a **ring
   of command buffers** — with one, `begin()` must wait — and a cross-submission hazard
   argument, because vkML currently relies on being fully synchronous.
2. **Fewer submissions.** Once the block is gone, `vkQueueSubmit2` itself is the floor at
   ~16.5 µs, and seven of them is 116 µs — still over the target on its own.

~~3. Per-dispatch host cost must fall from ~13 µs to ~1–2 µs.~~ **Withdrawn.** It is 0.24 µs.

And even with all of it, the training loop calls `loss.item()` every step, which forces a sync
no matter how asynchronous the backend is. Reaching the target on *this* workload additionally
requires not reading the loss every step.

## 4. What the alternatives are worth, measured

Prototyped against raw Vulkan before anything was rewritten, so the ceiling was known first.
Minimum of three independent process launches, one trivial dispatch per submission:

| model | µs/submission | against today |
|---|---|---|
| one command buffer, submit + wait (today) | 59.3 | — |
| ring of 2 command buffers, no wait | 36.3 | 1.6× |
| **ring of 4, no wait** | **17.7** | **3.4×** |
| ring of 8, no wait | 17.4 | 3.4× |
| record once, re-submit the same buffer (replay) | 15.2 | 3.9× |

Two things this settles:

- **A ring of 2 is not enough and a ring of 8 is no better than 4.** With two slots the host
  comes back round to a buffer that is still in flight and blocks anyway. Four is the design
  point, measured rather than assumed.
- **Replay is not the lever.** Re-recording a one-dispatch buffer costs ~2 µs; the submission
  costs ~17 µs whether or not anything was recorded. Command-buffer replay — the CUDA Graphs
  and tinygrad-JIT model — buys ~1.14× on top of the ring here, and it would cost fixed buffer
  addresses across steps and an invalidation rule. **Deferred, with its number**: it becomes
  interesting only once submissions are batched enough that recording is a real share.

## 5. What this means for the objective

- **On CIFAR-100 the GPU already wins by 3.4×** (8.07 ms/step against PyTorch CPU's 27.1 ms).
  That workload reaches the arithmetic; its remaining gap is `matmul` at 30.9%, which is
  `M3_ROADMAP`'s subject and a genuinely arithmetic problem.
- **On MNIST at batch 64 the GPU cannot win by 15–20×**, because the step's total budget is
  smaller than one host-GPU round trip. It can plausibly be made **2–3× faster than today**
  by removing the block from the critical path — which would also take it past PyTorch CPU —
  but 32 µs is below the floor set by the OS wake-up latency.
- **The measurement generalises**: any workload whose per-step GPU time is under ~100 µs is
  dominated by this constant, and every such workload will show the same shape. The batch-size
  sweep in `EXTENSIBILITY-ROADMAP.md` §4a is the same effect seen from the other end — the two
  GPUs stop tying above batch 256 precisely when the arithmetic grows past the constant.

## 6. Asynchronous submission — done, and what it moved

Implemented as `docs/adr/0012-asynchronous-submission.md`: a ring of four command buffers,
`realize()` submitting without waiting, a deferred-free queue, and a leading barrier carrying
the cross-submission ordering.

**End to end, against the frozen baseline at `d399fdd`** — a realistic MNIST step, DataLoader
and upload and `loss.item()` included, minimum of 200:

```
  before   1603 us
  after    1045 us      1.53x
```

The projection stated **before** the work was ~340 µs of 1189, "a 1.4× step". Measured after:

```
                                 before      after
  per submission (fixed)       66.32 us   19.52 us     3.4x
  per node (marginal, wall)     4.83 us    2.40 us
  per node (marginal, GPU)      5.40 us    5.35 us     unchanged

  MNIST MLP 784-128-10, b=64     1189 us     883 us     1.35x
```

The per-node GPU cost did not move, which is the control: nothing about the barriers between
nodes changed, so a figure that *had* moved would mean the measurement was wrong.

## 7. What is now the largest line

The host is no longer the per-node bottleneck — 2.40 µs of wall against 5.35 µs of GPU — so the
order of the remaining work has changed:

First, what the step is now made of. Measured without the profiler, so nothing can be
misattributed: time `realize()` alone, then time it followed by an explicit drain, and take the
difference. MNIST MLP at batch 64, minimum of 150, two independent runs:

```
  host (submit only)          408-428 us
  GPU on the critical path    416-429 us
  full step, with item()      885-894 us
```

**Host and GPU are now within a few percent of each other.** That is what the change bought:
before it, the two were strictly additive.

1. ~~**The barrier between every node.**~~ **Prototyped and rejected, measured.** llama.cpp's
   `overlaps_unsynced` — track written and read ranges, barrier only on a real overlap — was
   implemented behind a switch and is **bit-identical** on both workloads. It is worth
   **4% on an MLP step (885 → 850 µs) and 3% on a CNN step**, because the graphs are mostly
   dependency chains where nearly every barrier is load-bearing; only about a quarter are
   elidable. §2's 1.68–1.95× is the ceiling for *independent* dispatches and real graphs are
   not that.

   Deleted rather than kept behind the switch. A 4% gain does not pay for hazard tracking in
   the hottest loop whose failure mode is silent memory corruption — and, decisively, **it
   cannot be verified here**: removing barriers *entirely* still passes all 1552 tests on this
   driver (ADR 0012 §4b bis). An unverifiable safety mechanism guarding the project's defining
   guarantee is a bad trade at 4%. Revisit if a workload appears with genuinely wide
   independent work, or on a driver that can falsify it.
2. ~~**The upload, 352 µs.**~~ **Fixed, and it was a regression this change caused.**
   `StagingBuffer::upload` blocks so the staging memory is not overwritten while a copy still
   reads it, and it did that by waiting on the copy's own ticket *after* submitting. That is
   equivalent while every submission blocks anyway, and wrong once they do not: the timeline is
   monotonic, so waiting on a later ticket drains **every earlier submission**. An upload in the
   middle of a step drained the step.

   The wait belongs immediately *before* the next memcpy, which is the thing it actually
   protects. A single-chunk upload — every tensor smaller than the 32 MiB staging buffer — now
   blocks not at all.

   ```
     one 200 KiB upload, device idle           109 us
     the same, behind a step's queued work     477 us   ->   40 us
   ```

3. **`vkQueueSubmit2` at ~16.5 µs, six times per step.** With the block gone this is the floor,
   and the only lever left is fewer submissions — batching upload, backward and the optimiser
   into one. ADR 0006 already took this from 39 to 8.
4. ~~**The optimiser, 427 µs to update 101,770 parameters.**~~ **Partly fixed.** It issued 24
   dispatches for two lines of arithmetic, a third of them only to materialise a scalar as a
   rank-0 tensor. `scaled_add` — `a*alpha + b*beta` as one operator — takes it to 8 and
   458 µs → 309 µs. A whole-step `sgd_step` kernel measured better still (5.2× against 2.7×
   on GPU) and was rejected on maintainability; `docs/adr/0013` has both numbers and the trade.

   It is still **43× off its 7.1 µs bandwidth floor**, and 8 dispatches cost 189 µs here
   against 47 µs for the same 8 at the Recorder level. That gap is graph and allocation
   overhead, not kernels, and it is the next measurement.

5. **`loss.item()`, 371 µs of the 863.** Attributed by removal rather than by synchronising
   between phases, which would destroy the overlap being measured:

   ```
     loss.item()        371 us      optimiser step   295 us
     backward           149 us      zero_grad         36 us
     forward (build)     12 us      upload (when present)  352 us
   ```

   This is not a regression — `item()` is where the whole step's GPU work now lands, because
   nothing before it blocks. It is the honest shape of an asynchronous backend, and it says the
   remaining win on *this* workload is not reading the loss every step.

Command-buffer replay stays rejected on §4's number until submissions are batched enough that
recording is a meaningful share of one — at 0.24 µs a dispatch, about seventy of them.
