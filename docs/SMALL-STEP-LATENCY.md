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

## 6. The next concrete step

**Asynchronous submission**, and it is an ADR before it is code, because it changes a stated
invariant -- `Recorder` is documented as synchronous and `dispatch/executor.h` relies on it.
Its parts, in order:

1. A **ring of command buffers** so `begin()` need not wait for the previous submission.
   Depth 4, from the table in §4.
2. `realize()` **submits without waiting**; `copy_to_host` and `item()` wait first.
3. A **deferred-free queue**. Under the synchronous model an allocation could be released the
   moment its `Storage` died, because nothing was ever in flight. Asynchronously it can be
   handed to a new tensor while a running dispatch still writes it.
4. A **cross-submission hazard argument**. Within a submission the global barrier orders
   everything; between submissions on one queue, ordering needs stating rather than assuming.
   `docs/adr/0006` §8 records that vkML's synchronisation cannot be machine-checked -- it uses
   buffer device addresses, so the validation layer sees no resource access -- so this argument
   has to be made on the specification and held by construction.

Expected effect, from §4: seven blocking submissions per step at ~66 µs become seven
non-blocking ones at ~17 µs, worth roughly **340 µs of the 1189**. That is a **1.4× step**,
and the honest projection is stated before the work rather than after. The step does not get
7× faster because the block is not the only thing in it -- §2's table accounts for 460 µs of
1189, and the rest is upload, autograd graph construction and Python.
