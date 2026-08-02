# Why an MNIST step costs 1.4 ms when its arithmetic is 6 µs

**Date:** 2026-08-02 · **Hardware:** AMD RX 5600M (RDNA1), RADV · **Status:** measured,
nothing implemented from it yet.

The objective this answers: *make the GPU 15–20× faster than PyTorch CPU on workloads where
the GPU should dominate*, with MNIST named as the case that is currently slower and treated
as a bug until proven otherwise.

**It is not a bug in any kernel, and no kernel work can fix it.** This document says what it
is instead, with the measurement for each claim, and what the target would actually require.

---

## 1. The floor, and the gap

MNIST MLP 784-128-10 at batch 64. Forward is `64 × (784×128 + 128×10) × 2` = 13.0 MFLOP;
with backward, ~39 MFLOP. Weights and gradients are ~1.2 MB of traffic.

| | |
|---|---|
| Arithmetic floor at 6.9 TFLOP/s | **5.7 µs** |
| Bandwidth floor at 288 GB/s | **4.2 µs** |
| Measured step (best of 200, warm) | **1406 µs** |

**140–240× above the floor.** PyTorch CPU does the same step in ~640 µs, so vkML's GPU is
2.2× slower on this workload while being 3.4× *faster* on CIFAR-100. The difference between
those two facts is the whole subject here.

## 2. Where the 1406 µs goes

Minimum of 200 warm steps, each phase timed separately:

```
  batch (DataLoader)      35 µs     2.5%
  upload (2 submissions) 216 µs    15.4%
  zero_grad               11 µs     0.8%
  forward (graph build)   19 µs     1.4%
  backward               499 µs    35.5%
  optimiser              421 µs    29.9%
  loss.item()            195 µs    13.9%
  ------------------------------
  total                 1406 µs
```

Nothing here is arithmetic. `forward` is 19 µs because it only *builds* a graph — the work
happens at the next realise.

## 3. The two constants that explain it

Measured directly rather than inferred from the table above.

```
  1 dispatch, 1 submission             69.8 µs
  16 dispatches, 1 submission         255.9 µs
  => marginal cost per dispatch        12.4 µs
  => fixed cost per submission         57.4 µs
  GPU time inside that 1 dispatch      13.4 µs
```

**A submission costs 57 µs of host time to carry 13 µs of GPU work.** An MNIST step makes
**seven** of them — two uploads, one backward, three optimiser, two for `.item()`.

### It is the block, not the call

`vulkan_synchronize()` on an already-idle device costs **1.95 µs**, so the wait *path* is
cheap. What is expensive is a wait that genuinely blocks: the host thread sleeps in
`vkWaitSemaphores`, the GPU finishes, an interrupt wakes it. That round trip is the bulk of
the 57 µs and is a property of the OS scheduler, not of vkML or of Vulkan.

`Recorder::begin()` compounds it: there is **one command buffer**, so it must wait for the
previous submission to complete before it can be reset. Consecutive realises serialise
completely — host and GPU never overlap.

### And the per-dispatch cost is not the pipeline key

Graph construction is **0.72 µs/node** — Python objects, C++ nodes, the lot. So the 12–14 µs
is inside `realize()`: topological order, storage binding, allocation, recording.

`PipelineCache::get` looked like the culprit: it builds `name + ":" + config.key()` — a dozen
`std::format` calls and three heap allocations — on **every dispatch**. Replacing it with a
POD key and a hand-written hash was implemented and measured:

```
  per-dispatch, string key (3 runs)   14.167  16.561  16.719 µs   min 14.167
  per-dispatch, POD key    (3 runs)   16.059  14.129  14.713 µs   min 14.129
```

**0.3% — indistinguishable, ranges fully overlapping.** Reverted: it added a 60-line key
struct and a hand-written hash for no measured gain, which is the trade §3 says loses.
Recorded so the next reader does not spend the same afternoon on the same plausible suspect.
**Where the 12–14 µs actually goes is not yet known** — the remaining candidates are
`topological_order`, per-node storage allocation (10,236 suballocations were counted over
640 realises), and the per-node `vkCmdPipelineBarrier2`.

## 4. What 15–20× would require, arithmetically

PyTorch CPU is 640 µs/step. The target is **32–43 µs/step**.

> **One submission costs 57–77 µs. The target is below the cost of a single submission.**

So the target is unreachable by any amount of batching, kernel work or shape dispatch. It
requires *both* of:

1. **The host must stop blocking per step.** Submit and continue; block only when a value is
   read. Then the step costs `max(host, GPU)` rather than `host + GPU`. This needs a **ring
   of command buffers** — with one, `begin()` must wait — and a cross-submission hazard
   argument, because vkML currently relies on being fully synchronous.
2. **Per-dispatch host cost must fall from ~13 µs to ~1–2 µs.** 14 dispatches × 13 µs = 182 µs
   on its own, which already exceeds the target four-fold.

And even with both, the training loop calls `loss.item()` every step, which forces a sync no
matter how asynchronous the backend is. Reaching the target on *this* workload additionally
requires not reading the loss every step.

## 5. What this means for the objective

- **On CIFAR-100 the GPU already wins by 3.4×** (8.07 ms/step against PyTorch CPU's 27.1 ms).
  That workload reaches the arithmetic; its remaining gap is `matmul` at 30.9%, which is
  `M3_ROADMAP`'s subject and a genuinely arithmetic problem.
- **On MNIST at batch 64 the GPU cannot win by 15–20×**, because the step's total budget is
  smaller than one host-GPU round trip. It can plausibly be made **4–8× faster than today**
  by removing the block from the critical path — which would also take it past PyTorch CPU —
  but 32 µs is below the floor set by the OS wake-up latency.
- **The measurement generalises**: any workload whose per-step GPU time is under ~100 µs is
  dominated by this constant, and every such workload will show the same shape. The batch-size
  sweep in `EXTENSIBILITY-ROADMAP.md` §4a is the same effect seen from the other end — the two
  GPUs stop tying above batch 256 precisely when the arithmetic grows past the constant.

## 6. The next concrete step

**Asynchronous submission**, and it is an ADR before it is code, because it changes a stated
invariant — `Recorder` is documented as synchronous and `dispatch/executor.h` relies on it.
Its parts, in order:

1. A **ring of command buffers** so `begin()` need not wait for the previous submission.
2. `realize()` **submits without waiting**; `copy_to_host` and `item()` wait first.
3. A **cross-submission hazard argument**. Within a submission the global barrier orders
   everything; between submissions on one queue, ordering needs stating rather than assuming.
   `docs/adr/0006` §8 records that vkML's synchronisation cannot be machine-checked — it uses
   buffer device addresses, so the validation layer sees no resource access — so this argument
   has to be made on the specification and held by construction.

Expected effect, from the numbers above: the seven blocking waits per step collapse to one,
worth roughly 340–460 µs of the 1406. That is a **1.9–2.4× step**, not 15×, and the honest
projection should be stated before the work rather than after.
