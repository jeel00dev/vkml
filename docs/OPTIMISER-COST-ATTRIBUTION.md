# Where an SGD step's time goes, item by item

**Date:** 2026-08-02 · **Hardware:** AMD RX 5600M (RDNA1), RADV, 36 CU, 4 MiB L2
**Workload:** SGD with momentum over the MNIST MLP's four parameters (101,770 elements,
1.94 MiB), 8 dispatches, 3 submissions, after `scaled_add` (ADR 0013).
**Status:** measurement only. Nothing was optimised. Validation layers off throughout
(`MEASUREMENT-AUDIT.md` §6c); every figure is a minimum of the stated repetitions.

---

## 0. The premise this began with was wrong, and it was mine

The investigation was framed as: *real optimiser path 189 µs, recorder/kernel path 47 µs,
therefore a 142 µs gap to attribute.* Both numbers appear in `docs/adr/0013` §5, and putting
them side by side was my error.

| | what it actually measured |
|---|---|
| **189 µs** | `drained − submit_only`: a **blocking wall-clock wait**, over **three** submissions, unprofiled |
| **47 µs** | a **GPU timestamp** `submit` window, over **one** submission, profiled |

`MEASUREMENT-AUDIT.md` rule 4 forbids comparing a profiled figure with an unprofiled one, and
this project's own handoff already records the specific trap — *"do not subtract a profiled GPU
total from an unprofiled wall clock"*. **Their difference is not a gap; it is two different
quantities.** The 189 µs also carries the ~43 µs host wake-up (§3, item 3), which is not GPU
work at all.

Re-measured, each figure internally consistent:

```
  host    (submit, nothing waited on, unprofiled)     120.1 us
  GPU     (sum of `submit` windows, profiled)         148.9 - 151.9 us
  drained (host + block, unprofiled)                  294.8 - 318.5 us
  bandwidth floor                                       7.1 us
```

Rule 3 permits summing whole-submit windows *across* submissions because submissions are serial.
**There is still a real gap** — 148.9 µs of GPU against 47 µs for the same eight dispatches run
synthetically — and §4 is what it turned out to be.

---

## 1. Two items are structurally zero

Not measured, because they do not exist. Verified by reading the tree, not by timing.

| Item | Finding |
|---|---|
| **4. Descriptor allocation / update** | **Zero, by construction.** There is no `vkAllocateDescriptorSets`, `vkUpdateDescriptorSets`, `VkDescriptorPool` or `vkCreateDescriptorSetLayout` anywhere in `src/` or `include/`. `vk_pipeline.cpp` creates every pipeline layout with `setLayoutCount = 0`. Buffers reach shaders as 64-bit device addresses in push constants. This is the single largest structural advantage vkML holds over a descriptor-set backend, and it removes an item that is a per-dispatch cost elsewhere |
| **8. Memory mapping / unmapping** | **Zero on the step path.** `vkMapMemory` is called once per *block* in `Allocator::create_block`, and only for `HostStaging` or `DeviceLocalMapped`. The optimiser's allocations are `DeviceLocal` and never mapped; blocks are created rarely because allocation is suballocated |

---

## 2. Host primitives, measured individually

Every per-call figure times a **batch of 1000** and divides — a single `vkCmd*` is far below the
clock's reliable resolution. Batch size is stated so the number can be checked.

| Item | Cost | Per step (×count) |
|---|---|---|
| **10. `vkResetCommandBuffer`** | 0.003 µs | 0.01 µs (×3) |
| `vkBeginCommandBuffer` + `vkEndCommandBuffer` | 0.250 µs | 0.75 µs (×3) |
| **1. `vkCmdBindPipeline`** | 0.005 µs | 0.04 µs (×8) |
| **1. `vkCmdPushConstants` (84 B)** | 0.007 µs | 0.06 µs (×8) |
| **1. `vkCmdDispatch`** | 0.048 µs | 0.38 µs (×8) |
| **11. `vkCmdPipelineBarrier2`, host side** | 0.064 µs | 0.51 µs (×8) |
| **1+11 → recording one dispatch + barrier** | **0.124 µs** | **1.0 µs total** |
| **5. `PipelineCache::get`, cache hit** | 0.366 µs | 2.9 µs (×8) |
| **6+7. `Allocator::allocate` + `free`** | 0.038 µs | 0.3 µs (×8) |
| **2. `vkQueueSubmit2`** (empty buffer) | **13.68 µs** | **41.0 µs (×3)** |
| `Recorder::begin()` | 3.71 µs | 11.1 µs (×3) |
| **3. `vkWaitSemaphores`, already signalled** | 1.82 µs | — |
| **3. `vkWaitSemaphores`, genuinely blocking** | 43.57 µs | ~43.6 µs (once) |

> **Command buffer recording is not a cost.** All eight dispatches and their barriers record in
> **1.0 µs**, 0.8% of the host time. `vkQueueSubmit2` alone is **41 µs — thirteen times the
> recording, and independent of what is in the buffer** (measured on an empty one: 13.68 µs
> against 15.09 µs with a dispatch).

### Where the rest of the host time is

Timed in Python, the three passes `Optimizer.step` performs:

```
  _plan: build 4 velocity nodes (Python)          9.4 us
  realize(state):   4 nodes -> 1 submission      37.9 us
  finish + realize(values): 4 -> 1 submission    44.4 us
  assign: 4 copies -> 1 submission               24.4 us
  ---------------------------------------------------
  total host                                    116.1 us
```

Each `realize` carries one Recorder cycle (`begin` 3.7 + record + `submit` 15.1 ≈ 19 µs), so:

```
  vkQueueSubmit2 (x3)                     41.0 us    35%
  Recorder::begin (x3)                    11.1 us     9%
  command recording (8 dispatches)         1.0 us     1%
  pipeline lookup (x8)                     2.9 us     2%
  allocation + free (x8)                   0.3 us     0%
  ---------------------------------------------------
  accounted at the Recorder               56.3 us    48%
  12. graph layer + Python (by difference) 59.8 us   52%
```

**Item 12 is 52% of the host time and has not been decomposed further.** It is
`topological_order`, `supports()`, `bind_storage`, the dispatch `switch`, `to_gpu_operand`, the
`assign` address lookups, and Python's own object churn. `_plan` alone — pure Python building
four nodes — is 9.4 µs of it.

---

## 3. GPU time, per dispatch

Read directly from the profile history of the best of 150 real steps. This is the measurement
that mattered, and the one I should have taken first.

```
  submission 1 (velocity pass)          92.84 us window
    scaled_add (128, 784)  STRIDED      78.84 us
    scaled_add (128,)                    5.80 us
    scaled_add (10, 128)                 5.76 us
    scaled_add (10,)                     5.88 us
  submission 2 (parameter pass)         56.08 us window
    scaled_add (128, 784)  contiguous   42.32 us
    scaled_add (128,)                    4.68 us
    scaled_add (10, 128)                 4.20 us
    scaled_add (10,)                     3.52 us
  ---------------------------------------------------
  total                                148.92 us
```

Two facts fall straight out:

- **The same shape costs 78.84 µs strided and 42.32 µs contiguous.** They are not a controlled
  pair — different operands, different cache state — but they are the same kernel on the same
  extent in the same step, and the difference is 36.5 µs, **25% of the step's GPU time**.
- **Six dispatches over ~1,500 elements cost 29.8 µs**, 20% of GPU time, for work whose
  bandwidth floor is under 0.1 µs. That is pure launch and barrier cost.

### 9. The assign copy is invisible to this instrument

The third submission is copies with no dispatches. `Recorder::submit` deliberately reports no
`submit` window for such a submission — the guard that prevents a download from wiping the
compute profile. So **the assign's GPU cost is not in the 148.9 µs above.** Measured separately,
by differencing a submission with and without the four copies:

```
  assign: 4 copies, 398 KiB, host side    20.2 us
  assign: 4 copies, GPU side              36.9 us      floor 2.8 us
```

**True GPU total for the step is therefore ~186 µs, not 149.** Any future report summing submit
windows over a step that contains copies is under-counting by the same mechanism.

---

## 4. The strided gradient: root cause, and why three benchmarks missed it

**Every weight gradient in vkML is a transposed view.** Measured:

```
  param 0 (128, 784)  grad (128, 784)  strides [4, 512] bytes   contiguous = False
  param 1 (128,)      grad (128,)      strides [4]              contiguous = True
  param 2 (10, 128)   grad (10, 128)   strides [4, 40]          contiguous = False
  param 3 (10,)       grad (10,)       strides [4]              contiguous = True
```

Strides `[1, 128]` in elements on a `(128, 784)` tensor means the buffer is physically
`(784, 128)` and viewed transposed: the matmul backward rule produces `(in, out)` and transposes
it to match the parameter, without copying. Adjacent flat indices therefore address 128 elements
apart, so adjacent lanes touch different cache lines.

`scaled_add` selects its `CONTIGUOUS` specialisation constant from the operands, and the trace
confirms the split — **exactly 2 of the 8 dispatches take the strided path, and they are the two
largest**:

```
  spec=[256,0,0]  (128, 784)     <- velocity update, transposed gradient
  spec=[256,1,0]  (128,)
  spec=[256,0,0]  (10, 128)      <- velocity update, transposed gradient
  spec=[256,1,0]  (10,)
  spec=[256,1,0]  (128, 784)     <- parameter update, contiguous
  spec=[256,1,0]  (128,)  (10, 128)  (10,)
```

### The index arithmetic is not the cost; the access pattern is

Isolated, at a size large enough that the clock is stable:

```
  scaled_add over (1024, 1024), 12 MiB of traffic
    both contiguous, CONTIGUOUS=1                     56.2 us   223.9 GB/s
    contiguous data, CONTIGUOUS=0 (index ALU only)    71.0 us   177.3 GB/s     1.26x
    second operand transposed                        618.3 us    20.2 GB/s    11.0x
```

Reproduced across runs (10.92× and 11.08×). **The `offset_from` integer division costs 1.26×;
the memory access pattern costs 11×.** That is the opposite of ADR 0011's finding for
`im2col`, where the arithmetic was the problem — and the reason to separate the two arms.

### Three synthetic benchmarks said the penalty was zero, and they were wrong

Reproducing the optimiser's exact 8-dispatch mix in C++ with the same strides measured **0.2 µs
of penalty (1%)**, at working sets of 1.2, 4.7 and 18.6 MiB. Two further hypotheses were tested
and disproven: fresh allocation each step (no effect, 0.97–1.13×) and idle host gaps between
submissions (no effect across a 0–200 µs stall, 52–53 µs throughout).

**The synthetic benchmarks kept the data L2-resident.** They ran the same buffers in a tight
loop; the real optimiser runs immediately after a forward and backward pass that has moved
megabytes of activations through a 4 MiB L2. The same contiguous `(128, 784)` dispatch measures
13.0 µs in the tight loop and 42.32 µs in situ — **3.3×, entirely cache and clock state.**

> **The lesson, and it is the same one `MEASUREMENT-AUDIT` §1 keeps teaching**: a synthetic
> benchmark that reuses its buffers measures a cache-resident workload. It is the right
> instrument for comparing two kernels and the wrong one for predicting a step. The per-dispatch
> profile of the real workload was the only instrument that answered this, and it cost less than
> any of the three benchmarks that missed it.

---

## 5. Full attribution

Percentages are of the **drained step** (294.8 µs), which is what a training loop actually waits
for. "Scales with size" means the cost grows with tensor elements.

| # | Item | Absolute | % step | % of the 142 µs | Scales? | Avoidable? | What would remove it |
|---|---|---|---|---|---|---|---|
| 2 | **`vkQueueSubmit2` × 3** | 41.0 µs | 13.9% | — | No | Partly | Fewer submissions: merge the velocity and parameter passes (blocked by `detach()` forcing a realise), and fold `assign` into the last op |
| 12 | **Graph layer + Python** | 59.8 µs | 20.3% | — | No | Partly | Fewer nodes; a planner that binds output storage to an existing buffer removes 4 of 8 nodes and the assign entirely |
| 3 | **Blocking wait** | 43.6 µs | 14.8% | — | No | **No** | OS wake-up. ADR 0012 already removed six of seven per step; the last one is `item()` |
| — | GPU: strided velocity update | 78.8 µs | 26.7% | — | **Yes** | **Yes** | A contiguous gradient from the matmul backward rule |
| — | GPU: contiguous parameter update | 42.3 µs | 14.4% | — | Yes | No | This is the real work, and it is at the device's practical rate |
| — | GPU: 6 small dispatches | 29.8 µs | 10.1% | — | No | Partly | Fewer, larger dispatches — a multi-tensor kernel over all four parameters |
| 9 | **Assign copy (GPU)** | 36.9 µs | 12.5% | — | **Yes** | **Yes** | In-place update: `dst` may alias `a` in `scaled_add`, so the copy exists only because the graph allocates a fresh output |
| 9 | Assign copy (host) | 20.2 µs | 6.9% | — | No | Yes | As above |
| 11 | Barriers (GPU) | ~2.4 µs each | — | — | No | Partly | Hazard tracking — measured at 4% overall and **rejected** in the previous session as unverifiable on this driver |
| 1 | **Command buffer recording** | 1.0 µs | 0.3% | — | No | No | Nothing. It is not a cost |
| 5 | Pipeline lookup | 2.9 µs | 1.0% | — | No | Yes | A POD key — already tried and reverted at 0.3%, indistinguishable |
| 6,7 | Buffer allocation + free | 0.3 µs | 0.1% | — | No | No | Nothing. Suballocation already works |
| 10 | Command buffer resets | 0.01 µs | 0.0% | — | No | No | Nothing |
| 4 | Descriptor allocation | **0** | 0% | — | — | — | Already absent by design |
| 8 | Memory mapping | **0** | 0% | — | — | — | Already per-block |

**The "142 µs gap" column is deliberately empty.** §0 shows the quantity it was computed from
does not exist; filling it in would propagate the error.

---

## 6. Roadmap, ranked by evidence

Expected speedup is on the **drained optimiser step** (294.8 µs) unless stated. Nothing below
has been implemented.

### R1 — Contiguous weight gradients · ~27% of the step · medium complexity · low risk

The matmul backward rule produces `(in, out)` and transposes it. One dispatch pays 78.8 µs where
the contiguous equivalent pays 42.3.

- **Benefit** up to 36.5 µs on this step, and it compounds: *every* elementwise op consuming a
  weight gradient takes the strided path, not only the optimiser. The penalty is 11× on a
  cache-resident-exceeding workload, so it grows with model size.
- **Cost** the fix must be upstream, in the backward rule. Inserting `.contiguous()` at the
  optimiser was measured and is **worse** — 177.1 µs against 151.1 — because the copy costs more
  than the strided access saves at this size.
- **Verification** easy: bit-identity against the current gradients, plus the existing
  torch step-for-step comparison.
- **Risk** low. It changes a layout, not an arithmetic order, so bit-identity should hold — but
  that needs checking, because a different reduction order in the GEMM would not.

### R2 — In-place parameter update · ~19% of the step · high complexity · high risk

`assign` copies 398 KiB back over the parameters at 36.9 µs of GPU and 20.2 µs of host, against
a 2.8 µs floor. `scaled_add` reads `a[i]` and writes `dst[i]` at the same index, so `dst` may
safely alias `a`.

- **Benefit** removes the copy, one of three submissions (13.7 µs), and four of eight
  allocations.
- **Cost** the executor allocates fresh storage for every computed node (`bind_storage`). Binding
  an output to an existing buffer is the M5 memory planner's job and changes a documented
  invariant — ADR territory, not a patch.
- **Risk** high. Aliasing is exactly the class of bug the previous session showed **cannot be
  verified on this driver**: removing barriers entirely still passed all 1576 tests.
- **Verification** hard, for the same reason.

### R3 — Merge the two compute submissions · ~5% · low complexity · low risk

The velocity and parameter passes are separate only because `detach()` on an uncomputed node
forces a realise, so `finish()` cannot run before the velocity pass is submitted. Restructuring
`Optimizer.step` to build both and realise once would save one `vkQueueSubmit2` (13.7 µs) plus a
`begin` (3.7 µs).

- **Cost** none architectural — it is a Python restructuring in one function.
- **Risk** low. **Recommended first**: it is the cheapest, and it is the only item that needs no
  decision from anyone.

### R4 — Multi-tensor kernel over all parameters · ~10% · medium · medium

Six dispatches over ~1,500 elements cost 29.8 µs. One kernel handling all four parameters would
make it one.

- **Cost** needs a device-side descriptor array, which is new machinery.
- **Note** ADR 0013 already rejected a fused `sgd_step` on maintainability grounds. This is the
  same trade one level up and should be judged against that decision, not separately.

### Not recommended

- **Command buffer recording, pipeline lookup, allocation, resets, descriptors, mapping.**
  Together they are **4.2 µs, 1.4% of the step.** The pipeline key was already tried and
  reverted at 0.3%. There is nothing here.
- **Barrier elision.** Measured at 4% overall in the previous session and rejected because no
  test on this driver can distinguish a correct implementation from a broken one.

---

## 7. What this investigation should change about how the next one is run

1. **Measure the real workload per dispatch before building a synthetic model of it.** The
   profile history answered in one run what three C++ benchmarks got wrong.
2. **A synthetic benchmark that reuses buffers measures a cache-resident workload.** State it, or
   arrange for the cache to be cold.
3. **Copy-only submissions emit no `submit` window.** Any total summed from submit windows over a
   step containing copies is under-counted — here by 36.9 µs, 20%.
4. **Never put two numbers in a table without checking they are the same quantity.** §0 exists
   because I did, in an ADR, and it framed this entire investigation.
