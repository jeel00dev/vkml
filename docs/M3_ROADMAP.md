# M3 Roadmap — Ranked Optimisation Plan

Derived from `docs/GAP_ANALYSIS.md`. Every entry is supported by at least one production
project; the supporting evidence is named in each row. Nothing here has been implemented,
and no benchmark in this document is a prediction — expected benefit is stated as a class,
not a number, except where a production project publishes a measurement.

## Scoring

**Benefit** — expected effect on vkML's real workloads (training small-to-medium nets, batch
1–64), not on the 1024³ square-GEMM benchmark alone.
**Complexity** — implementation effort in vkML's existing structure.
**Maintenance** — ongoing cost: kernels to keep validated, data to keep current.
**Portability** — how much survives a move to another GPU or vendor.
**Compatibility** — interaction with the BACKWARD tolerance policy and bit-reproducibility.

| # | Optimisation | Benefit | Complexity | Maint. | Port. | Compat. | Converged |
|---|---|---|---|---|---|---|---|
| 1 | **Dedicated GEMV kernel (N=1, small N)** | **Very high** | Low | Low | High | **Free** | 5/5 |
| 2 | **Runtime shape dispatch (s/m/l tiles)** | **High** | Low | Med | High | **Free** | 4/5 |
| 3 | **Autotuning + tuned-parameter store** | **High** | Med | Med | High | **Free** | 3/5 |
| 4 | Aligned / unaligned kernel variants | Med–high | Low | Med | High | **Free** | 3/5 |
| 5 | Disk-serialised pipeline cache | Med (latency) | Low | Low | High | **Free** | — |
| 6 | **Three-level tile hierarchy (warp tile, WMITER)** | **High** | **High** | Med | High | **Free** | 3/5 |
| 7 | LDS bank-conflict padding | Low–med | Low | Low | High | **Free** | 3/5 |
| 8 | Tuned vector width per operand | Low–med | Low | Low | High | **Free** | 3/5 |
| 9 | **Split-K / GlobalSplitU** | **High** (tall-skinny) | Med | Med | High | **Needs re-derivation** | 4/5 |
| 10 | Register-scope software pipelining | Med | High | Med | High | **Free** | 2/5 |
| 11 | Epilogue phase + fusion | **High** (training) | High | Med | High | **Free** | 3/5 |
| 12 | Threadblock swizzle (L2 locality) | Low–med | Low | Low | Med | **Free** | 2/5 |
| 13 | Operand pre-pass (repack/pad/transpose) | Med | Med | Med | High | **Free** | 2/5 |
| 14 | Carry stack relocation to LDS | Unknown | Med | Med | Med | **Free** | 0/5 |
| 15 | StaggerU (DRAM channel spreading) | Low | Low | Low | **Low** (AMD) | **Free** | 1/5 |
| 16 | Sliced-K / LocalSplitU | Low–med | Med | Med | High | **Needs re-derivation** | 2/5 |

Excluded entirely — see `GAP_ANALYSIS.md` §7: cooperative matrix (no RDNA1 hardware), fp16
accumulation (violates the numerical policy), quantised weight formats (inference-only),
Hopper-specific machinery.

---

## The workload this now has to be re-read against (measured 2026-08-02)

This document was written before vkML could attribute a training step, and its scoring says
*"expected effect on vkML's real workloads … not on the 1024³ square-GEMM benchmark alone"*
— an intention it had no way to check. It can be checked now, and it changes what the table
means.

`matmul` is **30.9% of a CIFAR-100 step and the largest line by a factor of five**
(`EXTENSIBILITY-ROADMAP.md` §4a), so this roadmap is no longer mis-sequenced. But the GEMMs
it has to serve are not the ones the items were researched against:

```
  the CNN's actual GEMMs        ms    GFLOP/s  % compute   GB/s  % mem  intensity  bound by
  conv1 (32,27)@(N,27,1024)  0.1425     794.8     11.5%   108.6  37.7%       7.3   memory
  conv2 (64,288)@(N,288,256) 0.3664    1648.4     23.8%    63.2  21.9%      26.1   compute
  conv3 (128,576)@(N,576,64) 0.3726    1620.8     23.4%    31.7  11.0%      51.1   compute
  head  (64,2048)@(2048,100) 0.0707     370.9      5.4%    19.4   6.7%      19.1   memory
  --- for reference ---
  square 1024                0.8975    2392.8     34.6%    14.0   4.9%     170.7   compute
  square 2048                6.7006    2563.9     37.1%     7.5   2.6%     341.3   compute
```

Roofs: ~6.9 TFLOP/s fp32 (36 CU × 64 lanes × 2 × 1.5 GHz) and 288 GB/s, so the memory roof
binds below ~24 flop/byte.

**Three findings, each of which reweights the table above.**

1. **Two of the four are memory-bound, and one is the smallest of them.** `conv1` has an
   arithmetic intensity of 7.3 — every tiling item in this roadmap is aimed at the compute
   roof and can do nothing for it. Its 108.6 GB/s against a 288 GB/s roof is a *bandwidth*
   problem, which is item 13's territory (operand layout) and nobody else's.
2. **The compute-bound ones run at 23–24% of peak where the square reference reaches
   34–37%.** So the headroom on `conv2`/`conv3` is real and is roughly **1.5×**, not the
   3–4× a naive reading of "23% of peak" suggests — the kernel's own demonstrated ceiling on
   this device is 37%, and closing to *that* is what items 2, 3 and 6 are for.
3. **Every one is batched with a broadcast operand, and M is 32–128.** The square benchmark
   has M = 1024 and no batch axis. Item 2 (runtime shape dispatch) and item 1 (GEMV) are
   scored against a shape distribution this workload does not have; a tile geometry chosen
   for M = 1024 is being asked to serve M = 32.

> **What that makes this roadmap.** Still correctly ordered — M3.1's shape dispatch is
> exactly what a workload with M ∈ {32, 64, 128, 1024} needs — but its *benefit* column was
> estimated against square shapes and should be re-read with the table above in hand. The
> first thing any of these items should do is reproduce that table, because it is now cheap
> to produce and was not when this was written.

---

## Recommended implementation order

The ordering is not the score ranking. Three constraints reshape it:

1. **Cheap and numerically free before expensive and load-bearing.** Items 1, 2, 4 change no
   floating-point ordering, so goldens stay pinned and the existing validation suite is
   sufficient as-is.
2. **Search before redesign.** Item 6 changes what the optimal tile size is. Doing it before
   item 3 means evaluating it by developer judgement — the method that produced the Stage 6.5
   false positive and the Stage 8 mis-prediction. Doing it after means evaluating it by search.
3. **Numerical work is done up front, not retrofitted.** Item 9's error re-derivation is a
   gate, not a follow-up.

### M3.1 — Kernel family and GEMV *(items 1, 2, 4)*

The single highest-value stage, and the one with the least risk.

- Introduce a shape-driven dispatch layer selecting among GEMM tile sizes and a GEMV path.
  llama.cpp's thresholds (`m <= 32 || n <= 32` → small; `m <= 64 || n <= 64` → medium)
  are a starting point, not a conclusion — item 3 will replace them with measured ones.
- Add a dedicated GEMV kernel for `N == 1` and small `N`. At N=1 the current 32×32 tile
  kernel computes 97 % padding.
- Add aligned variants that omit bounds checks when the shape divides the tile, selected by
  a spec constant as llama.cpp does with `ALIGNED`.

**Why first:** vkML's stated target workloads — chess-eval nets, small CNNs at batch 1–64 —
are dominated by exactly the shapes the current single kernel handles worst.
`ARCHITECTURE.md` §1.2 already records that this GPU is bandwidth-bound in that regime.
No fold order changes, so the BACKWARD policy and all pinned goldens are untouched.

**Gate:** every existing GEMM test passes unchanged on every dispatch path; the shape sweep
(1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 257) covers each selected kernel;
GEMV validated against PyTorch and the CPU oracle independently of GEMM.

### M3.2 — Autotuning infrastructure *(items 3, 5, 8)*

Convert the manual Stage 5–8 walk into a search.

- A search driver over the existing specialisation constants — `BM`, `BN`, `BK`, `RM`, `RN`,
  workgroup size, vector widths. These are already spec constants, so this is a search over
  what vkML has, not a rewrite.
- **Prune on `PipelineStats` before timing.** Reject candidates with non-zero
  `scratch_bytes` or `spilled_vgprs`, or with `max_waves` below a threshold, without ever
  dispatching them. This is the step no studied project can perform, and Stage 8 is the proof
  it works: the regression was fully visible in the statistics before any benchmark ran.
- Algebraic constraints to keep the space small, in CLBlast's style
  (`src/tuning/kernels/xgemm.hpp:160-208`): LDS must fit, `(BM/RM)·(BN/RN)` must equal the
  workgroup size, tile widths must divide vector widths.
- Persist results keyed on device name + problem shape class, as CLBlast and rocBLAS both
  ship. Serialise the `VkPipelineCache` alongside it so compilation is paid once across runs.

**Why second:** it makes every later stage measurable rather than argued. It also directly
addresses the recurring meta-failure documented across M1–M2 — eight cases where an apparent
implementation bug was a measurement error — by removing the developer from the measurement
loop.

**Gate:** the tuner independently rediscovers the Stage 6 configuration as optimal for
1024³ on this device. If it does not, the tuner is wrong, or Stage 6 was — and either answer
is worth having before proceeding.

### M3.3 — Three-level tile hierarchy *(items 6, 7, 10)*

The redesign that unblocks the register-block ceiling.

- Introduce a warp/subgroup tile between the threadblock tile and the thread tile, with a
  `WMITER`-equivalent so a subgroup can cover more output without proportionally more
  registers. llama.cpp uses precisely this knob to control register pressure for K-quants
  (`ggml-vulkan.cpp:4043`); vkML had no equivalent lever in Stage 8.
- Add LDS stride padding while the tile layout is being touched anyway.
- Evaluate register-scope double buffering (CUTLASS's second pipelining scope) **in this
  stage, not separately.** CUTLASS's stated rationale is that large accumulator tiles are
  what *create* the latency-hiding problem pipelining solves
  (`media/docs/cpp/efficient_gemm.md:131-155`). vkML's Stage 7 tested double buffering at 2×2,
  where occupancy was already 16 waves/SIMD — the regime where it has the least to offer.
  The Stage 7 negative result should be treated as untested at large tiles, not as settled.

**Why third:** it is the highest-complexity item that is still numerically free, and item 3
must exist first so its many possible geometries can be searched rather than guessed.

**Gate:** goldens byte-identical (fold order is unchanged — only the thread→output mapping
moves); `PipelineStats` shows zero scratch at the selected geometry.

### M3.4 — Split-K *(items 9, 14, 16)*

- **Numerical work first.** Derive the error bound for the split-K fold order and decide
  whether the BACKWARD policy holds unchanged, before any kernel is written. Splitting K
  across workgroups produces a deterministic tree, but a *different* tree — the existing
  bound does not transfer by inspection, and goldens will need re-pinning under a new RFC.
- Then the kernel: partial GEMM writing per-split results to a workspace, plus a
  deterministic reduction pass. This is CUTLASS's two-kernel structure
  (`efficient_gemm.md:168-200`) and llama.cpp's `split_k_reduce` (`ggml-vulkan.cpp:5226`).
- Trigger on occupancy, as llama.cpp does: split only when the tile grid leaves compute units
  idle (`:8452-8483`), which requires querying `VK_AMD_shader_core_properties2`
  `activeComputeUnitCount` — not currently in `DeviceInfo`.
- Split-K also **shortens the in-register carry stack** by `log2(split_k)`, which is the most
  promising route to the §1.1 finding. At K=1024, `split_k=8` moves the stack bucket from 6
  to 4. Item 14 (stack in LDS) can be evaluated in the same stage as the alternative, using
  `PipelineStats` to compare — it is a hypothesis with no production precedent and must be
  measured, not assumed.

**Why fourth:** it is the first item that touches numerics, and it benefits from the tuner
existing (the split-count heuristic is itself tunable) and from the tile hierarchy existing.

**Gate:** re-derived error bound documented before implementation; goldens re-pinned under a
new RFC; classic path (`split_k == 1`) byte-identical to M3.3.

### M3.5 — Epilogue and fusion *(items 11, 12, 13)*

- An epilogue phase that exchanges accumulators through shared memory before writing, giving
  coalesced stores and a place to fuse bias, activation, and scaling — CUTLASS treats this as
  a first-class component (`efficient_gemm.md:110-123`).
- Threadblock swizzle for L2 locality.
- Operand pre-pass for non-contiguous inputs, as both llama.cpp and CLBlast do rather than
  handling strides inside the mainloop.

**Why last, and why it may matter most:** for a *training* framework the fused
matmul→bias→activation path, and its backward counterpart, is worth more than the remaining
GEMM tuning — and no amount of GEMM optimisation reaches it. It is placed last only because
it depends on the kernel structure settling first.

---

## What this roadmap deliberately does not do

- **No fp16 accumulation**, at any tile size, for any speedup. It is how llama.cpp buys its
  throughput and it is incompatible with vkML's stated policy.
- **No cooperative-matrix work.** RDNA1 has none. The M3.3 hierarchy is what makes adopting
  it a substitution rather than a rewrite if vkML later targets RDNA3+ — llama.cpp uses the
  same `mul_mm.comp` for both paths.
- **No tolerance loosening** to enable any item here. Item 9 is gated on a re-derived bound,
  not a relaxed one.
- **No timing-only autotuning.** Every item in M3.2 gates on measured resources before it
  gates on wall time.

## Open questions this study did not settle

1. Whether the carry stack in LDS (item 14) performs. No precedent exists; it needs measuring.
2. Whether vkML's `Operand`-stride in-kernel approach or a repack pre-pass (item 13) wins on
   this GPU. Two projects chose repack; neither published the comparison.
3. Where the crossover between GEMV and small-tile GEMM actually falls on Navi 10. llama.cpp
   uses `N == 1` plus a `mul_mat_vec_max_cols` bound; CLBlast tunes a scalar threshold (384 on
   RX 5700). Both are device-specific numbers that must be measured, not copied.
4. Whether Stage 7's double-buffering negative result survives at larger tiles (M3.3).
