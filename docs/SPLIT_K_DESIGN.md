# M3-02 — Split-K Feasibility and Numerical Design

Research and design only. No production code, shader, or kernel was modified. No hypothetical
kernel was benchmarked; the measurements below are of the **existing** kernel, used to size the
opportunity.

---

## 0. Conclusion first

**Split-K is compatible with vkML's numerical guarantees and should be implemented.**
Success criterion 1.

Stronger than expected, and it corrects a claim I made in the M3-01 report:

> M3-01 stated: *"split-K genuinely reorders the fold"* and required a re-derived BACKWARD
> bound before implementation. **That was wrong.** Split-K reorders the fold only if it is
> designed the way production libraries design it. Under one specific constraint — derived in
> §2 and verified in §3 — split-K in vkML is **bit-identical to the current kernel**, for any
> K, for any partition count.

The constraint has two parts, and both are cheap:

1. Each partition must cover exactly **2^q k-tiles** (equal chunks; the last may be short).
2. The final reduction must apply **vkML's own carry-stack algorithm** to the partials — *not*
   the flat sequential sum that llama.cpp, CUTLASS and Tensile all use.

Consequently: no goldens change, no RFC, no tolerance discussion, no new error analysis. The
BACKWARD bound is not re-derived because the arithmetic is unchanged.

The performance case is measured and large: at 1 % GPU fill the current kernel achieves
**42.6 GFLOP/s against 1356.7 at full fill — a 32× gap** that split-K exists to close.

---

## 1. Research summary

| Project | Split-K? | Selection | Accumulation | Deterministic | Workspace |
|---|---|---|---|---|---|
| **llama.cpp** | ✓ | runtime | separate `split_k_reduce` kernel; flat sequential loop over partials | **✓** | `S·M·N·4` |
| **CUTLASS** `kSerial` | ✓ | compile-time template | in-place, ordered by **semaphore** on `k` index | **✓** | semaphore only |
| **CUTLASS** `kParallel` | ✓ | compile-time template | separate batched reduction kernel | **✓** | `S·M·N·4` |
| **Tensile** `MultipleBuffer` | ✓ | offline-tuned, per shape | `postGSU` reduction kernel | **✓** | `Tc·GSU·M·N` |
| **Tensile** `SingleBuffer`+atomic | ✓ | offline-tuned | `atomic_add` float | **✗** | `Tc·M·N` |
| **tinygrad** `GROUP`/`GROUPTOP` | ✓ (local) | **beam search** | LDS tree inside the workgroup | ✓ | LDS |
| **CLBlast** | **✗** | — | — | — | — |

### 1.1 Findings

**Determinism is not the obstacle.** Every production split-K is deterministic except
Tensile's atomic path, which Tensile's own source annotates `NOTE: This is not recommended`
(`Tensile/Common.py:605`). The common design is a workspace plus a second kernel. CUTLASS's
serial variant is deterministic by a different route — `semaphore.wait(threadblock_tile_offset.k())`
(`gemm.h:346`) forces partitions to accumulate strictly in ascending `k`.

**But none of them is order-preserving.** All of them fold partials *sequentially*:
`((p₀+p₁)+p₂)+…`. llama.cpp's reduce kernel is explicit
(`mul_mat_split_k_reduce.comp`): one invocation per output, `for i in 0..k_num: result += data_a[...]`.
This is deterministic — the same order every run — but it is a **different order** from the
unsplit kernel. For llama.cpp that is free, because it has no order guarantee to lose. For
vkML it would break bit-identity, and §3 measures exactly that.

**When it is selected** (llama.cpp, `ggml-vulkan.cpp:8444-8483`) — the only runtime rule in
the set, and worth copying in structure:

```
if K >= 2048 and m_tiles·n_tiles <= shader_core_count/2:
      split_k = shader_core_count / (m_tiles·n_tiles)
elif  m_tiles·n_tiles <= shader_core_count·2/3:
      split_k = 3
split_k = min(split_k, 8)                       # "Unless k is huge this is a lot of overhead"
while k_split·(split_k-1) < K: split_k--        # never leave the last split empty
```

Two things matter here: the trigger is **occupancy**, not shape *per se*, and the cap is small
(8). Tensile permits `GlobalSplitU` up to 4096 but its shipped navi21 SGEMM logic uses
`GlobalSplitU: 1` for the entries inspected — large splits exist but are rarely selected.

**Partition assignment.** Tensile parameterises this
(`GlobalSplitUSummationAssignmentRoundRobin`): `False` gives each partition a contiguous
chunk `k = s·K/S … (s+1)·K/S-1`; `True` interleaves `DepthU`-sized chunks round-robin. vkML
must use **contiguous** (`False`-equivalent) — §2 shows round-robin destroys the tree
alignment.

---

## 2. Mathematical analysis

### 2.1 vkML's current fold

For one output element, `gemm_reg.comp` computes `T = ⌈K/BK⌉` k-tile partial sums
`t₀ … t_{T-1}`, each an in-order sequential sum of `BK = 32` products, and folds them with a
binary-counter carry stack:

```
push(v):  c ← count;  ℓ ← 0
          while c odd:  v ← v + stack[ℓ];  c ← c>>1;  ℓ ← ℓ+1
          stack[ℓ] ← v;  count ← count+1

drain():  total ← 0
          for ℓ ascending: if bit ℓ of count set: total ← total + stack[ℓ]
```

**Invariant.** `stack[ℓ]` holds the perfectly-balanced sum of an *aligned* run of exactly `2^ℓ`
consecutive tiles. The drain then folds the set bits ascending — most-recent (smallest) group
first. Write the resulting value `C(t₀…t_{T-1})`.

### 2.2 Split-K formally

Partition `K` into `S` contiguous chunks. Let chunk size be `c` k-tiles, so chunk `s` covers
tiles `[s·c, min((s+1)·c, T))` and `S = ⌈T/c⌉`. Each partition computes an independent partial
GEMM producing `p_s`, and a reduction combines them into `D`.

`p_s` is computed by the *same kernel* on a shorter K range, so `p_s = C(t_{s·c} … )`.

### 2.3 The alignment lemma

> **Lemma.** If `c = 2^q`, then the carry stack's behaviour inside chunk `s` is independent of
> `s`.

*Proof.* At chunk start `count = s·2^q`, whose low `q` bits are zero. The push loop folds while
the low bit of `c = count` is set, so during the first `2^q - 1` pushes of the chunk no fold can
propagate past level `q-1`. The chunk's internal association therefore depends only on its
position *within* the chunk, not on `s`. ∎

So a chunk of `2^q` tiles always produces exactly the same subtree the unsplit kernel would
have built over those same tiles, and a short final chunk produces exactly the low-level stack
entries the unsplit kernel would hold at that point.

### 2.4 The equivalence theorem

> **Theorem.** For `c = 2^q` and a reduction that applies the *same* push/drain algorithm to
> `p₀ … p_{S-1}` in index order,
> ```
>     C(p₀ … p_{S-1})  =  C(t₀ … t_{T-1})     bit-for-bit
> ```
> for every `T` and every `q ≤ log₂T`.

*Sketch.* By the lemma each `p_s` equals the unsplit kernel's aligned subtree over its chunk.
The unsplit stack after `T` tiles decomposes uniquely into aligned subtrees whose sizes are the
set bits of `T`; each such subtree is a balanced combination of whole chunks (plus, for the low
bits, the short final chunk). Applying the push/drain algorithm to the chunk sums reproduces
exactly that decomposition, because the algorithm's association depends only on the *count* of
items pushed, and `count_chunks = ⌈T/2^q⌉` has the same high-bit structure as `T` shifted right
by `q`. The ascending drain visits groups in the same relative order in both cases. ∎

**Worked example** (`T=12`, `q=2`, `S=3`):

```
unsplit:  count=12=1100₂ → drain levels 2,3 → tree(8..11) + tree(0..7)
chunked:  p₀=tree(0..3) p₁=tree(4..7) p₂=tree(8..11)
          push→ stack[1]=p₀+p₁, stack[0]=p₂, count=3=11₂
          drain levels 0,1  → p₂ + (p₀+p₁)  =  tree(8..11) + tree(0..7)   ✓
```

### 2.5 What breaks the theorem

- **Non-power-of-two chunk size.** The lemma fails; `count` at a chunk boundary has set low
  bits and folds cross the boundary.
- **Round-robin partition assignment** (Tensile's `GSUSARR=True`). Chunks are no longer
  contiguous, so no partition owns an aligned run.
- **Flat sequential reduction of partials** (llama.cpp, CUTLASS, Tensile). Produces
  `((p₀+p₁)+p₂)+…`, a left-leaning chain, not the balanced decomposition.
- **Unequal interior chunks.** Only the *last* chunk may be short.

### 2.6 Which rounding points change

**None**, under the theorem. Every partial sum in the split computation is the value of a node
in the unsplit computation's tree; every addition is between the same two operands as before.
The only new operations are the *stores and loads* of the `p_s` values, which are exact fp32
round-trips through memory.

### 2.7 Is a new BACKWARD analysis required?

**No.** Accumulation depth is unchanged: `B = 32` sequential within a tile, plus `⌈log₂T⌉`
tree levels, exactly as today. The existing bound `|computed − exact| ≤ γ·Σ|terms|` with
`γ ≈ (B + log₂(n/B))·ε` holds with the same γ because the operation tree is the same tree.

*For completeness*, if a future need forced an unaligned split, the damage would be bounded:
the depth becomes at most `⌈log₂(T/S)⌉ + ⌈log₂S⌉ ≤ ⌈log₂T⌉ + 1`, i.e. **one extra level**, so
γ would grow by one ε. That is a small, derivable change — but it is not needed and should not
be taken.

---

## 3. Numerical verification

The theorem was checked in fp32 before being trusted, simulating only the fold order (no GPU,
no kernel):

```
power-of-two chunks:      496 cases, 0 mismatches
non-power-of-two chunks:  253 mismatches   (expected — §2.5)
flat sequential reduce:   126 mismatches   (expected — §2.5)
```

Cases span `T = 1…79, 96, 128, 129, 192, 256, 257, 384, 512` — deliberately including
non-power-of-two `T` — against chunk sizes `1…128`. Zero mismatches for every aligned case.

Both negative controls fire, which is what makes the positive result meaningful: the test can
distinguish the correct design from the two obvious wrong ones.

| Property | Status under the proposed design |
|---|---|
| Deterministic execution | **preserved** |
| Run-to-run reproducibility | **preserved** |
| Pairwise reduction guarantee | **preserved — same tree** |
| Bit-identity with current kernel | **preserved** |
| BACKWARD policy | **unchanged, not re-derived** |
| Validation tolerances | **unchanged; none weakened** |
| Golden hashes | **unchanged** |

**Recommendation: preserve current guarantees.** No weakening is required, so none is
justified. This is the rare case where the conservative option is also the fast one.

---

## 4. Performance model

### 4.1 Measured opportunity (existing kernel)

Fill = `tiles / (36 CUs × 8 concurrent workgroups/CU)`, the independent-barrier-domain capacity
established in `PERFORMANCE-MODEL.md` §5e.

```
     M      K      N   tiles   fill   gpu_min   GFLOP/s
  2048    256   2048    4096  1422%   1.583ms    1356.7
  1024   1024   1024    1024   356%   1.754ms    1224.1
   512   4096    512     256    89%   2.086ms    1029.4
   256  16384    256      64    22%   1.735ms    1237.9
   256   4096    256      64    22%   0.704ms     762.2
   128   8192    128      16     6%   1.454ms     184.6
    64  16384     64       4     1%   3.153ms      42.6
```

Throughput collapses with fill below ~90 %. The 4-tile case runs at **3 % of the
full-fill rate**. These are precisely the shapes split-K addresses, and they are not exotic:
**the weight-gradient GEMM in training is exactly this shape.** `dW = Xᵀ·dY` has `K = batch`
(large) and `M,N = in,out features` (small). A 64-wide layer trained at batch 16384 is the last
row of that table.

Note `256×16384×256` reaches 1237.9 GFLOP/s at the same 22 % fill where `256×4096×256` reaches
762.2. Fill is necessary but not sufficient — with very large K the per-workgroup prologue
amortises. **Split-K should therefore be triggered on tile count, not on K alone**, and K
should act only as a floor (§6).

### 4.2 Expected effects

| Quantity | Effect | Reasoning |
|---|---|---|
| **Carry-stack depth** | `⌈log₂T⌉+1` → `q+1` | chunk of `2^q` tiles. K=1024, chunk=4 → bucket 6 → 4 |
| **Register pressure** | **down** | stack is `RM·RN·STACK_LEVELS` floats; 2×2 saves `4·2 = 8` |
| **Occupancy** | likely unchanged | `max_waves=16` is LDS-bound, not VGPR-bound (M3-01 §6) |
| **Scratch risk** | **down** | the Stage 8 spill was the stack; a shorter stack can only help |
| **Arithmetic intensity** | unchanged | `BM/BN/BK` untouched |
| **Memory bandwidth** | **up** by `(2S−1)·M·N·4` bytes | write S partials, read them back |
| **Synchronisation** | +1 dispatch barrier | GEMM → reduce, once per matmul |
| **Parallelism** | **×S workgroups** | the entire point |

The bandwidth cost is small exactly where split-K is wanted. For `64×16384×64` at `S=64`:
`2·64·64·64·4 = 2 MB` → ~0.007 ms at 288 GB/s, against a current runtime of 3.153 ms. For
`2048×2048` at `S=8` it would be 268 MB — which is why §6 caps the workspace, and that cap
binds exactly where split-K is not needed anyway.

### 4.3 Where split-K helps, and where it must never be used

| Workload | Verdict |
|---|---|
| Small M·N, large K (weight gradients, thin layers) | **primary target** — up to ~30× headroom |
| Large square GEMM (1024³, 2048³) | **never** — already 356 %/1422 % fill; pure overhead |
| Skinny M or N with large M·N product | no benefit; enough tiles already |
| Small K | **never** — `T < 2·chunk` leaves nothing to split |
| GEMV (N=1) | **never** — a dedicated kernel is the right answer (M3.1) |
| Batched GEMM with many batches | **never** — batch already supplies workgroups |

### 4.4 Does this change the performance model?

Yes, in one respect: §4.1 quantifies the **fill → throughput curve**, which §5e of
`PERFORMANCE-MODEL.md` asserted qualitatively but never measured. `PERFORMANCE-MODEL.md` is
updated with that table. (The brief names `docs/PERFORMANCE_MODEL.md`; the file in the tree is
`docs/PERFORMANCE-MODEL.md` and that is the one updated.)

---

## 5. Architectural design

### 5.1 Kernel structure — two kernels, one new shader

**Kernel A — partial GEMM.** `gemm_reg.comp` *unmodified*. Split-K is expressed entirely
through existing parameters:

- a new push-constant pair `k_begin`, `k_end` bounding the k-tile loop, **or** (preferred,
  zero shader change) reuse the existing `k` push constant and offset the `A`/`B` base
  addresses by `k_begin`, since operands are already addressed through 64-bit device pointers
  with per-operand strides;
- `STACK_LEVELS` specialised to `q+1` rather than `⌈log₂T⌉+1`;
- destination is the workspace slice `W[s]` rather than `D`, when `S > 1`.

The grid becomes `tiles_m × tiles_n × S`, with `s = workgroup.z` (or folded into the existing
linear decomposition alongside the batch dimensions).

**Kernel B — `gemm_split_k_reduce.comp` (new, ~40 lines).** One invocation per output element:

```
total ← carry_fold( W[0][idx], W[1][idx], …, W[S-1][idx] )   // push/drain, §2.1
D[idx] ← total
```

This is the one place vkML must **not** follow the references. llama.cpp's reduce is a flat
`result += data_a[i*ne + idx]` loop; substituting it here costs bit-identity (§3, 126
mismatches). The carry stack for `S ≤ 16` needs 5 registers and no LDS.

Vectorisation over 4 consecutive outputs (as llama.cpp does) is available later and is
order-neutral — each output element folds independently.

### 5.2 Dispatch and synchronisation

```
if S == 1:  dispatch gemm_reg → D                    (today's path, byte-identical)
else:       dispatch gemm_reg  → W, grid ×S
            barrier(shader-write → shader-read)
            dispatch gemm_split_k_reduce W → D
```

A single `vkCmdPipelineBarrier2` between the two dispatches; vkML's `Recorder` already emits
these. Both dispatches land in the same command buffer, so no host round-trip and no extra
submit. Timestamp queries wrap each dispatch separately, so the reduction's cost is measured
rather than inferred.

### 5.3 Workspace

Backend-internal, not a graph-planner concern:

- Allocated from the existing `vk::Allocator` (`MemoryKind::DeviceLocal`), cached on the
  backend `Impl` and grown on demand — the same lifetime strategy as llama.cpp's
  `prealloc_split_k`.
- Size `S · M · N · 4` bytes, reused across dispatches; freed with the backend.
- Never observable above `backend/`, so **no public API change** and no change to the
  `plan/` refcount simulation. This keeps the layering check clean at 53 files.

### 5.4 What does not change

No public API. No `KernelConfig` change. No autograd, graph, or dispatch-layer change. No
change to `gemm_tiled`, `gemm_naive`, or `gemm_db`. `VKML_GEMM_BLOCK` and `VKML_GEMM_TILE`
keep working.

---

## 6. Runtime dispatch rules

Split-K must be **off by default and enabled only on evidence**. Proposed rules, each with its
justification:

| # | Rule | Justification |
|---|---|---|
| 1 | `chunk = 2^q` k-tiles; `S = ⌈T/chunk⌉` | **Required** for bit-identity (§2.4). Note this inverts the production order: llama.cpp picks `S` then derives the chunk; vkML must pick the chunk then derive `S`. |
| 2 | Only if `tiles < CU_count · 2` | Occupancy is the trigger (llama.cpp `:8452`). Needs `shader_core_count`, not yet queried — add via `VK_AMD_shader_core_properties2` `activeComputeUnitCount`, fallback 16 as llama.cpp does. |
| 3 | Only if `T ≥ 2·chunk`, i.e. `S ≥ 2` | A single partition is the unsplit path. |
| 4 | `chunk ≥ 4` k-tiles (K ≥ 128 per split) | Below this each partition is prologue-dominated; §4.1 shows large K amortises and small K does not. |
| 5 | `S ≤ 16` | llama.cpp caps at 8 — *"Unless k is huge this is a lot of overhead"*. 16 chosen because rule 1 forces coarser granularity than llama.cpp's free choice of S. |
| 6 | `S·M·N·4 ≤ 64 MiB` | Bounds workspace on a 5.75 GiB device. Binds only for large M·N, where rule 2 already declines. |
| 7 | fp32 only, contiguous output | Matches the current kernel's guarantees. |

Selection then reduces to: choose the **largest** `q` satisfying rules 3–6 that brings
`tiles·S` to at least `CU_count·2`. Largest `q` means fewest partitions, hence least workspace
and least reduction traffic, for the required parallelism.

`VKML_GEMM_SPLITK=<n>|off` should force or disable it, matching the project's existing
experiment handles.

---

## 7. Compiler predictions

Stated now so they can be checked before benchmarking, per the M3 workflow. Baseline is the
current 1024³ pipeline: `VGPR=41 SGPR=35 LDS=8192 waves=16 scratch=0 instr=1124`.

**Kernel A** (`gemm_reg`, `STACK_LEVELS` 6 → 4 at K=1024, chunk=4):

| Statistic | Prediction | Confidence |
|---|---|---|
| VGPR | **33–39** (8 fewer stack floats at 2×2, minus allocator rounding) | high |
| SGPR | ≈35, unchanged | high |
| LDS | 8192, unchanged | certain — `BM/BN/BK` untouched |
| `max_waves` | **16, unchanged** | high — occupancy is LDS-bound, not VGPR-bound |
| scratch | **0** | high |
| instructions | slightly **down** (shorter drain loop) | medium |

**Kernel B** (`gemm_split_k_reduce`, new): VGPR < 20, LDS 0, scratch 0, `max_waves` at the
device maximum, instructions < 200.

**Falsifiable and checkable later:**

1. VGPR falls on kernel A — if it does not, the carry stack is not costing what §4.2 claims.
2. `max_waves` stays 16 — if it rises, occupancy was VGPR-bound after all and M3-01 §6's
   analysis needs revisiting.
3. scratch stays 0 at 2×2, **and** the 4×4 register block's 24 KiB scratch (Stage 8) shrinks or
   disappears at `STACK_LEVELS=4`. This is the test of whether split-K reopens the register
   block — the strongest secondary claim available, and it is **not** assumed here.

---

## 8. Validation plan

Success criteria defined before implementation.

**Gate 1 — bit-identity (the decisive test).** SHA-256 over concatenated outputs must be
**identical** with split-K forced on and off, over a shape matrix crossing:
`K ∈ {32, 33, 96, 128, 129, 512, 1024, 4096, 16384, 65537}` × `M,N ∈ {1, 7, 32, 64, 127, 256}`
× `S ∈ {2, 4, 8, 16}`. This subsumes the tolerance question entirely: identical bytes cannot
have a tolerance failure. §3 predicts zero mismatches; a single mismatch **blocks the merge**
and means the theorem or its implementation is wrong.

**Gate 2 — goldens unchanged.** Existing pinned hashes must not move. Any movement is a bug,
not a re-pin.

**Gate 3 — oracles.** Full CPU-oracle and PyTorch comparison under the unchanged BACKWARD
policy, on the same shape matrix. **No tolerance may be adjusted.**

**Gate 4 — determinism.** 100 repeats of the same split-K matmul, byte-identical every time;
and identical across `S` values, which is strictly stronger than run-to-run stability.

**Gate 5 — pathological shapes.** `K` not a multiple of `BK`; `K` smaller than one chunk;
`T` not a power of two (the case §2.4 says still works and naive intuition says should not);
`S` larger than `T`; `M=1`; `N=1`; zero-size dims; the largest K that fits the workspace cap.

**Gate 6 — resources.** `PipelineStats` for both kernels must show zero scratch and zero
spilled VGPRs before any timing is interpreted (`PERFORMANCE-MODEL.md` §5d).

**Gate 7 — no regression when off.** With split-K declined by rule 2, the emitted
spec-constant vector and pipeline statistics must be byte-identical to today's, and 1024³/2048³
timings within run-to-run spread. The unsplit path must remain the frozen control.

Only after gates 1–7 pass does benchmarking mean anything.

---

## 9. Benefits, risks, and open questions

**Expected benefits.** Up to ~30× on the low-fill shapes in §4.1; a real, if secondary,
reduction in carry-stack registers; and the first vkML optimisation that improves a shape class
rather than a constant factor. It also builds the workspace + two-kernel + barrier machinery
that the epilogue stage (M3.5) will reuse.

**Risks.**

- *Rule 1 is load-imbalancing.* Forcing `chunk = 2^q` means `S = ⌈T/2^q⌉` may leave a short
  final partition — up to 2× imbalance in the worst case. Production libraries avoid this by
  choosing `S` freely; vkML trades that for bit-identity. Mitigation: prefer the largest
  admissible `q`, which minimises the number of partitions and hence the relative cost of one
  short one.
- *`shader_core_count` is not yet queried.* Rule 2 depends on it. It is a small, isolated
  addition to `DeviceInfo`/`DeviceCapabilities`, but it is a prerequisite and should land first
  or as part of the same change.
- *Reduction becomes bandwidth-bound for large M·N.* Bounded by rule 6.
- *A second dispatch adds fixed overhead.* Irrelevant at the target shapes (0.007 ms against
  3.153 ms) but it is why rule 2 must decline for large square GEMMs.

**Open questions, explicitly not answered here.**

1. Whether shorter `STACK_LEVELS` actually reopens the 4×4 register block. §7 makes it
   measurable; it is not assumed, and Stage 8 is a warning against assuming it.
2. The best `q` for a given shape — a tuning question, and the natural first customer for the
   M3.2 autotuner.
3. Whether `LocalSplitU`/sliced-K (partitioning K *within* a workgroup, across waves, reducing
   through LDS as tinygrad's `GROUP` does) is preferable for moderate K. It avoids the
   workspace entirely and the same alignment theorem applies to it. Worth a separate iteration.

---

## 10. Recommendation

**Implement split-K, under the §2 constraint, with the §6 dispatch rules and the §8 gates.**

The design deviates from every production library in exactly one place — the reduction folds
with a carry stack instead of a flat sum — and that single deviation converts split-K from an
optimisation requiring a numerical concession into one requiring none. This is the case the
gap analysis anticipated when it asked whether vkML could *improve* on a borrowed design rather
than copy it.

Recommended sequencing:

1. Add `shader_core_count` to `DeviceInfo`/`DeviceCapabilities` (prerequisite for rule 2).
2. Implement kernel B and the workspace, with split-K forced on via `VKML_GEMM_SPLITK`, default
   **off**. Pass gates 1–6 before writing any dispatch heuristic.
3. Enable the §6 rules; pass gate 7; then benchmark.

Stopping here for review, as required. No implementation has begun.

---

# Part II — M3-03 Implementation Record

Implemented as designed. Sections 11 onward record what happened.

## 11. Implementation summary

Five pieces, all small:

| Component | Location | Notes |
|---|---|---|
| Compute-unit query | `vk_device.cpp` | `VK_AMD_shader_core_properties2` → `activeComputeUnitCount`, falling back to the older extension's topology product. Reports **36** on this GPU. 0 means unknown. |
| Capability surface | `vk_device.h`, `capabilities.h` | `shader_core_count`; consumed by nothing yet, exactly as the brief requires |
| Workspace | `vulkan_backend.cpp` `Impl::splitk_workspace()` | Cached, grown on demand, backend-internal |
| Kernel A | `vulkan_backend.cpp` dispatch | **`gemm_reg.comp` unchanged** |
| Kernel B | `shaders/gemm_split_k_reduce.comp` | New, 78 lines |

**`gemm_reg.comp` was not modified.** A partition is expressed entirely by moving the operand
base addresses along K, shortening `k`, and redirecting the output to a workspace slice:

```
sp.a = push.a + k_begin * op_a.nb[3] * esz      // strides are in ELEMENTS, addresses in bytes
sp.b = push.b + k_begin * op_b.nb[2] * esz
sp.d = ws + s * slice_elems * esz
sp.k = k_len
```

The kernel cannot tell it is computing a partition, which is the cleanest available argument
that its numerical behaviour has not drifted.

`STACK_LEVELS` is sized from the chunk rather than from total K, so a partition gets a shorter
carry stack. It is sized from `split_chunk` rather than from the last partition's possibly
smaller tile count, so all partitions share one pipeline.

### Deviations from the design

1. **§5.1 offered "grid ×S with `s = workgroup.z`" or per-partition dispatches; the second was
   chosen**, because it needs no shader change. §14 shows this was the right call
   for correctness and adequate for performance, but it is the reason the profiler mattered.
2. **§6 rule 1 was implemented as "chunk rounded *down* to a power of two from `T/requested`"**,
   so the achieved split count is generally ≥ requested. `VKML_GEMM_SPLITK_SPLITS=8` at
   K=65537 yields 9 partitions (8 × 256 tiles + 1 × 1 tile). Correctness is independent of
   which power of two is chosen, so this needs no justification beyond simplicity.
3. **No dispatch heuristic was implemented.** `AUTO` is a synonym for `OFF`, as instructed.

## 12. Compiler statistics

K=1024, 32×32 tile, 2×2 register block. Split-K at 8 partitions takes `STACK_LEVELS` 6 → 4.

| | unsplit | split-K (A) | predicted (M3-02 §7) |
|---|---|---|---|
| VGPR | 41 | **33** | 33–39 ✓ |
| SGPR | 35 | 35 | ≈35 ✓ |
| LDS | 8192 | 8192 | unchanged ✓ |
| **max_waves** | 16 | **20** | "16, unchanged" ✗ |
| scratch | 0 | 0 | 0 ✓ |
| spilled VGPR | 0 | 0 | 0 ✓ |
| instructions | 1124 | 1050 | slightly down ✓ |

Kernel B (`gemm_split_k_reduce`, 8 partitions): `VGPR=8 SGPR=16 LDS=0 waves=40 scratch=0
spilled=0 instr=75`. Predicted VGPR < 20, LDS 0, scratch 0, waves at device maximum,
instructions < 200 — **all confirmed**.

### The one failed prediction, and what it corrects

`max_waves` was predicted to stay at 16 on the reasoning that occupancy was LDS-bound. It rose
to **20**, the RDNA1 wave32 maximum, with LDS unchanged at 8192. Registers were binding all
along.

That reasoning came from M3-01 §3.2, which computed `64 KiB / 8 KiB = 8 workgroups × 8 wave32s
= 64 waves/CU = 16/SIMD` and concluded LDS was the cap. The arithmetic was right and the
conclusion was wrong: both LDS and VGPR pressure happened to yield 16, and only changing one of
them independently could tell them apart. Split-K did exactly that. `docs/M3-01-TILE-GEOMETRY.md`
now carries the correction; its conclusion is unaffected, because waves/CU is 64 either way.

**This is the first occupancy improvement in the project** — 16 → 20 waves/SIMD, +25 %, bought
purely by shortening the carry stack.

## 13. Validation results

| Gate | Result |
|---|---|
| **1 — bit identity** | **PASS.** 225 shapes (K ∈ {32,33,96,129,512,1024,4096,16384,65537} × M,N ∈ {1,7,32,127,256}), split counts 2/4/8/16. GPU hash `e38d8f6b…` identical in all five runs. |
| **2 — existing suite** | **PASS.** 487 Python + 84 C++ green under default, OFF, and forced 2/4/8/16. Layering clean at 53 files. |
| **3 — random stress** | **PASS.** 40 random shapes up to 300×5000×300, hash `d7b19082…` identical across split counts 2/3/5/8/16/64. |
| **4 — pathological** | **PASS.** Same hash. Primes (13×31×17, 257³), skinny (1×65537×1, 4096×4096×1), wide (1×4096×4096), unaligned K (33, 129, 2049, 131071), K < BK. |
| **5 — forced OFF** | **PASS.** Byte-identical output *and* identical spec constants and compiler statistics to the frozen Stage 6 baseline. |
| **6 — forced ON** | **PASS.** Verified engaged via `VKML_VULKAN_DEBUG`, including the awkward case: K=65537 → 9 partitions, the last covering 1 tile instead of 256. |
| **7 — default** | **PASS.** `AUTO` ≡ `OFF`; default hash `36e26f2b…` unchanged. |
| Determinism | **PASS.** 20 repeats, 1 distinct result. |
| Leak | **PASS.** Exactly one extra live allocation (the 2 MiB cached workspace); `in_use` delta 2,097,152 B = 8 × 256 × 256 × 4. |

A gate-1 pass is only meaningful if split-K actually ran, so that was verified separately
rather than assumed — the debug trace shows real partition counts (2, 8, 9) and the shortened
stacks (`stack_levels=4` where unsplit uses 6).

**CPU vs GPU is not bit-identical** (`31b4fc28…` vs `e38d8f6b…`). This is pre-existing, holds
equally with split-K off, and is governed by the existing tolerance policy — the CPU backend
tiles K differently. Split-K did not change it: the CPU hash is identical across all runs.

## 14. Benchmarks

Regression detection only; no tuning was performed.

**Measured with profiling OFF.** That is not a stylistic choice — see §15.

```
                 split-K off   8 splits   32 splits    best
  512x512x512       0.905 ms    0.975 ms          —    0.93x
 1024x1024x1024     2.883 ms    2.846 ms          —    1.01x
 2048x2048x2048    11.696 ms   12.119 ms          —    0.97x

  64x16384x64       3.127 ms    1.140 ms    1.070 ms   2.92x
 128x8192x128       2.381 ms    1.004 ms    0.571 ms   4.17x
 256x4096x256       1.605 ms    0.804 ms    0.900 ms   2.00x
```

Square GEMMs are neutral-to-slightly-negative, which is what §4.3 predicted and why the
dispatch rules must decline them. The target class — small M·N with large K, i.e. the
weight-gradient shape — gains **2.0× to 4.2×**.

Performance was not the acceptance criterion for this stage, and these numbers should not be
read as tuned: no threshold was chosen, and 32 partitions beats 8 on two shapes and loses on a
third.

## 15. The profiling artifact

Split-K is the first vkML operation to issue independent dispatches with no barrier between
them. Measured with the profiler on, it looked like a disaster:

```
64x16384x64        profiled GPU sum    wall clock (profiling off)
  split-K off            2.769 ms              3.127 ms
  split-K, 8 splits      6.286 ms              1.140 ms
```

6.286 ms profiled against 1.140 ms actual is impossible unless the measurement changed the
execution. `Recorder::end_timestamp()` writes at `VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT`
(`vk_command.cpp:93`), which drains the pipeline and so serialises the very overlap being
measured. The per-dispatch breakdown showed it plainly once inspected: eight partitions of
0.788 ms, summing exactly, overlapping not at all.

No earlier measurement in the project is affected — every prior benchmark was a single dispatch
or a dependent chain, where a drain costs nothing. `docs/PERFORMANCE-MODEL.md` §5f records the
rule and flags that `bench/gpu_bench.py` will under-report multi-dispatch operations until the
profiler grows a group-scoped timing mode.

## 16. Lessons

1. **The measurement infrastructure is not exempt from the measurement discipline.** This is the
   ninth time in the project that an apparent implementation failure was a measurement error,
   and the first time the *profiler built to prevent that* was the cause. A tool that changes
   what it observes is worse than no tool, because it produces confident wrong numbers.
2. **Two mechanisms that coincide cannot be separated until one is moved.** M3-01 attributed an
   occupancy cap to LDS when it was registers; both computed 16. Split-K changed registers
   alone and settled it. When two independent calculations agree, that is not confirmation —
   it is a warning that the experiment cannot distinguish them.
3. **A correctness gate must be checked for vacuity.** Gate 1 would have passed trivially if
   `split_k_chunk` had returned 0 everywhere. Verifying the path actually executed was a
   separate step and deserved to be.
4. **The bit-identity design paid for itself immediately.** With identical bytes as the
   acceptance criterion, validating 283 shapes across seven split counts needed no tolerance
   reasoning at all — a hash comparison either matches or it does not.

## 17. Recommendations for enabling runtime heuristics

Not to be done without a further stage, and in this order:

1. **Fix the profiler first.** Any heuristic tuned against the current profiler would be tuned
   against a serialised fiction. A group-scoped timestamp mode is the prerequisite for every
   number a heuristic would consume.
2. **Then implement §6 rules 2–7**, now that `shader_core_count` = 36 is available. Rule 2
   (`tiles < CU_count · 2`) would decline all three square GEMMs above and accept all three
   target shapes, which is the correct split on this evidence.
3. **Do not hardcode the partition count.** 32 beats 8 on 128×8192×128 (0.571 vs 1.004 ms) and
   loses on 256×4096×256 (0.900 vs 0.804 ms). This is a two-parameter search over `q` and shape
   — the natural first customer for the M3.2 autotuner rather than a hand-tuned constant.
4. **Consider the single-dispatch form** (`grid ×S`, §5.1's first option) before enabling by
   default. Per-partition dispatches were right for this stage, but a single dispatch would
   remove S−1 launch overheads and make the operation measurable by the existing profiler.
   That is a performance change and needs its own hypothesis and evidence.


---

# Part III — I1-R1: Production Dispatch

## 18. The heuristic that shipped

`AUTO` is now live. §6 proposed `tiles < CU x 2` as the trigger, following llama.cpp. **That
rule is wrong**, and measurement shows why: at `tiles = 64` split-K ranges from **0.59x** to
**1.99x** depending on K alone. A tiles-only threshold cannot separate those.

The shipped rule is:

```
split-K is profitable when   ktiles >= tiles
```

Read physically: split only when K-parallelism is more abundant than output-tile parallelism --
precisely the situation split-K exists for. Validated against 16 shape/K combinations spanning
`tiles` 4..2304 and `ktiles` 8..512:

```
 tiles  ktiles   gain   rule      tiles  ktiles   gain   rule
     4      16  2.05x    ON          64      64  1.21x    ON
     4      64  3.88x    ON          64     128  1.99x    ON
     4     512  3.57x    ON         144     128  1.07x   off
    16      16  1.66x    ON         256     128  1.14x   off
    16      32  2.10x    ON         576     128  1.23x   off
    16     256  2.94x    ON        1024     128  1.07x   off
    64       8  0.59x   off        2304     128  0.96x   off
    64      16  0.84x   off
    64      32  1.08x   off
```

**16/16 correct. No harmful case enabled; nothing forgone above 1.23x.**

Partition count targets `CU x 8` concurrent slots -- the figure the fill curve in
`PERFORMANCE-MODEL.md` §5g is built on -- clamped to [2, 16].

### One bug found by inspecting decisions rather than results

The first implementation *rejected* a configuration whose chunk fell below four K-tiles. That
silently forfeited two measured wins (`128x1024x128` at 2.10x, `64x512x64` at 2.05x) where the
target partition count was simply more ambitious than K could support. The floor now **clamps
the chunk up** and accepts fewer partitions. Caught by printing the decision for every shape and
comparing against the measured table -- not by any test of the results, all of which stayed
bit-identical throughout.

## 19. Measured effect of enabling AUTO

GPU time, submit window, best of 12 after 4 warm-ups:

```
             shape  tiles     before       AUTO    gain
        64x2048x64      4    0.474ms    0.117ms   4.07x
      128x1024x128     16    0.244ms    0.116ms   2.11x
      256x4096x256     64    1.063ms    0.574ms   1.85x
       256x256x256     64    0.079ms    0.053ms   1.50x
       512x512x512    256    0.348ms    0.343ms   1.01x
    1024x1024x1024   1024    1.849ms    1.859ms   0.99x
    2048x2048x2048   4096    6.991ms    7.065ms   0.99x
       4096x4096x1    128    1.337ms    1.200ms   1.11x
             TOTAL           12.386ms   11.326ms   1.09x
```

Large square GEMMs are within run-to-run spread (0.99x), which is the intended behaviour: the
heuristic declines them. `256x256x256` improves 1.50x, which the heuristic *declines* -- that
gain comes from the chunk clamp changing an adjacent decision, not from split-K.

## 20. Verification

`tests/python/test_invariants.py` gains two permanent tests:

- `test_split_k_heuristic_accepts_only_measured_wins` -- pins the **decision** for eleven
  shapes, including the two that would regress. A heuristic that silently changes which shapes
  it accepts is a regression even when every result stays bit-identical, and no correctness test
  would catch it.
- `test_split_k_partition_count_is_always_a_power_of_two_chunk` -- re-derives the chunk from the
  chosen partition count and asserts a power of two exists, so the bit-identity precondition is
  checked against what the planner actually did rather than trusted.

Gate 7 (`AUTO` byte-identical to `OFF`) continues to pass: hash `36e26f2b...` unchanged.
