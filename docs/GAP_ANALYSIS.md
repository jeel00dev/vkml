# M2.5 — Production GEMM Gap Analysis

Research phase. No vkML source was modified and no new kernel was written or benchmarked.

## Sources studied

Cloned read-only into `third_party/reference/`, full history preserved, never modified:

| Project | Revision | Date | What it contributes |
|---|---|---|---|
| llama.cpp / ggml | `55b7d6c4c` | 2026-07-26 | The only production **Vulkan** GEMM; same API, same GPU class |
| tinygrad | `97a226536` | 2026-07-26 | Search-based autotuning; no hand-written GEMM at all |
| CLBlast | `eeff2514` | 2026-04-13 | Parameterised GEMM + shipped tuning DB **containing gfx1010** |
| CUTLASS | `e64a9136` | 2026-07-23 | The reference architecture vocabulary for GEMM |
| rocBLAS | `defce200a` | 2025-08-13 | Vendor BLAS dispatch and per-architecture solution logic |
| Tensile | `e8a8999e` | 2025-06-13 | rocBLAS's actual kernel generator and tuning parameter space |

> Tensile was added beyond the requested list because rocBLAS contains almost no kernel logic
> of its own — its GEMMs are generated and tuned by Tensile, so studying rocBLAS alone would
> have answered none of the questions in the brief.

---

## 1. Executive summary

vkML's GEMM is at **1,270 GFLOP/s** at 1024³ fp32. ggml on the same GPU reaches **3,183 GFLOP/s**
at the same size (`docs/ARCHITECTURE.md` §1.2) — vkML is at **40 %** of a tuned Vulkan
implementation, and ggml itself reaches ~74 % of the 5.8 TFLOP/s hardware peak at 4096³.
The remaining gap is a factor of **2.5×**.

Five findings, in descending order of importance.

### 1.1 The pairwise carry stack — not occupancy — is what caps vkML's register block

This is the central result of the study, and it reframes Stage 8.

vkML's accumulator cost is `RM × RN × STACK_LEVELS` registers. Every production library's is
`RM × RN`. At K=1024 (`STACK_LEVELS` = 6 after bucketing, `src/backend/vulkan/vulkan_backend.cpp:707`):

| Register block | vkML accumulator VGPRs | Production equivalent | vkML overhead |
|---|---|---|---|
| 2×2 | 4 × 6 = **24** | 4 | 6× |
| 4×2 | 8 × 6 = **48** | 8 | 6× |
| 4×4 | 16 × 6 = **96** | 16 | 6× |

Stage 8 measured 4×4 spilling **24,576 bytes** to scratch. That is the first non-zero scratch in
the project, and `stack[RM*RN*STACK_LEVELS]` (`shaders/gemm_reg.comp:127`) is the array that spilled.

Now compare what production libraries actually run on this hardware class:

- **rocBLAS/Tensile, navi21 SGEMM**: `ThreadTile: [8, 8]`, `WorkGroup: [16, 8, 1]`
  (`Logic/asm_full/navi21/navi21_Cijk_Ailk_Bjlk_SB.yaml:248,268`) — **64 accumulators per thread.**
- **CLBlast, tuned for `gfx1010` (Navi 10 — vkML's exact target)**:
  RX 5700 → `MWG=128, NWG=64, MDIMC=16, NDIMC=8` = **8×8 register tile**;
  device default → `MWG=NWG=64, MDIMC=NDIMC=16, KWG=32, VWM=VWN=4` = **4×4 register tile**
  (`src/database/kernels/xgemm/xgemm_32.hpp:68-71`).
- **llama.cpp**, non-coopmat AMD: `TM=4, TN=2` per thread with `WMITER=2`
  (`ggml-vulkan.cpp:4030-4032`).

So the geometry vkML measured as catastrophic (4×4) is the geometry CLBlast's autotuner
*selected* for this GPU, and rocBLAS runs 4× larger than that on the next architecture up.
**vkML's Stage 8 did not discover that 4×4 is bad on Navi 10. It discovered that 4×4 plus a
6-deep per-accumulator carry stack is bad on Navi 10.** The two are different findings, and
only the second one is true.

This does not mean the carry stack should be deleted — it is the guarantee that lets vkML
hold a BACKWARD error bound that none of these libraries even attempt. It means the stack
must be **relocated**, and §5.2 sets out the options with evidence.

### 1.2 vkML has one kernel where every production library has a family

vkML dispatches one GEMM shader for every shape (plus two frozen experiments behind env vars).
Every project studied dispatches from a family selected at runtime:

- llama.cpp: 3 tile sizes × {aligned, unaligned} × {f32, f16, bf16, ~25 quant types} ×
  {GEMM, GEMV, mul_mat_id} — plus dedicated `p021` and `nc` kernels for specific permutations.
- CLBlast: `direct` (general, any size/offset/transpose) vs `indirect` (fast, requires padded
  input), chosen by a **tuned** threshold `XGEMM_MIN_INDIRECT_SIZE` (384 on RX 5700).
- rocBLAS: a shipped YAML table mapping problem-size tuples to solution indices.

The single highest-value missing kernel is **GEMV**. llama.cpp routes `N == 1` to an entirely
separate shader family (`ggml-vulkan.cpp:9772`), because a GEMM tile kernel at N=1 wastes
`(BN-1)/BN` = 97 % of its work. vkML currently runs its 32×32 tile kernel for N=1.

### 1.3 Nobody hand-tunes: they either search, or ship a database

Not one project studied picks tile sizes the way vkML currently does (a developer reasoning
about a performance model and committing a constant).

- **tinygrad** searches at runtime with beam search over a 9-op action space and caches the
  winner to disk (`codegen/opt/search.py:13-25,113-181`).
- **CLBlast** searches offline over 16 parameters with explicit constraints, and ships the
  results as a C++ header keyed by device name.
- **Tensile/rocBLAS** searches offline and ships per-architecture YAML including the measured
  GFLOP/s of each winning solution.

vkML's Stage 5–8 campaign is a manual, single-threaded walk through a search space these
projects explore automatically. The infrastructure vkML already built (specialisation
constants, `PipelineStats`, timestamp queries, the regression suite) is **most of an
autotuner**; what is missing is the loop.

### 1.4 rocBLAS ships no tuned kernels for this GPU at all

`library/src/blas3/Tensile/Logic/asm_full/` contains `navi21`, `navi31`, `navi32`, `navi33` —
and no `navi10`/`gfx1010`. CLBlast is the only project studied with tuned parameters for this
exact chip. This is direct evidence for vkML's founding premise, and it means the CLBlast
gfx1010 database entries are the single most valuable external data point available.

### 1.5 vkML's compiler tooling is ahead of every project studied

Searching all six repositories for use of `VK_KHR_pipeline_executable_properties` or any
equivalent register/occupancy introspection returns **nothing**. llama.cpp reasons about
registers only in source comments ("K-quants use even more registers, mitigate by setting
WMITER to 1", `ggml-vulkan.cpp:4043`). Tensile has `MaxVgprNumber` as a *search constraint*
(`Common.py:1491`) but derives it from its own assembler, not from the driver.

vkML queries actual VGPR, SGPR, scratch, LDS, wave-occupancy and instruction counts from the
driver and regression-tests them. That capability is genuinely novel here, and §1.1 is a
finding that only that infrastructure could have produced. It should be the input to the
autotuner in §1.3 — a search that prunes on measured scratch/occupancy before timing is
strictly better-informed than tinygrad's, which prunes on crude static bounds
(`BEAM_UPCAST_MAX`, `BEAM_LOCAL_MAX`).

---

## 2. Comparison table

| | vkML | llama.cpp | tinygrad | CLBlast | CUTLASS | rocBLAS/Tensile |
|---|---|---|---|---|---|---|
| Target | Vulkan | Vulkan | many | OpenCL | CUDA | HIP/asm |
| GEMM kernels | 1 (+2 frozen) | ~100s | generated | 2 + pre/post | templated | generated |
| Tile hierarchy | 2-level | **3-level** | searched | 2-level | **3-level** | 3-level |
| Runtime shape dispatch | ✗ | ✓ (3 sizes) | ✓ (per-AST) | ✓ (2 kernels) | user | ✓ (table) |
| Dedicated GEMV | ✗ | ✓ | ✓ (heuristic) | ✓ | ✓ | ✓ |
| Split-K | ✗ | ✓ | ✓ (GROUP) | ✗ | ✓ | ✓ |
| Autotuning | ✗ | ✗ | **runtime beam** | **offline DB** | profiler | **offline DB** |
| Software pipelining | built, disabled | ✗ (scalar path) | ✗ | ✗ | **✓ core** | ✓ |
| Operand pre-pass | ✗ | ✓ repack | n/a | ✓ pad/transpose | ✗ | ✗ |
| LDS bank padding | ✗ | ✓ | searched | via params | ✓ | ✓ |
| Threadblock swizzle | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ (WGM) |
| Epilogue fusion | ✗ | partial | **✓ fusion** | ✓ alpha/beta | **✓ core** | ✓ |
| Compiler statistics | **✓** | ✗ | ✗ | ✗ | ✗ | own asm |
| Deterministic reduction | **✓ pairwise** | ✗ | ✗ | ✗ | ✗ | ✗ |
| fp32 accumulate | **✓ always** | opt-in | ✓ | ✓ | configurable | ✓ |
| Error bound policy | **✓ BACKWARD** | ✗ | ✗ | ✗ | ✗ | ✗ |

---

## 3. Kernel architecture comparison

### 3.1 The three-level hierarchy vkML does not have

CUTLASS and llama.cpp both decompose **threadblock → warp → thread**; vkML decomposes
**threadblock → thread**.

llama.cpp (`mul_mm.comp:103-110`, `:168-171`):

```
BM × BN     threadblock tile     (128×128 large, 64×64 medium, 32×32 small)
WM × WN     warp tile            (per subgroup)
TM × TN     thread tile          (4×2 typical on AMD)
WMITER      warp sub-iterations  (warp covers WMITER×WNITER sub-tiles)
```

The warp level exists for a reason vkML's two-level design cannot express: it makes the
threads of one subgroup read a *contiguous* LDS region, and it decouples "how much output a
workgroup owns" from "how many registers one thread needs". **`WMITER` is the knob that
trades LDS re-reads against register pressure** — llama.cpp sets `WMITER=1` specifically for
K-quants because they "use even more registers" (`ggml-vulkan.cpp:4043`).

That is precisely the trade vkML had no way to make in Stage 8. vkML's only lever was
"grow the thread tile and grow the threadblock tile together", which forced register pressure
up with no way to buy it back.

### 3.2 Shape specialisation

llama.cpp's dispatch tree (`ggml-vulkan.cpp:9715-9777`), in evaluation order:

1. `src0` larger than `maxStorageBufferRange` → split the M dimension, loop.
2. permuted 0213, `dst->ne[1]==1` → `mul_mat_vec_p021`.
3. non-contiguous `src0`, `dst->ne[1]==1` → `mul_mat_vec_nc`.
4. `dst->ne[1]==1` (or ≤ `mul_mat_vec_max_cols` with batch 1) → **`mul_mat_vec`** (GEMV family).
5. otherwise → `mul_mm` (GEMM family), which then selects size and alignment.

Within the GEMM family (`:8524-8531`), non-coopmat:

```
(m <= 32 || n <= 32)  -> small    32×32 tile,  align 32
(m <= 64 || n <= 64)  -> medium   64×64 tile,  align 64
otherwise             -> large  128×128 tile,  align 128
```

Note these are `||` not `&&`: a 4096×32 matrix takes the small kernel. The rule is "if either
dimension is short, a big tile mostly computes padding".

CLBlast's equivalent is a single tuned scalar: below `XGEMM_MIN_INDIRECT_SIZE` use the general
`direct` kernel, above it run pad/transpose kernels and then the fast `indirect` one
(`src/routines/level3/xgemm.cpp:65-90`). The threshold is itself a database entry — 384 for
RX 5700.

### 3.3 Split-K

Three independent projects implement it, with the same structure:

- **CUTLASS** — "parallel reduction splitK", explicitly 2 kernels: partitionedK GEMM writing
  per-split partials to a workspace, then a batched reduction
  (`media/docs/cpp/efficient_gemm.md:168-200`).
- **llama.cpp** — `pipeline_matmul_split_k_reduce` (`ggml-vulkan.cpp:5226`), triggered when
  `k >= 2048` **and** the tile grid would leave the GPU less than half occupied
  (`:8452-8483`). Split count = `shader_core_count / (m_tiles * n_tiles)`, capped at 8,
  with splits rounded to multiples of 256 and reduced if that would empty the last split.
- **Tensile** — `GlobalSplitU` (across workgroups) and `LocalSplitU` (across waves within a
  workgroup, = CUTLASS's "sliced-K").

llama.cpp's trigger condition is worth stating precisely because it is occupancy-driven, not
shape-driven: it queries `activeComputeUnitCount` from `VK_AMD_shader_core_properties2`
(`:6141`) and splits only when there is idle hardware to fill.

### 3.4 What nobody does

No project studied uses **persistent kernels** for GEMM. CUTLASS has them for Hopper warp
specialisation, which depends on hardware vkML's target does not have. This is not a gap.

---

## 4. Memory hierarchy comparison

| Level | vkML today | Production practice | Evidence |
|---|---|---|---|
| Global → LDS | vec4 when provably stride-1 | same, plus width as a **tuned parameter** | CLBlast `VWM`/`VWN` ∈ {1,2,4,8}; gfx1010 default picks 4 |
| LDS layout | `As[BM*BK]`, unpadded | **padded stride** to break bank conflicts | llama.cpp `(BM+BN) × (BK + 1)`, `:3757-3761` |
| LDS ↔ registers | scalar + optional vec2 on B | warp-tile fragments, double-buffered | CUTLASS `:131-155` |
| Registers | `RM×RN` accum **× STACK_LEVELS** | `RM×RN` accum, nothing else | Tensile `ThreadTile [8,8]` |
| Subgroup ops | none in GEMM | reductions, ballot | llama.cpp `mul_mat_vec` |
| Cooperative matrix | detected, unusable | used when present | RDNA1 has none |
| L2 locality | none | **rasterisation / swizzle** | CUTLASS `:157-166`, Tensile `WorkGroupMapping` |
| DRAM channels | none | **StaggerU** offsets K-start per workgroup | Tensile `Common.py:915-935` |

Two entries deserve expansion.

**LDS bank padding.** vkML reads A as `As[(ty*RM + i)*BK + kk]` with `BK=32`. On a 32-bank LDS
with 4-byte words, addresses `32r + kk` map to bank `kk mod 32` for every `r` — so all rows
land in one bank. Within a wave32 at the current geometry only 2 distinct `ty` values occur,
so the conflict degree is 2, not 32. It is a real but small effect, and vkML's own Stage 6.5
concluded LDS issue rate was not the limiter at 2×2. Padding becomes materially more
interesting at larger tiles, where more `ty` values share a wave.

**Cooperative matrix.** `DeviceInfo::cooperative_matrix` is already queried
(`vk_device.h:38`). RDNA1 does not implement `VK_KHR_cooperative_matrix`; RDNA3 and later do.
Nothing to do now, but the three-level tile hierarchy in §3.1 is the structure that makes
adopting it later a substitution rather than a rewrite — llama.cpp uses the *same* `mul_mm.comp`
for both paths, switching on a `COOPMAT` define at the warp-tile level.

---

## 5. Autotuning comparison

### 5.1 What each project searches

| | When | What is searched | Cached | Selection metric |
|---|---|---|---|---|
| tinygrad | **runtime**, first execution | UPCAST, UNROLL, LOCAL, THREAD, GROUP, GROUPTOP, SWAP, PADTO, TC, NOLOCALS | disk, keyed on AST + device + renderer | min of 3 timings, L2 flushed |
| CLBlast | **offline**, developer machine | GEMMK, KWG, KWI, MWG, NWG, MDIMA, MDIMC, NDIMB, NDIMC, VWM, VWN, STRM, STRN, SA, SB, KREG | shipped C++ header, keyed on device name | GFLOP/s |
| Tensile | **offline**, vendor | ThreadTile, WorkGroup, DepthU, GlobalSplitU, LocalSplitU, PrefetchGlobalRead/LocalRead, VectorWidth, StaggerU, WorkGroupMapping, MaxVgprNumber | shipped YAML, keyed on arch + problem size | GFLOP/s, recorded in the table |
| llama.cpp | never | — | — | developer judgement |
| CUTLASS | offline via profiler | template parameters | user's choice | GFLOP/s |

### 5.2 How they keep the space tractable

This matters more than the parameter list, because vkML's space is small enough that the
pruning technique transfers directly.

**CLBlast — algebraic constraints** (`src/tuning/kernels/xgemm.hpp:160-208`). Rather than
generating and rejecting, it declares relations that must hold:

```
KWG  multiple of KWI
MWG  multiple of MDIMC × VWM        NWG multiple of NDIMC × VWN
MWG  multiple of MDIMA × VWM        NWG multiple of NDIMB × VWN
KWG  multiple of (MDIMC×NDIMC)/MDIMA
```

plus a `LocalMemSizeInfo` functor that computes LDS bytes for a candidate so configurations
that cannot fit are never compiled (`:209-215`). vkML already asserts LDS against
`maxComputeSharedMemorySize`; the difference is that CLBlast uses it as a *search filter*.

**tinygrad — static bounds and compute filtering.** `BEAM_UPCAST_MAX=256` and
`BEAM_LOCAL_MAX=1024` bound the product of upcast and local axes (`search.py:91,105`), and any
candidate performing 1000× the minimum compute op count is dropped (`:151-155`). Beam width
is the user-set `BEAM` value; the loop stops when improvement falls below
`BEAM_MIN_PROGRESS` (0.01 µs).

**Tensile — an explicit register ceiling.** `MaxVgprNumber` ∈ [0,256] is a first-class search
parameter (`Common.py:1491`). This is the closest any project comes to vkML's compiler
statistics, and it is still only a constraint on Tensile's own generator, not a measurement.

### 5.3 The opportunity specific to vkML

vkML can prune on **measured** resources rather than estimates, because `PipelineStats` returns
real VGPR/scratch/LDS/occupancy from the driver *after compilation but before dispatch*. A
search that compiles a candidate, reads `spilled_vgprs`/`scratch_bytes`/`max_waves`, and
discards it without ever timing it, is cheaper and better-founded than any of the four
strategies above. Stage 8 is the existence proof: the regression was fully visible in the
statistics before a single benchmark ran.

---

## 6. Numerical comparison

This is the section where vkML is not behind — it is doing something none of them do.

| | Accumulation over K | Accumulator precision | Reduction determinism | Error policy |
|---|---|---|---|---|
| **vkML** | **pairwise, block 32** | **fp32 always** | **bit-reproducible** | **BACKWARD bound, γ·Σ\|terms\|** |
| llama.cpp | sequential | **fp16 by default**, clamped to `ACC_TYPE_MAX`; fp32 only on `GGML_PREC_F32` | no (split-K order varies) | none |
| CLBlast | sequential | matches operand type | no | none |
| Tensile | sequential | fp32 for SGEMM | no (`GlobalSplitU` reorders) | none |
| CUTLASS | sequential | configurable | no | none |
| tinygrad | sequential | fp32 | no (GROUP reorders) | none |

llama.cpp's `mul_mm.comp:354-364` clamps accumulators to `ACC_TYPE_MAX` after the mainloop —
an overflow guard that is only necessary *because* it accumulates in fp16. It is a speed
trade with a correctness cost, made deliberately, and gated behind a precision flag when the
caller cares.

**Consequences for vkML's roadmap, stated precisely:**

- Every optimisation in §7 that does **not** change the order of floating-point additions is
  numerically free: GEMV, shape dispatch, LDS padding, alignment variants, swizzle, StaggerU,
  vectorisation width, workgroup geometry. These are the majority.
- **Split-K changes the fold order.** It is not disqualified — the resulting order is still a
  deterministic tree, and moving the cross-tile fold out of registers is the most promising
  route to §1.1 — but it requires re-deriving the BACKWARD bound and re-pinning goldens
  before it can ship. It must not be adopted on the assumption that "a tree is a tree".
- fp16 accumulation is **incompatible** with vkML's stated policy and should not be adopted at
  any tile size. fp16 *storage* with fp32 accumulate is a separate, compatible question.

---

## 7. Optimisation matrix

Categorised as the brief requires. "Converged" marks techniques ≥3 independent projects use —
the brief's criterion for a strong M3 candidate.

### Already implemented in vkML

| Technique | Notes |
|---|---|
| Shared-memory tiling | Stage 4 |
| Register blocking | Stage 5, 2×2 |
| K-tile depth tuning | Stage 5.75, BK=32 |
| Vectorised global loads | Stage 6, vec4 with provable stride-1 |
| Specialisation constants | one module, many pipelines — matches llama.cpp exactly |
| Pipeline cache | `VkPipelineCache`, not yet serialised to disk |
| Compiler statistics | **exceeds all five projects** |
| Timestamp profiling | exceeds all five |
| Deterministic pairwise reduction | unique |
| Double buffering | built (`gemm_db.comp`), measured, currently disabled |

### Applicable now — no architectural change

| Technique | Converged | Evidence |
|---|---|---|
| **Dedicated GEMV kernel (N=1)** | ✓ 5/5 | llama.cpp `mul_mat_vec` family; tinygrad `MATVEC` heuristic; CLBlast `xgemv_fast`; rocBLAS; CUTLASS |
| **Runtime shape dispatch (s/m/l tiles)** | ✓ 4/5 | llama.cpp `:8524`; CLBlast direct/indirect; rocBLAS table; tinygrad per-AST |
| **Aligned / unaligned kernel variants** | ✓ 3/5 | llama.cpp `ALIGNED` spec constant + `a_l/a_m/a_s` pipelines; CLBlast pad kernels; tinygrad `PADTO` |
| **LDS bank-conflict padding** | ✓ 3/5 | llama.cpp `bank_conflict_offset`; CUTLASS; Tensile |
| **Disk-serialised pipeline cache** | — | vkML already has the in-memory half |
| **Tuned vector width per operand** | ✓ 3/5 | CLBlast `VWM`/`VWN`; Tensile `VectorWidth`; tinygrad `UPCAST` |
| Threadblock swizzle for L2 | ✓ 2/5 | CUTLASS `:157`; Tensile `WorkGroupMapping` |
| StaggerU (DRAM channel spreading) | 1/5 | Tensile `:915-935`; AMD-specific but vkML's target is AMD |

### Applicable now — but numerically load-bearing

| Technique | Converged | Why it is not "free" |
|---|---|---|
| **Split-K / GlobalSplitU** | ✓ 4/5 | Reorders the K fold. Requires re-deriving the BACKWARD bound and re-pinning goldens. **Also the most direct route to §1.1**, since it shortens the in-register carry stack by `log2(split_k)`. |
| Sliced-K / LocalSplitU | ✓ 2/5 | Same, across waves rather than workgroups |

### Requires architectural redesign

| Technique | Converged | What must change first |
|---|---|---|
| **Three-level tile hierarchy (warp tile + WMITER)** | ✓ 3/5 | The shader's thread→output mapping. This is the enabler for everything below it. |
| **Software pipelining at register scope** | ✓ 2/5 | CUTLASS double-buffers *register fragments* as well as LDS tiles; vkML's Stage 7 only did LDS |
| Relocating the carry stack out of registers | — | No precedent — no other project has one. Options in §8.2. |
| Epilogue as a separate shared-memory phase | ✓ 3/5 | Enables fused bias/activation and coalesced stores |
| Operand pre-pass (repack / pad / transpose) | ✓ 2/5 | llama.cpp and CLBlast both materialise operands; vkML handles strides in-kernel |
| **Autotuning loop** | ✓ 3/5 | Needs a persistent tuned-parameter store and a search driver |

### Hardware-specific — not applicable to RDNA1

| Technique | Why not |
|---|---|
| Cooperative matrix / tensor cores | RDNA1 has no matrix cores; already correctly detected and unused |
| Hopper warp specialisation, TMA, cluster launch | NVIDIA-only |
| `coopmat2` tile sizes (llama.cpp's 256×128) | Requires NV coopmat2 |
| Integer dot-product MMQ path | Needs `VK_KHR_shader_integer_dot_product`; also a quantisation feature, out of scope |
| Asahi BK-loop unroll workaround (`ggml-vulkan.cpp:2727`) | Driver-specific bug workaround |

### Incompatible with vkML's design goals

| Technique | Conflict |
|---|---|
| **fp16 accumulation** | Directly contradicts the BACKWARD tolerance policy. llama.cpp does this by default and clamps to avoid overflow. |
| Non-deterministic reduction order | vkML guarantees bit-reproducibility |
| Quantised weight formats (Q4_K, IQ2_XXS, …) | Inference-only; vkML is a training framework |
| Timing-only autotuning without resource pruning | vkML's own history — eight separate cases — is that wall-clock alone misleads. Any vkML autotuner must gate on `PipelineStats` first. |

---

## 8. Where the evidence converges

### 8.1 Strong candidates (≥3 projects independently)

1. **Dedicated GEMV** — 5/5. Largest single win for the smallest risk, and directly relevant:
   `ARCHITECTURE.md` §1.2 already records that small-batch training on this GPU is
   bandwidth-bound. A chess-eval net at batch 1 is a GEMV workload.
2. **Runtime shape dispatch** — 4/5.
3. **Aligned/unaligned variants** — 3/5.
4. **Autotuning with a persistent database** — 3/5.
5. **Three-level tile hierarchy** — 3/5.
6. **Split-K** — 4/5, gated on numerical re-derivation.
7. **LDS bank padding** — 3/5.

### 8.2 The carry-stack question

No project studied has this problem, so there is no borrowed answer. The study does, however,
bound the options:

- **Shorten the stack.** Depth is `ceil(log2(K/BK)) + 1`. Split-K divides `K` per workgroup, so
  `split_k=8` at K=1024 takes the bucket from 6 to 4 — a 33 % reduction in stack registers, and
  the cross-split fold lands in a reduction kernel where it costs no registers at all. This is
  the only option with production precedent (CUTLASS's two-kernel structure exists for a
  different reason but has exactly the right shape).
- **Move the stack to LDS.** At 4×4 the spill was 24 KiB — which fits the 64 KiB LDS budget
  that is currently 87 % idle (8 KiB used at 2×2). Folds happen once per k-tile, i.e. once per
  32 MACs, so the traffic is low. **This is a hypothesis, not a finding** — it must be measured
  with the existing `PipelineStats` and timestamp infrastructure before being believed.
- **Decouple stack depth from the register block** via the warp tile of §3.1, so `WMITER`
  iterations reuse one stack.

These are not mutually exclusive, and the first is a prerequisite for evaluating the others
cheaply.

---

## 9. Recommended roadmap

Detailed ranking, complexity and ordering are in `docs/M3_ROADMAP.md`. In outline:

- **M3.1 — shape dispatch and GEMV.** Highest benefit, no numerical impact, no redesign.
  Establishes the kernel-family structure everything later plugs into.
- **M3.2 — autotuning infrastructure.** Turn the manual Stage 5–8 walk into a search that
  prunes on `PipelineStats` before timing. vkML has more of this built than any project studied.
- **M3.3 — three-level tile hierarchy.** The redesign that unblocks large register tiles,
  `WMITER`, register-scope pipelining, and later cooperative matrix.
- **M3.4 — split-K**, with the numerical re-derivation done first, not after.
- **M3.5 — epilogue and fusion**, which is where a training framework's real wins live
  (bias + activation + backward accumulation) and which no amount of GEMM tuning reaches.

One caution carried forward from M2: every stage above changes what the *right* tile size is.
Sequencing M3.2 before M3.3 means the redesign is evaluated by search rather than by a
developer's model — which, on the evidence of Stage 6.5 and Stage 8, is the more reliable of
the two.

---

# Part II — Cross-Project Invariants (M3-R1)

Part I catalogued what production libraries *do*. This part asks what they **agree on**, which
is more useful: a design choice made independently by four projects with different languages,
hardware targets and authors is evidence about the problem, not about the projects.

Each invariant is graded by evidence strength, and each is checked against vkML.

---

## I1. Workgroup size is bounded; the tile grows through per-thread work

**Proven** (M3-01, measured + four sources).

```
llama.cpp   128x128 tile -> 128 invocations   |  64x64 tile -> 128 invocations
CLBlast      64x64  tile -> 256 (gfx1010 tuned MDIMC=NDIMC=16)
rocBLAS     128x64  macro tile -> 128 (navi21 WorkGroup [16,8,1])
```

No project studied exceeds 256 invocations for any tile size, and llama.cpp's workgroup is
*constant* across a 4x change in tile area. vkML tested the opposite -- growing the workgroup to
1024 -- and measured a monotonic regression (1.00x -> 0.84x -> 0.77x), because the number of
independent barrier domains per CU halves at each step.

**Holds for vkML.** Now encoded in `PERFORMANCE-MODEL.md` 5e.

## I2. Split-K reduction is deterministic, and atomics are explicitly avoided

**Proven** (M3-02, five sources).

llama.cpp, CUTLASS (both `kSerial` and `kParallel`) and Tensile `MultipleBuffer` all reduce
through a workspace or an ordered in-place accumulation. Tensile's atomic path exists and its
own source annotates it `NOTE: This is not recommended` (`Common.py:605`).

**Holds for vkML, and vkML goes further.** All of them fold partials *sequentially*, which is
deterministic but reorders relative to the unsplit kernel. vkML folds with the carry stack
instead, making split-K bit-identical -- the one place vkML deliberately diverges from unanimous
production practice, and the divergence is what preserves its guarantee.

## I3. GEMV is a separate kernel family, never GEMM with N=1

**Proven** (5/5 sources). Still **not implemented** in vkML; the highest-value remaining gap.

## I4. Tile selection is never a hardcoded constant

**Proven** (4/5). tinygrad searches at runtime; CLBlast and Tensile ship per-device databases;
llama.cpp selects from three sizes by shape at runtime. Only vkML uses a compile-time constant.

## I5. K-tile depth lives in 8-32, and the optimum is shape-dependent within that band

**Newly identified (M3-R1).** Evidence:

```
llama.cpp     BK = 16 (fp path), 32 (quant), 64 (coopmat2 only)
CLBlast       KWG = 32 (gfx1010 default, RX 5700), 16 (RX 5700 XT)
rocBLAS       DepthU across 251 tuned navi21 SGEMM solutions:
                  DepthU=16 -> 100 solutions
                  DepthU=8  ->  79
                  DepthU=32 ->  72
vkML          BK = 32, fixed
```

The rocBLAS distribution is the interesting part. Given a free choice and an offline tuner,
**no single depth wins** -- the three legal values split roughly evenly across tuned shapes.
That is direct evidence that K-depth belongs in an autotuner rather than in a constant.

**Partially holds for vkML, with a real constraint.** vkML fixed `BK=32` in Stage 5.75 for two
reasons: it halves the k-tile count (shortening the carry stack), and it makes each block an
exact 32-element sequential sum matching `kPairwiseBlock`. The second reason makes `BK` unlike
every other tunable parameter: **changing it changes the reduction tree**, so it cannot be
searched freely the way `BM`, `BN`, `RM` and `RN` can -- a new `BK` means re-pinned goldens.

> **Consequence for M3.2 (autotuning).** The search space splits in two. `BM`, `BN`, `RM`, `RN`,
> workgroup size, vector widths and split count are **fold-neutral** and can be searched with
> bit-identity as the acceptance test. `BK` is **fold-changing** and must be treated as a
> versioned decision, not a search dimension. This distinction did not exist before M3-R1 and
> materially changes how the autotuner should be built.

## I6. Occupancy decisions are made from a device query, never assumed

**Proven** (llama.cpp `shader_core_count`, Tensile `MaxVgprNumber`, CLBlast `LocalMemSizeInfo`).

**Now holds for vkML** -- `shader_core_count` = 36 was added in M3-03, though nothing consumes
it yet.

## I7. Nobody guarantees reduction order

**Proven** (6/6). No project studied offers bit-reproducibility or an error-bound policy.
llama.cpp accumulates in **fp16 by default** and clamps to `ACC_TYPE_MAX` to avoid overflow.

**Deliberately violated by vkML.** This is the project's defining constraint and the source of
its main structural limit (Part I 1.1): the carry stack costs `RM*RN*STACK_LEVELS` registers
where production libraries pay `RM*RN`.

---

## Where the projects *disagree*

Disagreement is as informative as consensus -- it marks a genuinely open design question rather
than a solved one.

| Question | Positions | Reading |
|---|---|---|
| Tuning: runtime or offline? | tinygrad runtime beam search; CLBlast + Tensile shipped offline DB; llama.cpp neither | Unsettled. Depends on whether shapes are known ahead of time -- for a *training* framework they largely are, favouring a cache keyed on shape. |
| Operand handling | llama.cpp and CLBlast **materialise** non-contiguous operands in a pre-pass; CUTLASS and vkML handle strides in-kernel | Unsettled; neither side published a comparison. |
| Split-K accumulation | separate reduction kernel (llama.cpp, Tensile, CUTLASS `kParallel`) vs ordered in-place via semaphore (CUTLASS `kSerial`) | Trade-off: workspace vs serialisation. vkML chose the workspace. |
| Split-K partition cap | llama.cpp caps at 8; Tensile permits 4096 but ships 1 | Suggests large split counts are rarely worth it -- consistent with vkML measuring 32 partitions beating 8 on one shape and losing on another. |

---

## Evidence grading used above

- **Proven** -- verified in the reference source *and* independently corroborated by vkML
  measurement or by >=3 projects agreeing.
- **Newly identified** -- read directly from reference source in this stage; not yet
  corroborated by a vkML experiment.
- Anything weaker is stated as a hypothesis or open question, never as an invariant.
