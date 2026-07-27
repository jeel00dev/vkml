# M3-01 — Shape-Driven Threadblock Tile Geometry

One optimisation. Derived from `docs/GAP_ANALYSIS.md` §1.1 and §3.2.

Sections 1–5 were written and fixed **before** any code was changed. Sections 6 onward record
what happened.

---

## 1. Problem

vkML dispatches a single threadblock tile — 32×32, `BK=32`, 2×2 register block, 256 invocations —
for every GEMM shape. Arithmetic intensity is therefore a constant:

```
AI(32×32) = 2·BM·BN·BK / ((BM·BK + BK·BN)·4)
          = 2·32·32·32 / ((32·32 + 32·32)·4)
          = 8.0 FLOP/byte
```

Every production library studied selects a tile at runtime instead (`GAP_ANALYSIS.md` §3.2):
llama.cpp picks from 32×32 / 64×64 / 128×128 on `m` and `n`; CLBlast's tuned `gfx1010` entry
uses `MWG=NWG=64`; rocBLAS's navi21 SGEMM uses a 128×64 macro tile.

## 2. Why the obvious route is closed

The natural way to enlarge a tile is to enlarge the register block, and Stage 8 measured that
as a 21 % regression with 24 KiB of scratch. `GAP_ANALYSIS.md` §1.1 established the cause:
vkML's accumulator cost is `RM·RN·STACK_LEVELS`, not `RM·RN`, because the pairwise carry stack
is per-accumulator. At K=1024 (`STACK_LEVELS=6`) a 4×4 block needs 96 registers of carry stack
alone.

**So this iteration enlarges the tile along the axis that does not touch the carry stack.**
`BM` and `BN` set how much output a *workgroup* owns; `RM` and `RN` set how much a *thread*
owns. Holding `RM=RN=2` and raising `BM`/`BN` raises the workgroup's invocation count instead
of its per-thread register count. This is a point in the design space Stage 8 did not visit —
Stage 8 deliberately held the workgroup at 256 invocations so that occupancy changes could be
attributed to register pressure. That control is exactly what has to be released now.

Device limits permit it: **1024 max invocations, 64 KiB shared memory** (measured, this GPU).
A 64×64 tile at 2×2 needs `(64/2)·(64/2) = 1024` invocations and `(64·32 + 32·64)·4 = 16` KiB.
Both are exactly at or under the limit.

## 3. Hypothesis

> Enlarging the threadblock tile while holding the register block at 2×2 raises arithmetic
> intensity in proportion to the tile area, at no cost in per-thread registers, because
> vkML's register cost scales with `RM·RN·STACK_LEVELS` and is independent of `BM`/`BN`.
> Global memory traffic is a first-order limiter at 1024³, so the larger tile will be faster.

Predicted intensity:

| Tile | Invocations | LDS | AI (FLOP/byte) | vs 32×32 |
|---|---|---|---|---|
| 32×32 | 256 | 8 KiB | 8.0 | 1.00× |
| 64×32 | 512 | 12 KiB | 10.67 | 1.33× |
| 64×64 | 1024 | 16 KiB | **16.0** | **2.00×** |

### 3.1 Quantified prediction for 1024³

At 32×32 the kernel issues `(1024/32)² = 1024` workgroups, each reading
`(32·32 + 32·32) = 2048` floats per k-tile across `1024/32 = 32` k-tiles:

```
32×32:  1024 wg × 32 ktiles × 2048 floats × 4 B = 256 MiB of global reads
64×64:   256 wg × 32 ktiles × 4096 floats × 4 B = 128 MiB
```

At the measured 288 GB/s peak, 256 MiB is 0.89 ms against a measured runtime of 1.748 ms —
**global traffic alone accounts for 51 % of the frozen baseline's time.** If the kernel were
purely bandwidth-bound, halving traffic predicts 1.748 → 1.30 ms (1.34×). L2 already absorbs
some of that traffic, so the realistic expectation is **smaller than 1.34× and larger than
1.0×**.

### 3.2 Expected compiler statistics

This is where the hypothesis is most sharply falsifiable. Per-thread work is *identical*
between the geometries — each thread still owns 2×2 outputs, still walks the same number of
k-tiles, and still performs `RM + RN = 4` LDS reads per 4 FMAs. Therefore:

| Statistic | Prediction |
|---|---|
| VGPRs | **unchanged** (41) — the carry stack is `RM·RN·STACK_LEVELS`, untouched |
| Scratch | **0** — the Stage 8 failure mode must not reappear |
| Spilled VGPRs | 0 |
| LDS | 8192 → 16384 |
| Instruction count | ≈ unchanged (~1124) — same per-thread loop structure |
| `max_waves` | **at risk** — see below |

**The occupancy risk is the falsifier.** The baseline reports `max_waves = 16` at VGPR=41.
41 registers cannot be what caps 16, so something else does — and 64 KiB LDS ÷ 8 KiB = 8
workgroups × 8 wave32s = 64 waves/CU = 16/SIMD says the cap is **LDS, not registers**.

> **Corrected by M3-03.** This attribution was wrong. Split-K later reduced the carry stack
> from 6 levels to 4, dropping VGPRs 41 → 33 with **LDS unchanged at 8192**, and `max_waves`
> rose 16 → 20. Registers were the binding constraint all along; the LDS arithmetic above
> coincidentally also produced 16. The M3-01 conclusion is unaffected — waves/CU is 64 either
> way, so independent workgroups per CU are still 8 / 4 / 2 — but the *reason* 16 was the cap
> is register pressure, not shared memory.
Doubling LDS per workgroup to 16 KiB while quadrupling the workgroup's wave count leaves the
waves-per-CU arithmetic unchanged in principle, but only measurement will confirm the driver
agrees. If `max_waves` falls, the AI gain will be spent on latency-hiding loss, exactly as it
was in Stage 8 — and the hypothesis is refuted.

### 3.3 Falsification criteria

The hypothesis is **rejected** if any of:

1. VGPR count rises materially, or scratch becomes non-zero.
2. `max_waves` falls below 16.
3. 1024³ does not improve.

The hypothesis is **confirmed** only if per-thread resources are flat *and* 1024³ improves.
A speedup with degraded resources would mean the mechanism is not the one claimed, and would
have to be explained before being accepted.

### 3.4 Risks

- **Small shapes lose parallelism.** A 64×64 tile at M=N=128 launches 4 workgroups on a 36-CU
  GPU. Guarded by requiring `m >= 128` / `n >= 128` before widening; a compute-unit-aware
  rule needs `VK_AMD_shader_core_properties2`, which vkML does not yet query, and belongs to
  the split-K iteration (`M3_ROADMAP.md` M3.4).
- **1024 invocations is the hardware maximum.** No headroom for a larger tile at 2×2 without
  the warp-tile hierarchy of M3.3.
- **Barrier cost scales with workgroup size.** Two barriers per k-tile now synchronise 1024
  invocations rather than 256.

## 4. Numerical compatibility

**No change to floating-point ordering. Bit-identical results. Goldens stay pinned.**

The argument is structural, not empirical. For a fixed output element `D[i][j]`, the fold order
is determined entirely by:

1. `BK` — the sequential block length within a k-tile (unchanged at 32, still exactly
   `kPairwiseBlock` from `src/backend/cpu/reduce.h`);
2. the number of k-tiles, `ceil(K/BK)` (unchanged — it depends on `K` and `BK` only);
3. the carry-stack fold across those tiles (unchanged — same algorithm, same `STACK_LEVELS`
   bucket).

`BM` and `BN` determine *which invocation* computes `D[i][j]`, not *in what order* its products
are summed. No product moves between invocations, because each output element is owned entirely
by one invocation in both geometries.

This is the same reasoning that let Stage 8 validate 4×2 and 4×4 against the oracle without
re-pinning goldens, and it is why this optimisation was chosen first: the entire existing
validation suite applies unchanged, with no tolerance discussion.

- Floating-point order: **unchanged**
- Pairwise reduction: **unchanged**
- Determinism: **unchanged**
- BACKWARD policy: **unaffected**
- Tolerances: **unchanged** — no weakening, none needed

## 5. Design

### 5.1 Selection rule

Chosen over two alternatives:

- *Rejected — llama.cpp's `(m <= 32 || n <= 32)` ladder.* Its `||` couples the two dimensions,
  so a 4096×64 matrix takes a 32×32 tile in both dimensions. That is right for llama.cpp,
  whose tiles are square; vkML can pick `BM` and `BN` independently and does not need to
  discard the large-`M` opportunity because `N` is small.
- *Rejected — a tuned threshold table.* Premature: M3.2 is the autotuner, and hardcoding a
  table now would be a constant the tuner immediately replaces.

Adopted, one independent decision per dimension:

```
BM = (m >= 128) ? 64 : 32
BN = (n >= 128) ? 64 : 32
workgroup = (BM/RM) · (BN/RN)
```

Yielding 32×32 (256), 64×32 (512), 32×64 (512), 64×64 (1024). The frozen baseline is recovered
exactly whenever `m < 128 && n < 128`, so small shapes are untouched by construction.

### 5.2 Keeping the control arm frozen

Stage 6.5 produced a spurious 1.4× because the A/B control arm had itself been modified. The
same mistake is avoided here structurally: `VKML_GEMM_TILE=s|m|l` forces a geometry, and `s` is
byte-identical to the frozen baseline because it is the same spec-constant vector, compiled
from the same unmodified `gemm_reg.comp`.

**No shader source changes.** `gemm_reg.comp` already parameterises `BM`/`BN` as specialisation
constants and strides its cooperative load loops by `gl_WorkGroupSize.x`, so it generalises to
the new geometries without edits. The change is confined to host-side geometry selection.

### 5.3 Layering

`src/backend/vulkan/vulkan_backend.cpp` only. No public API change, no new dependency, no
change to `KernelConfig`. Existing `VKML_GEMM_BLOCK` (the Stage 8 register-block experiment)
is left intact and takes precedence when set, so both experiments remain independently
reproducible.

---

## 6. Compiler analysis (Phase 7 — collected before benchmarking)

```
tile     spec constants                          VGPR SGPR   LDS  waves scratch instr
32×32 s  256_32_32_32_2_2_6_1_1                    41   35  8192     16       0  1124
64×32 m  512_64_32_32_2_2_6_1_1                    41   33 12288     16       0  1130
64×64 l  1024_64_64_32_2_2_6_1_1                   41   35 16384     16       0  1126
```

**Every prediction in §3.2 confirmed.** VGPRs flat at 41 — the carry stack is
`RM·RN·STACK_LEVELS` and never saw `BM`/`BN`, exactly as argued. Scratch stayed 0, so the
Stage 8 failure mode did not reappear. LDS moved 8192 → 16384 as computed. Instruction count
moved by 0.5 %, confirming per-thread work is unchanged. `max_waves` held at 16 — the one
statistic §3.2 flagged as at risk, and it did not degrade.

On the statistics alone this optimisation should have worked.

## 7. Validation (Phase 8)

487 Python + 84 C++ tests pass on the auto path and on each forced geometry (`s`, `m`, `l`).
Layering clean at 53 files. The Stage 8 register-block variants (`VKML_GEMM_BLOCK=4x2|4x4`)
still validate, so neither experiment disturbed the other.

The §4 claim was bit-identity, which is stronger than "within tolerance", so it was checked
directly: SHA-256 over the concatenated outputs of five shapes including
non-power-of-2 (257×129×131) and 1024³.

```
tile=s     36e26f2b50fe9b588bbf42eb645fc121e07d58eec7c4dd23330e81f63255a2db
tile=m     36e26f2b50fe9b588bbf42eb645fc121e07d58eec7c4dd23330e81f63255a2db
tile=l     36e26f2b50fe9b588bbf42eb645fc121e07d58eec7c4dd23330e81f63255a2db
tile=auto  36e26f2b50fe9b588bbf42eb645fc121e07d58eec7c4dd23330e81f63255a2db
```

Identical. No tolerance was touched, and none needed to be. The structural argument in §4 holds.

## 8. Benchmark (Phase 9)

GPU timestamps, 15 reps, minimum reported. Baseline is the frozen Stage 6 geometry, reached by
the unmodified spec-constant vector.

```
     M      K      N   32×32 (base)     64×32          64×64
   512    512    512      0.342ms   0.367ms 0.93×   0.475ms 0.72×
  1024   1024   1024      1.807ms   2.141ms 0.84×   2.333ms 0.77×
  2048   2048   2048      7.093ms   8.426ms 0.84×  11.025ms 0.64×
  4096   1024     64      0.608ms   0.682ms 0.89×          — (BN stays 32)
    64   1024   4096      0.565ms   0.627ms 0.90×          — (BM stays 32)
  2048   2048    128      1.078ms   1.253ms 0.86×   2.542ms 0.42×
```

**Every shape regressed, monotonically in workgroup size.** Falsification criterion 3 was met;
criteria 1 and 2 were not. The hypothesis is rejected.

## 9. Performance analysis (Phase 10)

### Was the hypothesis wrong, the implementation wrong, or the model incomplete?

**Not the implementation.** The dispatch grid is derived from the same `kBM`/`kBN` that
configure the pipeline (`vulkan_backend.cpp:837-842`), so no redundant workgroups are
launched; and bit-identical output across geometries rules out a mis-mapped tile.

**Not the hypothesis's premise.** Arithmetic intensity really did double, and per-thread
registers really were unaffected — the compiler statistics prove both.

**The model was incomplete.** It used `max_waves` as the occupancy metric. That metric was
flat at 16 across all three geometries and still missed a 36 % regression, because the waves
of a single workgroup **all rendezvous at the same barrier** and are therefore not independent
work. The quantity that matters is concurrent workgroups per CU:

```
tile     waves/CU  waves/wg  independent workgroups/CU   1024³
32×32          64         8                          8   1.000×
64×32          64        16                          4   0.844×
64×64          64        32                          2   0.774×
```

Halves at every step; performance falls with it. The intensity gain was real and was simply
smaller than the loss of latency-hiding independence.

### Direct confirmation of the mechanism

If synchronisation is the cost, removing a barrier must help disproportionately at the large
tile. The Stage 7 double-buffered kernel, built and disabled, removes one of two barriers per
k-tile and was run as a diagnostic:

```
              no DB      +DB     recovered
32×32, 2048³  7.093ms  6.737ms      1.05×
64×64, 2048³ 11.025ms  7.782ms      1.42×
```

1.42× against 1.05×. The mechanism is confirmed, not inferred. (Even so, `64×64 + DB` at
7.782 ms still loses to plain `32×32` at 7.093 ms, so double buffering does not rescue the
geometry. The apparent 1.05 × gain for `32×32 + DB` is inside its own run-to-run spread —
sd 1.33 ms — and is **not** claimed as a result; Stage 7's small-tile conclusion stands.)

`docs/PERFORMANCE-MODEL.md` §5e records the new term.

## 10. Comparison against production libraries (Phase 11)

The libraries had already answered this, and the answer was legible before the experiment ran —
it was simply not read carefully enough.

```
llama.cpp, ggml-vulkan.cpp:4030-4032    warptile[0] is the WORKGROUP SIZE
  l_warptile = { 128, 128, 128, ... }   128×128 tile -> 128 invocations
  m_warptile = { 128,  64,  64, ... }    64×64  tile -> 128 invocations
  s_warptile = {  32,  32,  32, ... }    32×32  tile ->  32 invocations
```

**llama.cpp's workgroup size does not grow with its tile.** It is 128 for both the 64×64 and
the 128×128 tile. CLBlast's tuned `gfx1010` entry uses `MDIMC=NDIMC=16` = 256 threads for a
64×64 tile; rocBLAS's navi21 SGEMM uses `WorkGroup [16,8,1]` = 128 threads for a 128×64 macro
tile. Three independent projects, none exceeding 256 invocations, all growing the tile through
per-thread work.

The reason vkML cannot follow them is now sharp, and it closes a loop with `GAP_ANALYSIS.md`
§1.1. llama.cpp's large tile gives each thread 128 accumulators, affordable only because it
packs them two-per-register in fp16 (`ACC_TYPEV2`, `mul_mm.comp:270`). vkML accumulates in
fp32 by policy and multiplies every accumulator by a 6-deep carry stack. So:

- The tile cannot grow through **registers** — the carry stack (Stage 8, 24 KiB scratch).
- The tile cannot grow through **workgroup size** — barrier independence (this iteration).

Those are the only two axes available without a warp-tile hierarchy. **vkML's threadblock tile
is structurally pinned at 32×32 until the carry stack shortens.** That is a stronger and more
useful statement than "64×64 was slower", and it was not available before this experiment.

### Roadmap consequence

`M3_ROADMAP.md` ordered split-K fourth. This result promotes it: shortening the carry stack is
no longer one of several routes to a bigger tile, it is the **only** one. Splitting K by 8 at
K=1024 moves `STACK_LEVELS` from 6 to 4, which is the first lever that frees registers rather
than spending them. Its numerical gate (re-deriving the BACKWARD bound) is unchanged and still
comes first.

The M3.3 warp-tile hierarchy also gains a second justification: `WMITER` decouples tile area
from workgroup size, which is precisely the axis this iteration proved is needed.

## 11. Outcome

**Rejected, and retained as a reproducible experiment.** `VKML_GEMM_TILE=m|l` selects the
enlarged geometries; the default is the frozen Stage 6 geometry, restored byte-identically
(same spec-constant vector `256_32_32_32_2_2_6_1_1`, same statistics, same output hash).
This follows the precedent of `gemm_db` (Stage 7) and `VKML_GEMM_BLOCK` (Stage 8): measured,
understood, disabled, kept.

### Hypotheses confirmed

- Threadblock tile area does not affect per-thread register pressure (VGPR flat at 41 across
  a 4× tile-area change).
- `BM`/`BN` do not affect floating-point ordering (bit-identical, proven by hash).
- Arithmetic intensity scales with tile area as computed (8.0 → 16.0).

### Hypotheses rejected

- That raising arithmetic intensity via the threadblock tile improves throughput on this GPU.
  It does not, at any measured shape, by up to 58 %.

### Eliminated from the search space

Enlarging the threadblock tile via workgroup size. Permanently, on this hardware, for any
kernel that barriers once per k-tile.

## 12. Lessons

1. **A flat resource table is not a prediction of flat performance.** Seven prior stages
   trained the expectation that compiler statistics predict the benchmark. Here they were
   complete, accurate, and all flat, while the kernel lost 36 %. The statistics describe one
   workgroup; they say nothing about how many independent workgroups a CU can interleave.
2. **`max_waves` answers "how many waves fit", not "how many independent things can the SIMD
   switch to".** Under a barrier these differ by exactly the workgroup's wave count.
3. **The evidence was in the reference code before the experiment.** `warptile[0]` is the
   workgroup size and it is constant across llama.cpp's tile ladder. The gap analysis recorded
   the tile sizes and missed the invariant. Reading a table is not the same as reading what is
   *not* varying in it.
4. **A negative result narrowed the design space more than a positive one would have.**
   Combined with Stage 8, both axes for enlarging the tile are now closed with measurements,
   which converts split-K from an option into the prerequisite.
