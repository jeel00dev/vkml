# A Working Theory of vkML

Ten stages have produced a collection of engineering laws. This document asks what they are
laws *of* — which are fundamental, which are consequences, and what the whole predicts.

The test of a theory is not that it explains what has happened. It is that it forbids things,
and that the forbidden things turn out not to occur.

Confidence vocabulary: **Proven** (derived, or measured with perfect separation over many
configurations, *and* used to predict an unseen case correctly) · **Strong** (many
observations, no counterexample, no successful novel prediction yet) · **Hypothesis** ·
**Conjecture** · **Open**.

---

## 1. The layer structure

Every established law sits in one of four layers. Laws only depend downward.

```
   L4  NUMERICAL      what vkML guarantees about arithmetic
        |
   L3  PERFORMANCE    what runs fast, and why
        |
   L2  ALLOCATION     what the compiler does with a kernel
        |
   L1  HARDWARE       fixed properties of Navi 10 / RADV
```

The layering matters because it says where a surprise can come from. A performance result that
contradicts an allocation law is a measurement error or a missing L2 term — never a reason to
adjust L1.

---

## 2. Layer 1 — hardware and driver constants

| Symbol | Value | How known |
|---|---|---|
| Compute units | 36 | `VK_AMD_shader_core_properties2` |
| Subgroup | 64 (controllable 32–64) | queried |
| Max workgroup | 1024 invocations | queried |
| LDS | 64 KiB / workgroup | queried |
| Private-array register budget | **64 floats (256 B) per invocation** | measured, §3.1 |
| Scratch allocation granule | **4 floats (16 B) per lane** | measured, §3.2 |

The last two are the interesting ones: they were *discovered*, not queried, and no Vulkan API
exposes them.

---

## 3. Layer 2 — allocation laws

### 3.1 A1 — the private-array budget. **Proven.**

> The register allocator keeps a dynamically-indexed private array in registers up to
> **64 floats**, and spills the entire array beyond that. The threshold is the **array size**,
> not total register pressure.

*Evidence.* Two independent kernels agree. In `gemm_reg.comp`, 27 configurations across six
register-block geometries separate perfectly at 64/80 with no overlap. In
`probe_private_array.comp` — a shader with one array and nothing else, no LDS, no barriers,
no accumulators — the boundary is bracketed exactly: **64 floats clean, 65 floats spilled.**

*What refuted the alternative.* Until M3-R3 the law was stated as a ~100-VGPR ceiling. The
discriminating pair killed it:

```
8x2, L=4 :  stack 64 floats,  VGPR 106  ->  NO SPILL
4x2, L=10:  stack 80 floats,  VGPR 106  ->  SPILLS
```

Equal register pressure, opposite outcome. Total pressure cannot be the criterion.

*Why the probe matters.* Everything before it observed the law inside one kernel, where LDS,
barriers and accumulators were all present and any of them could have been the cause. The
probe removes them and the law survives unchanged — so A1 belongs to the **allocator**, not to
GEMM. This also promotes it from "GEMM tuning fact" to a constraint on *every future vkML
kernel with a private array*.

**Scope, established by deliberate boundary tests (M3-R5).**

| Dimension | Finding | Evidence |
|---|---|---|
| **Units** | **256 BYTES**, not 64 array elements | `vec4[16]` (256 B) clean, `vec4[17]` (272 B) spills, `vec4[64]` (1024 B) spills. Cross-type: `vec4[17]` and `float[65]` are both 68 floats and produce *identical* scratch (17408 B). |
| **Indexing** | Applies only to arrays that **survive as arrays** | A constant-indexed `float[128]` — twice the budget — does not spill and uses 3 VGPRs: it is scalarised into SSA values. The dynamic version spills 32768 B. |
| **Subgroup** | Budget is **per lane**; threshold does not move | 64 clean / 65 spilled at both subgroup 32 and 64, with identical VGPR (77). |
| **Element type** | Type-independent | Only the byte count matters. |
| Driver / GPU | **Unverified** | One machine, RADV, Navi 10. See O5. |

So A1's precise statement is: *a dynamically-indexed private array occupying more than 256
bytes per invocation is spilled in its entirety.* The earlier phrase "64 floats" was a
special case of the byte budget, and "private array" was too broad — constant-indexed arrays
are outside the law entirely.

### 3.2 A2 — scratch volume. **Proven.**

> `scratch_bytes = ceil(bytes_per_lane · subgroup_size / 1024) · 1024`

Derived, not fitted, and generalised twice:

- **Per wave, not per workgroup.** Both readings gave 256 on every geometry measured, because
  all had workgroup 256 *and* subgroup 64. A geometry at workgroup 512 with the subgroup held
  at 64 forces them apart: scratch was **unchanged**, so the multiplier is `subgroup_size`.
- **The granule is 1024 bytes per wave**, not 4 floats per lane. M3-R4 saw a 4-float granule
  at subgroup 64; at subgroup 32 the apparent granule is 8 floats. Both are the same rule:
  `4 floats × 4 B × 64 lanes = 8 floats × 4 B × 32 lanes = 1024 B`. The subgroup-32 measurement
  is what exposed it — a prediction carrying the subgroup-64 granule forward missed 65 floats
  by 512 B (predicted 8704, measured 9216).

Exact on all 13 measured points across two subgroup sizes and two element types.

### 3.3 A3 — register cost of a private array. **Proven** (for `gemm_reg`).

> `VGPR = array_floats + C(RM, RN)`, slope exactly **1.000**.

Zero residual across six geometries. One array float, one VGPR.

### 3.4 A4 — spilling removes exactly the array. **Proven.**

> When A1 triggers, `VGPR == C(RM, RN)` — the A3 constant, exactly.

Measured 26, 23, 35, 35, 42 against predicted 26, 23, 35, 35, 42. **A4 is a consequence of A1
and A3**, not an independent fact: if the array costs `array_floats` registers (A3) and the
allocator moves the whole array to scratch (A1), what remains is precisely `C`. That it holds
to the register is strong evidence the A3 decomposition is *physical* rather than a curve fit —
the two terms can be separated by an experiment.

### 3.5 A5 — `C` is asymmetric in RM and RN. **Proven** (direction and mechanism); closed form **Open**.

```
C(2,2)=18   C(4,2)=26   C(2,4)=23   C(4,4)=35   C(2,8)=35   C(8,2)=42
```

`C(2,8) = 35` against `C(8,2) = 42` at identical `RM·RN` — a 7-register gap. *Cause:* A is read
with stride `BK` (`As[(ty*RM + i)*BK + kk]`), so each of `RM` rows needs its own address
register; B is contiguous and needs one base plus an index. **`RM` costs more than `RN`.**

A linear form fits the first four points exactly and fails the next two. No closed form is
claimed.

### 3.6 A1 applies per array, and register reuse is why. **Strong.**

The obvious reading of A1 — a 64-float budget for the kernel as a whole — is **refuted**:

```
A=40 B=0   total 40 floats -> vgpr 49  scratch 0
A=40 B=40  total 80 floats -> vgpr 57  scratch 0     <- per-kernel budget predicts a SPILL
A=64 B=4   total 68 floats -> vgpr 80  scratch 0
A=32 B=40  total 72 floats -> vgpr 56  scratch 0
```

Two 40-float arrays do not spill even though their sum exceeds 64.

The VGPR column explains why, and is more informative than the scratch column. Eighty floats of
declared array occupy only **57** registers — A3's slope-1 does *not* apply to the sum. In the
probe, array `a` is fully consumed before `b` is created, so their live ranges are disjoint and
the allocator reuses the same registers for both.

> **Refined A1.** The budget is not per declared array and not per kernel: it is per array
> *as sized against the register file at its point of use*. Disjoint live ranges each get the
> full 64 floats because they occupy the same registers.

This is deliberately not stated more strongly. The probe only tested **disjoint** live ranges,
so it cannot distinguish "per array" from "per simultaneously-live set". That distinction is
**O2'** in §7, and it matters: it decides whether a future kernel may hold two working sets at
once.

---

## 4. Layer 3 — performance laws

### 4.1 P1 — independent barrier domains, not waves, govern latency hiding. **Proven.**

> Effective parallelism is `waves_per_CU / waves_per_workgroup` — the count of *independent*
> workgroups, because all waves of one workgroup rendezvous at the same barrier.

Growing the threadblock tile through workgroup size took that count 8 → 4 → 2 and performance
1.00× → 0.84× → 0.77×, with **every compiler statistic flat**. Confirmed by removing one
barrier per k-tile (double buffering), which recovered 1.42× at the large tile against 1.05× at
the small one.

### 4.2 P2 — occupancy is a step function of VGPR. **Strong.**

```
VGPR <=33 -> 20 waves | 41-57 -> 16 | 65-81 -> 12 | 87-99 -> 8
```

Not promoted to Proven: measured only at workgroup 256, and a workgroup-512 configuration
reported **24 waves** and the probe reported **28**, both above the 20 that the table's top row
implies is a ceiling. So `max_waves` is either not waves-per-SIMD or is workgroup-dependent.
**Open question O3.**

### 4.3 P3 — high occupancy can indicate spilling, not efficiency. **Proven.**

A direct consequence of A1 + A4: when the array spills, VGPR collapses to `C`, and the reported
occupancy *rises*.

```
4x4, K=16384, no split-K:  VGPR=35  waves=16  scratch=40960   <- looks BETTER
4x4, K=16384, split-K:     VGPR=99  waves= 8  scratch=    0   <- is BETTER
```

> **Consequence for any future autotuner: reject non-zero `scratch_bytes` before reading
> occupancy.** Ranking on `max_waves` selects the spilling candidate.

### 4.4 P4 — arithmetic intensity is necessary but not sufficient. **Strong.**

Doubling AI from 8.0 to 16.0 FLOP/byte made the kernel *slower*, because the only available
route to a bigger tile cost more in P1 than AI returned.

### 4.5 P5 — throughput collapses below ~90 % GPU fill. **Strong.**

```
fill 1422% -> 1356 GFLOP/s     fill 22% ->  762
fill  356% -> 1224             fill  6% ->  185
fill   89% -> 1029             fill  1% ->   43
```

Necessary but not sufficient: `256×16384×256` reaches 1238 GFLOP/s at the same 22 % fill where
`256×4096×256` reaches 762, because a large K amortises the per-workgroup prologue.

---

## 5. Layer 4 — numerical laws

### 5.1 N1 — the carry-stack equivalence theorem. **Proven** (derived, then verified).

> Partitioning K into contiguous chunks of `2^q` k-tiles and folding the partials with the same
> push/drain algorithm reproduces the unsplit reduction tree **bit for bit**, for any K and any
> partition count.

Proof by the alignment lemma: at a chunk boundary `count = s·2^q` has `q` low zero bits, so no
fold inside a chunk can propagate past level `q-1`; the chunk's internal association is
therefore independent of which chunk it is. Verified in simulation (496 cases, 0 mismatches)
and on hardware (283 shapes × 7 partition counts, identical SHA-256).

### 5.2 N2 — bit-identity holds exactly among 32-block kernels. **Proven.**

`gemm_naive` and `gemm_reg` fold K in blocks of 32 and agree bit-for-bit under every dispatch
switch. `gemm_tiled` folds in blocks of `TILE = 16` and legitimately does not. The scope is the
block size, nothing else.

### 5.3 N3 — the numerical policy is what caps the register block. **Proven.**

The carry stack costs `RM·RN·STACK_LEVELS` registers where every production library pays
`RM·RN`. Combined with A1, this **derives** an impossibility (§6.1).

---

## 6. What the theory forbids

The value of the structure is what it rules out without an experiment.

### 6.1 No register block above 2×2 can avoid spilling for large K

From A1 (budget 64 floats) and the stack size `RM·RN·L`:

```
RM*RN = 4  ->  spills when L > 16   (unreachable: L is bucketed at 16)  -> NEVER spills
RM*RN = 8  ->  spills when L >  8   (i.e. K > ~8k unsplit)
RM*RN = 16 ->  spills when L >  4   (i.e. K > ~256 unsplit)
```

This *derives* what Stage 8 discovered by measurement, and explains the previously unexplained
observation that 2×2 never spills at any K. **Falsifiable:** a 2×2 configuration that spills, or
a 4×4 configuration with `L > 4` that does not, refutes A1.

### 6.2 Larger blocks lose even when the spill is removed

Split-K shortens `L` and removes the spill — measured, 40960 B → 0. The block is *still* slower
(0.884 ms against 0.763 ms) because A3 puts it at 99 VGPRs, which P2 maps to 8 waves against
2×2's 20. **So a larger register block fails for two independent reasons, and fixing one
exposes the other.** Stage 8 saw only the first.

### 6.3 The threadblock tile cannot grow either

Not through registers (§6.1), and not through workgroup size (P1). Those are the only two axes
without a warp-tile hierarchy. **The 32×32 tile is not a tuning choice; it is forced** by A1 and
P1 together, given N3.

---

## 7. Where the theory is incomplete

Honest gaps, ranked by impact × tractability.

| ID | Question | Why it matters | Experiment | Feasible? |
|---|---|---|---|---|
| **O1** | Closed form for `C(RM,RN)` | Would make A3 fully predictive for unseen geometries | Vary RM and RN independently over ≥8 geometries | Yes |
| **O2** | ~~Per-array or per-kernel budget?~~ **Answered in §3.6** — but a sharper question replaces it | see §3.6 | done | — |
| **O9'** | What sets the crossover LOCATION across kernels? MLP is excluded (M4-R5); it sets severity only. Candidates left: work per workgroup, dependency depth, cache hit rate. | Would make P1'' quantitatively portable | A probe varying work-per-workgroup at fixed MLP and occupancy | Yes, but the practical payoff is now small -- see 4.5e |
| **O2'** | Do two *simultaneously live* arrays share one 64-float budget? | §3.6 only tested disjoint live ranges | Interleave the two arrays' uses so both stay live | Yes, cheap |
| **O3** | What does `max_waves` actually count? | P2 is the basis of every occupancy argument | Vary workgroup at fixed VGPR | Yes |
| **O4** | Is the LDS↔register trade modelled at all? | No law covers LDS pressure | LDS-size sweep at fixed registers | Yes |
| **O5** | Does A1 hold on other drivers? | §11 classifies A1 as driver-specific; portability is unknown | Second GPU or AMDVLK/proprietary driver on this one | Partly — AMDVLK is installable |
| **O7** | Do llama.cpp's medium and small warptiles also fit 256 B? | Would turn §12 from coincidence into external corroboration | Arithmetic only, no hardware needed | Yes, trivial |
| **O6** | Instruction count has no law | Predictions missed by 17 % | Sweep at fixed resources | Partly |

**O2' is now the highest-value next experiment**: cheap, the probe already exists, and the
answer determines whether a kernel may hold two working sets at once.

---

## 8. Cross-checks against production libraries

The theory should explain other people's design choices, not just vkML's. It does:

- **Nobody exceeds 256 invocations per workgroup** (llama.cpp 128 for both 64×64 and 128×128;
  CLBlast 256; rocBLAS 128). P1 explains why: workgroup size buys tile area at the cost of
  independent barrier domains.
- **Everyone grows the tile through per-thread work instead** — which is available to them
  because they have no carry stack (N3) and llama.cpp additionally packs accumulators two per
  register in fp16.
- **rocBLAS's tuner spreads `DepthU` evenly across 8/16/32** across 251 tuned solutions. A3
  explains why no single value wins: `L` trades registers against k-tile count, and the optimum
  moves with shape.
- **Nobody guarantees reduction order**, so N3's cost is unique to vkML — and so is N1's
  benefit.

One divergence the theory endorses: vkML's split-K reduction uses a carry-stack fold where all
five production implementations use a flat sequential sum. N1 says that single deviation is what
buys bit-identity.

---

## 9. What the theory says about the next optimization

| Candidate | Does the theory predict success? | Confidence |
|---|---|---|
| **GEMV kernel** | **Yes.** P5 says a 4-tile shape runs at 3 % of peak; a GEMV kernel changes the tile count, which is the term P5 identifies. No law opposes it: it does not touch A1 (no large private array), P1 (no barrier growth) or N1 (fold order per output is unchanged). | **Strong** |
| Split-K heuristics | Yes — mechanism implemented and measured 2.0–4.2×; P5 supplies the trigger and A1 explains the register bonus. | Strong |
| Autotuner over fold-neutral params | Yes — A1–A4 let candidates be **rejected without timing them**, which no studied project can do. | Strong |
| Warp-tile hierarchy | **Unclear.** It would decouple tile area from both failed axes (§6.3), but A3 says registers still scale with per-thread accumulators, so it may only relocate the limit. | Moderate |
| Larger register blocks | **No — forbidden by §6.1 and §6.2.** | Proven negative |

**Recommendation: GEMV.** It is the only remaining candidate the theory positively predicts
will succeed, where all six studied projects agree vkML has a real gap, and where no
established law opposes it.

---

## 10. Scorecard

Predictions made before the experiment that produced them, across M3-R2 → M3-R4:

```
correct:  slope 1.000 on 3 unseen geometries · scratch law on 3 new points ·
          2x4 wave sequence · 2x4 spill point · C(8,2) exactly · 2x8 no-spill ·
          split-K removes the 4x4 spill (exact) · C independent of workgroup size ·
          scratch is per-wave not per-workgroup · A1 general, threshold exactly 64
wrong:    C(2,4) (two models, -2 and -3) · C(2,8) (-2) · 8x2 spill (produced A1) ·
          instruction count (+17 % vs +-5 %) · scratch at 65 floats (found the granule)
```

Ten correct, five wrong. Every one of the five failures produced a law: the `8x2` miss replaced
a false VGPR threshold with A1, and the 65-float miss produced A2's rounding term.

That ratio is the point. A theory that only ever confirms itself is not being tested hard
enough.

### 4.5b P1' — latency hiding has two terms, and they define two regimes. **Strong (M4-R2).**

M4-R1 left two live explanations for GEMV's residual gap and preferred one on parsimony alone:

- **H1** occupancy limited by the 16.9 KiB LDS footprint;
- **H2** cost of the 6-step cross-lane reduction, paid per output element.

Both explained the 1.8x gap and the N-scaling. **Neither had been eliminated.**

*The discriminating experiment.* A dummy shared array sized by a specialisation constant
changes the workgroup's LDS footprint and **nothing else** — same instructions, same barriers,
same memory pattern, same fold order. H1 predicts monotonic slowdown; H2 predicts a flat line.

```
 pad     LDS    driver waves   concurrent wg/CU   measured   vs pad=0
  0 KB  16896        4                3            2.665m     1.00x
  4 KB  20992        4                3            2.991m     1.12x
  8 KB  25088        2                2            3.659m     1.37x
 16 KB  33280        2                1           10.167m     3.82x
 32 KB  49664        2                1           10.433m     3.91x
```

> **H2 is REJECTED.** Padding cannot touch the reduction, yet performance falls 3.9x. A
> mechanism insensitive to LDS cannot produce this curve.

*But H1 alone does not fit either.* Predicted from concurrent workgroups: 1.00 / 1.00 / 1.50 /
3.00 / 3.00 against measured 1.00 / 1.12 / 1.37 / 3.82 / 3.91 — and the same rule
**over-predicts M3-01 badly** (0.50 / 0.25 predicted against 0.84 / 0.77 measured).

The resolution unifies both experiments under one mechanism with two terms, both derived from
`wg/CU = min(LDS_budget / LDS_wg, wave_budget / waves_wg)`:

```
resident waves/CU  = wg/CU x waves/wg      raw capacity to hide latency
independent groups = wg/CU                 how decorrelated that capacity is
```

- **M3-01 was wave-limited**: resident waves stayed at 64 throughout (8x8, 4x16, 2x32), so
  capacity was saturated and only independence moved — a *second-order* effect, 8 -> 2 groups
  costing 23 %.
- **M4-R2 is LDS-limited**: resident waves fell to 3, 2, 1, so capacity itself collapsed and the
  loss went *super-linear* — 3 -> 1 waves costing 282 %.

This **replaces two independent observations with one relationship in two regimes**, which is
why it is stated as P1' rather than added as a separate law. It is qualitative: the 23 % and
282 % are not derivable from it, so it is Strong evidence, not Proven.

*Consequence for O3.* The driver reported `waves = 2` for both the 2-workgroup and the
1-workgroup configuration, which differ by **2.8x in time**. `max_waves` is confirmed too coarse
to serve as an occupancy proxy; `min(LDS_budget/LDS_wg, wave_budget/waves_wg)` is computable
from values vkML already has and predicts where `max_waves` does not. **P2 is superseded for
LDS-limited kernels.**

### 4.5c P1'' — the quantitative form: occupancy is an INTEGER step function. **Strong (M4-R3).**

M4-R2 left P1' qualitative. The natural-units reformulation is:

```
LDS_per_wave      = LDS_per_workgroup / waves_per_workgroup
resident_wg/CU    = floor( LDS_budget / LDS_per_workgroup )      <- INTEGER
resident_waves/CU = min( resident_wg x waves_per_wg , wave_budget )
```

`LDS per wave` is the single number that decides whether a kernel can hide latency:

```
baseline 32x32   8192 B / 8 waves =  1024 B/wave -> 64 resident waves
M3-01 64x64     16384 B / 32      =   512 B/wave -> 64 (wave-capped)
GEMV wg=64      16640 B / 1       = 16640 B/wave ->  3 resident waves
```

GEMV is **16x worse per wave** than the baseline, and this is invariant to workgroup size:
scaling WG scales its LDS proportionally. That one number explains why GEMV cannot win, and it
supersedes the "less LDS" advice from M4-R1, which targeted the wrong quantity.

#### The measurement

A 9-point sweep of the LDS pad on a fixed workload (M=K=4096, N=1), 20 repetitions each:

```
 wg/CU  n            times (ms)          mean   spread
     3  3   2.755  2.695  2.833          2.761    5.0 %
     2  3   3.444  3.527  3.657          3.543    6.0 %
     1  3   9.984 10.354 10.494         10.277    5.0 %
```

**The prediction that time varies continuously as 1/resident_waves is refuted.** Time is *flat*
within a plateau and jumps between plateaus, because `floor()` quantises the hardware:
LDS 16896, 18944 and 20992 B all give 3 workgroups and all run at ~2.76 ms.

#### Measured parameters

```
 3 wg/CU -> 1.00x      2 wg/CU -> 1.28x      1 wg/CU -> 3.72x
```

The 2 -> 1 step costs **2.90x for a 2x reduction** -- 45 % worse than linear. That is not a
tuning constant but a qualitative change: with a single resident workgroup of one wave, a CU
that stalls at a barrier or a memory access has *nothing else to run*. With two it can alternate.
**The regime boundary is the integer transition 2 -> 1, not a continuous knee.**

#### Accuracy

Over the same nine points:

```
continuous 1/w model : mean error 21.5 %   max 45.6 %
step model           : mean error  1.9 %   max  3.1 %
```

An order of magnitude, obtained by correcting the model's *structure* (integer quantisation)
rather than by fitting parameters. Model uncertainty is now bounded by the within-plateau
spread, ~6 %, which is measurement noise rather than model error.

#### What this predicts, and where it stops

- Any kernel's resident waves are computable from `LDS_per_workgroup` and `waves_per_workgroup`
  before it is written -- both known at dispatch-configuration time.
- The multipliers 1.00 / 1.28 / 3.72 are measured **for this workload on this GPU**. The
  *structure* (flat plateaus, integer steps, a cliff at 1) should generalise; the numbers are
  local and should be re-measured per workload.
- Above ~8 resident waves the plateaus flatten out entirely (M3-01 held 64 resident waves across
  three geometries and varied only 23 %), so this model applies in the **scarce** regime. The
  crossover between scarce and saturated is not yet measured -- see O9.

### 4.5d Cross-kernel validation of P1'' (M4-R4)

P1'' was derived entirely from GEMV. Applying it unchanged to `softmax` -- a different
algorithm, different access pattern, different reduction, 256-thread workgroups -- tests
whether it is a theory or a curve fit.

**A1 first, at zero cost.** `reduce.comp` and `softmax.comp` both carry `float stack[24]`
(96 B), dynamically indexed by a data-dependent push loop. A1 predicts no spill, and every
existing kernel measures `scratch = 0, spilled = 0`:

```
cast     vgpr  4  lds     0    reduce   vgpr 32  lds 1024
unary    vgpr  4  lds     0    softmax  vgpr 34  lds 1024
gemm_reg vgpr 41  lds  8192
```

A1 **generalises unchanged** to kernels it was not derived from.

**P1'' by LDS padding on softmax**, sweeping resident waves 64 -> 4:

```
 padKB    LDS  resident      GPU time   vs pad0
     0   1024        64        4.970m     1.00x
     4   5120        48        5.021m     1.01x
     8   9216        28        5.023m     1.01x
    16  17408        12        5.083m     1.02x
    32  33792         4        7.278m     1.46x
    48  50176         4        7.349m     1.48x
```

> **The structure generalises; the crossover does not.** Flat plateau then degradation is
> reproduced on a completely different kernel. But saturation extends down to **12** resident
> waves, not 28 as predicted: the crossover is in **[4, 12]**, and the M4-R3 prediction of
> [12, 28] was wrong by about 2x.

This answers **O9** from a second kernel: degradation begins below ~12 resident waves and
becomes severe below ~2 (GEMV: 3.72x at 1 wave; softmax: 1.46x at 4).

**Magnitudes are kernel-specific, structure is not.** The multipliers 1.00 / 1.28 / 3.72
measured for GEMV do not transfer -- softmax loses only 1.46x at 4 waves where GEMV lost 1.28x
at 2. P1'' should be read as: *predict the regime, not the multiplier.*

### 4.5e The crossover question, partially resolved (M4-R5)

The last open question was why kernels stop hiding latency at different resident-wave counts
(GEMV ~2, softmax ~4-12). Three hypotheses were live: **H1** memory-level parallelism,
**H2** arithmetic between stalls, **H3** synchronisation density.

`probe_mlp.comp` varies MLP -- loads in flight per wave before any is consumed -- while holding
total bytes, arithmetic per byte, barrier count and private-array size constant. Sweeping MLP
against resident waves (penalty relative to each row's own 64-wave baseline):

```
 waves     MLP=1     MLP=2     MLP=4     MLP=8
    32     1.02x     1.00x     1.01x     0.99x
    16     1.28x     1.06x     1.03x     1.02x
     8     2.45x     1.87x     1.59x     1.45x
     4     4.47x     3.85x     3.35x     3.10x
```

**H2 and H3 are excluded.** Arithmetic per byte and barrier count are identical at every point,
yet behaviour changes with MLP.

**H1's strong form is REFUTED.** Little's law predicts crossover proportional to 1/MLP --
12.5 / 6.2 / 3.1 / 1.6 waves. Measured: MLP=1 degrades at 16 waves, MLP=2, 4 and 8 all degrade
at 8. An **8x change in MLP moved the crossover by at most 2x**. The product `waves x MLP`,
which Little's law says should be the governing quantity, does not collapse the data either:
at `waves x MLP = 32` the times span 3.03x.

**H1's weak form is CONFIRMED.** MLP does not decide *where* the regime changes; it decides
*how bad* the scarce regime is. At 4 resident waves the penalty falls monotonically 4.47x ->
3.10x as MLP rises 1 -> 8.

> **P1'' gains one parameter, not a new mechanism.**
> ```
> resident waves  ->  WHICH regime          (saturated / scarce)
> MLP             ->  HOW SEVERE the scarce regime is
> ```

#### What remains unexplained

MLP does **not** account for the kernel-to-kernel variation it was proposed to explain. The
probe crosses over at 8-16 resident waves for every MLP tested, while GEMV crosses at ~2 and
softmax at ~4-12. Those differ by more than the entire MLP sweep produces.

Expressed in resident *workgroups* the three are closer -- probe ~4, GEMV ~2, softmax ~1 -- but
still a 4x spread, so that reformulation does not close it either.

**Stated plainly rather than forced: no single measured parameter predicts the crossover
location across kernels.** H4 stands to that extent. The practical consequence is bounded and
already reflected in how the theory is used: predict the *regime* from resident waves, treat the
crossover as lying somewhere in **2-16 resident waves**, and do not quote a kernel-specific
number without measuring it.

### 4.6 P6 — a numerically-constrained fold order conflicts with coalescing. **Proven (M4-R1).**

> When the order in which a kernel must FOLD differs from the order in which the memory system
> wants it to LOAD, the two cannot both be satisfied reading directly from global memory.
> Shared memory is the mechanism that decouples them.

Discovered by a refuted prediction. A GEMV kernel in which each lane folded a contiguous run of
K-tiles straight from global memory was bit-exact and **5.6× slower** than the tiled kernel it
was meant to beat. Three requirements were in conflict:

```
N2            blocks of 32 summed SEQUENTIALLY
N1            each lane owns a CONTIGUOUS run of tiles
memory system consecutive LANES read consecutive ADDRESSES
```

Any two are satisfiable; all three are not. At a fixed step, neighbouring lanes were reading
addresses 256 bytes apart — 64 distinct cache lines per instruction.

Staging the row through LDS satisfies all three (load order and fold order become independent)
and recovered **3.1×**. This reframes something the model had recorded but not explained: the
LDS staging in `gemm_reg.comp` is not only about operand reuse. **It is what allows a kernel to
have a numerically-required fold order at all.** Every production GEMM stages through shared
memory even where reuse is low, and P6 is why.

### 4.7 The GEMV outcome, explained by existing laws

The staged kernel is still 1.8× slower than the baseline, and this needs **no new law**:

```
GEMV v2 :  LDS 16896 B, 64 threads = 1 wave  -> 3 workgroups/CU ->  4 waves/CU
baseline:  LDS  8192 B, 256 threads = 8 waves -> 8 workgroups/CU -> 64 waves/CU
```

**16× less latency hiding** — P1 plus the LDS occupancy relation. The fix bought coalescing by
spending occupancy. A working design must obtain coalescing *without* a 16.9 KiB per-workgroup
footprint: fewer staged elements per pass, or more outputs per workgroup so the reduction and
the LDS are amortised. That is a design change, not a tuning knob, and belongs to its own stage.

Retained as `VKML_GEMV=FORCED`, default OFF — the same disposition as `gemm_db`, the tile
geometries and the register blocks: measured, understood, disabled, kept.

---

## 11. Scope of every law (M3-R5)

A law without a stated domain is incomplete. Each classification below is justified, and
"universal" is never claimed merely because no counterexample has been looked for.

| Law | Scope | Justification |
|---|---|---|
| **A1** private-array budget = 256 B/lane | **Driver-specific** (RADV), plausibly **architecture-specific** | Boundary bracketed exactly on two unrelated shaders, invariant to element type and subgroup size — so it is not shader-pattern specific. But a register-file budget is a property of the allocator, and only one driver was tested. |
| **A1 scope: dynamic indexing required** | **Universal (compiler principle)** | Scalarisation of constant-indexed arrays is standard SSA construction, not a RADV behaviour. Confirmed here; expected anywhere. |
| **A1 scope: budget in bytes** | **Universal (hardware principle)** | Registers are byte-addressed storage; an element-count budget would be arbitrary. Confirmed across `float` and `vec4`. |
| **A2** scratch = 1024 B/wave granule | **Driver-specific** | 1024 B is a RADV allocation choice. The *form* (per-wave, granular) is likely universal; the constant is not. |
| **A3** VGPR = floats + C, slope 1.000 | **Shader-pattern specific** | Holds for `gemm_reg`'s single live array. The probe showed slope ≠ 1 when a second array shares registers, and C varies with the modulo codegen. Do not apply across kernels. |
| **A4** spilled VGPR == C | **Consequence of A1 + A3** | Same scope as its weaker parent, A3. |
| **A5** C asymmetric in RM/RN | **vkML-specific** | A property of `gemm_reg.comp`'s access pattern, not of any compiler. |
| **P1** independent barrier domains | **Universal (execution-model principle)** | Follows from barrier semantics, not from any hardware detail: waves of one workgroup cannot proceed independently past a barrier. Externally corroborated — no studied project exceeds 256 invocations per workgroup. |
| **P2** occupancy step function | **Driver-specific and provisional** | Measured only at workgroup 256; `max_waves` reported 24 and 28 elsewhere, above the 20 the table implies is a ceiling. **O3.** |
| **P3** high occupancy can mean spilling | **Consequence of A1 + A4** | Scope inherited. The *warning* generalises; the numbers do not. |
| **P5** fill → throughput collapse | **Architecture-specific** | 36 CUs is in the constant. The shape of the curve is general; the 90 % knee is not. |
| **N1** carry-stack equivalence | **Universal (mathematical)** | A theorem about float addition association. Independent of hardware, driver and compiler. The only law here that is *derived* rather than measured. |
| **N2** bit-identity among 32-block kernels | **vkML-specific** | A statement about vkML's own kernels. |
| **N3** numerical policy caps the register block | **vkML-specific** | Follows from a design choice no other project makes. |

**Only three laws are universal**, and their character is instructive: N1 is mathematics, P1 is
execution-model semantics, and A1's two scope clauses are compiler and hardware principles. All
the *numbers* are local. That is the honest summary — vkML has one portable theorem, two
portable principles, and a lot of well-characterised local constants.

---

## 12. External validation: does A1 explain someone else's design?

A theory earns credibility by explaining decisions made without knowledge of it.

llama.cpp's largest non-coopmat tile (`l_warptile`, subgroup 64) allocates
`sums[WMITER·TM·WNITER·TN/2]` accumulators (`mul_mm.comp:270`):

```
WNITER = (WM*WN)/(WARP*TM*TN*WMITER) = (128*64)/(64*4*4*2) = 4
sums[] = 2*4*4*4/2 = 64 elements of ACC_TYPEV2

   fp16 accumulate (f16vec2, 4 B):  64 x 4 = 256 bytes   <- exactly vkML's A1 budget
   fp32 accumulate (vec2,    8 B):  64 x 8 = 512 bytes   <- twice the budget
```

`ACC_TYPE = f16acc ? "float16_t" : "float"` (`vulkan-shaders-gen.cpp:470`), and fp16 is the
default with `f32acc` an opt-in.

So llama.cpp's default configuration lands **exactly on** the byte budget A1 identifies, and its
fp32 variant is exactly 2× over it.

**O7, resolved.** The follow-up was pure arithmetic, so it was done rather than deferred.
Across *every* non-coopmat warptile llama.cpp defines, at subgroup 64:

```
warptile      WNITER  sums[]  fp16 B  fp32 B    vs 256 B
generic  l         4      64     256     512    EXACTLY AT
generic  m         2      16      64     128    under
generic  s         2       8      32      64    under
AMD+coop l         4      32     128     256    under
Intel Xe l         2      16      64     128    under
mmq int  l         4      64     256     512    EXACTLY AT
mmq intk l         8      64     256     512    EXACTLY AT
```

**Three configurations land exactly on 256 bytes; four land under; none exceeds it.**

The third row is the most telling. `mmq_int_k` is the one llama.cpp annotates *"K-quants use
even more registers, mitigate by setting WMITER to 1"* — a deliberate register-pressure fix.
Setting `WMITER = 1` pushes `WNITER` to 8, and `sums[]` lands back at exactly 64 elements =
256 bytes. They reduced register pressure and stopped precisely at the same ceiling.

**Classification: strong external evidence, still not proof of causation.** Two reasons to stop
short of "confirmed":

1. This is llama.cpp's parameters combined with *vkML's* measured budget on *vkML's* hardware.
   Their tiles were tuned by benchmarking across many GPUs; a shared ceiling is consistent with
   a shared hardware constraint but does not demonstrate they knew of it.
2. The `f32acc` variants would be 512 B — twice the budget — and presumably still ship. Either
   they spill and it is accepted, or their targets differ. Untested, and the obvious next probe.

A ceiling that seven independent configurations respect, that three hit exactly, and that a
deliberate register-pressure fix converges back onto, is considerably more than a coincidence.

---

## 13. Prediction scorecard, M3-R5

```
correct:  S1 budget is bytes (vec4[16] clean / vec4[17] spills, predicted exactly)
          S2 constant indexing escapes the law entirely
          S3 threshold invariant to subgroup size
          S3 scratch halves at subgroup 32 for multiples of 8 floats
wrong:    S3 scratch at 65 floats, subgroup 32: predicted 8704, measured 9216
          -> carried the subgroup-64 granule forward; produced A2's 1024-byte form
```

Four correct, one wrong. Running total across M3-R2 → M3-R5: **fourteen correct, six wrong**,
and every one of the six produced or corrected a law.
