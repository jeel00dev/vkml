# vkml performance model

How this project reasons about GPU performance, and why it measures the things
it measures. Written after M1 and the first GEMM stages, where several
confidently-held beliefs turned out to be measurement artifacts.

**Target device throughout:** AMD RX 5600M (Navi 10, RDNA1), 36 CUs, ~5.8 TFLOPS
fp32 peak, ~288 GB/s memory bandwidth, 64 KiB LDS per workgroup, subgroup width
selectable 32–64, no cooperative matrix, no global float atomicAdd.

---

## 1. Arithmetic intensity

The ratio of arithmetic to memory traffic, in FLOP per byte:

```
AI = FLOPs performed / bytes moved through global memory
```

It determines which resource a kernel is limited by. The device's balance point
is roughly

```
5.8e12 FLOP/s / 288e9 B/s ~= 20 FLOP/byte
```

Below that, a kernel is bandwidth-bound; above it, compute-bound. Every kernel
in vkml today is below it:

| kernel | AI | bound by |
|---|---|---|
| elementwise (relu, exp) | 0.17 | bandwidth |
| reduction (sum, max) | 0.25 | bandwidth |
| GEMM naive | 0.25 | bandwidth |
| GEMM tiled, TILE=16 | 4.0 | bandwidth |
| GEMM register-blocked 2×2 | 8.0 | occupancy (see §2) |

For a tiled GEMM, AI follows directly from the tile geometry:

```
AI = 2·BM·BN·BK / ((BM·BK + BK·BN) · 4 bytes)
```

Note that BK cancels for square tiles — **increasing BK does not change
arithmetic intensity**. It was still worth doing (Stage 5.75), for register
reasons, which is exactly the kind of thing this model exists to keep straight.

### The trap

Measured global traffic can *exceed* DRAM bandwidth. The naive GEMM reported
683 GB/s on a 288 GB/s device. That is not an error: the counter measures
memory *requests*, and L2 was serving most of them. The excess is a direct
measure of cache reuse — and it correctly predicted that tiling would win far
less than the 16× traffic reduction suggested (it won 2.6×), because the traffic
was never reaching DRAM in the first place.

---

## 2. Occupancy and register pressure

Occupancy is how many waves a SIMD can host concurrently. It is what hides
memory latency: with few waves resident, a stall has nothing to switch to.

On this device the ceiling is **20 waves/SIMD**, and the usual limiter is VGPR
count per invocation. The compiler reports both.

This is the single most important lesson from the GEMM work:

> An optimisation that increases work per thread also increases register
> pressure, and the occupancy loss can cancel the entire gain.

Measured, at Stage 5:

```
                    VGPR  waves/SIMD  instr/output  net vs tiled
tiled (1×1)           48          20          1.86         1.00×
reg   (2×2) BK=16     84          12          0.93         1.03×   <- no gain
reg   (2×2) BK=32     64          16          0.93         1.22×
```

Register blocking delivered exactly what it promised — **2.01× fewer
instructions per output** — and gave all of it back through a 40 % occupancy
loss. The fix was not to abandon it but to find the registers: 57 % of the
kernel's VGPR budget was the pairwise carry stack, and sizing that from the
actual K rather than a worst case recovered 20 VGPRs and 4 waves/SIMD.

### The model

```
predicted speedup = (instruction efficiency gain) × (occupancy ratio)
```

It over-predicts absolute levels — it assumes throughput scales linearly with
occupancy, which holds only while latency-hiding is the limiter — but it tracks
*changes* well. Stage 5.75 predicted a 1.33× improvement from occupancy alone
and measured 1.36×.

Use it for deltas, not for absolute forecasts.

---

## 3. Shared memory

LDS is 64 KiB per workgroup on this device and is rarely the binding
constraint — vkml's most demanding kernel uses 8 KiB. It is therefore the
resource to **spend** when trading against registers, which are scarce.

Stage 5.75 is the worked example: doubling BK doubled LDS (4 KiB → 8 KiB) to buy
back 20 VGPRs. Both are "memory", but only one was scarce.

LDS can limit occupancy independently of registers when a workgroup's allocation
divides poorly into 64 KiB. Watch the reported `waves_per_simd` rather than
computing it.

---

## 4. Timestamp profiling

**Never evaluate kernel performance with wall-clock timing.**

Wall clock includes upload, submission, synchronisation and download. On this
machine those dominate: a `relu 1024×1024` is 0.29 ms of GPU work inside 2.5 ms
of wall clock, and before the staging path was fixed it was 24 ms.

The failure this caused is instructive. Wall clock reported softmax as **50×
slower than sum**. The real GPU ratio is **2.4×** — correct for three passes
against one — and the rest was download time, because softmax returns a 4 MB
result and sum returns 4 KB. Acting on that number would have meant optimising a
kernel that was already fine.

vkml measures GPU time with `VK_QUERY_TYPE_TIMESTAMP` (`timestampPeriod` = 10 ns
here), `TOP_OF_PIPE` to `ALL_COMMANDS`, resolved after the wait. Off by default,
enabled with `vulkan_set_profiling(True)`.

Benchmarks report GPU time, upload, download and wall clock **separately**.
Combining them is how the softmax mistake happened.

---

## 5. Compiler resource statistics

`VK_KHR_pipeline_executable_properties` makes the driver report, per compiled
pipeline: VGPRs, SGPRs, spilled registers, scratch bytes, LDS bytes, waves per
SIMD, instruction count and code size.

vkml normalises vendor-specific statistic names into a `PipelineStats` struct
inside the Vulkan backend; no raw driver strings escape that layer. RADV calls
occupancy "Subgroups per SIMD" and other drivers say "Waves" — matching both
belongs in one place, not at every call site.

These counters answer questions timestamps cannot. Stage 5.5 had a specific,
plausible hypothesis — that dynamic indexing into the carry stack was spilling
to scratch memory — and one query settled it in minutes:

```
Spilled VGPRs = 0    Spilled SGPRs = 0    Scratch size = 0
```

Hypothesis rejected. The real cause was occupancy, visible in the same output.
Without the query that would have been a day of pointless restructuring.

---

## 5b. Static instructions vs dynamic issue

Global vec4 loads gave a **1.39×** speedup (Stage 6) with *identical* VGPRs,
SGPRs, LDS and occupancy, and with a **higher** static instruction count
(1084 vs 1036). Widening LDS reads for the same kernel (Stage 6.5) gave
**nothing**: 1.833 → 1.768 ms, inside noise.

Both results turn on the same distinction.

**Static instruction count is what the compiler emitted. Dynamic issue is what
executes.** They diverge whenever a branch is gated on a specialization constant
mixed with a runtime condition: the vec4 build contains *both* the vector path
and the scalar fallback, so it is larger, while the taken path issues 4× fewer
memory instructions. Judging that kernel by its instruction count would have
rejected a 39 % win.

So the model needs a term the first version lacked:

```
predicted speedup = (instruction efficiency) × (occupancy) × (issue-rate relief)
```

The third term only pays when the widened access is on the **critical issue
path**. Global loads were: the kernel was load-issue-bound at 153 GB/s of 288
and 1.23 of 5.8 TFLOPS — saturating neither headline resource. LDS reads were
not, because the B-tile read was already conflict-free and cheap relative to the
global fetch it feeds.

**A resource being *used* is not the same as it being *binding*.** Before
widening an access, establish that it is on the critical path, not merely that
it is frequent.

### And a warning about controls

The Stage 6.5 experiment first appeared to give a 1.28–1.42× speedup. It did
not. The A/B toggle's "scalar" arm had been rewritten to index a `vec2` array
component-wise (`Bs2[i/2][i%2]`), which is *worse* than the plain float array it
replaced — 1498 instructions against the real baseline's 1084. The comparison
was internally consistent, low-variance and plausible in magnitude, and still
measured nothing but the damage done to its own control.

**An A/B toggle is only valid if the A arm is the frozen baseline.** Compare
against the recorded baseline, not against a path modified in the same change.

---

## 5c. Latency hiding: occupancy and software pipelining are substitutes

Double buffering the K-tile pipeline (Stage 7) produced **nothing**: 1.007× at
512³, 0.989× at 1024³. Registers were untouched (64 VGPRs both ways) and
occupancy was untouched (16 waves/SIMD), so the change was clean -- it simply
did not matter.

The reason is worth stating, because it is not obvious:

> Double buffering hides memory latency *within* a wave, by overlapping the next
> tile's loads with this tile's arithmetic. Occupancy hides the same latency
> *across* waves, by switching to another wave when one stalls. **They are
> substitutes.** At 16 waves/SIMD there is already something else to run, so
> adding instruction-level overlap buys nothing.

Software pipelining pays when occupancy is low — when a stalled wave leaves the
SIMD with nothing to do. That is not this kernel's situation.

A useful corollary: doubling LDS from 8 KiB to 16 KiB cost **zero** occupancy,
confirming §3 — LDS really is the resource to spend here. It just bought
something that was not needed.

### What this rules out

By this point the GEMM kernel has been measured against every resource the model
tracks, at ~1.23 TFLOP/s (21 % of the tuned reference):

| candidate | status | evidence |
|---|---|---|
| DRAM bandwidth | ruled out | 153 GB/s of 288 |
| compute throughput | ruled out | 1.23 of 5.8 TFLOPS |
| register spilling | ruled out | scratch = 0, spills = 0 |
| occupancy | ruled out as *dominant* | 16/20; Stage 5.75 recovered it, gain was 1.22× |
| LDS issue rate | ruled out | Stage 6.5, no change |
| LDS bank conflicts | ruled out | access pattern analysed conflict-free |
| global load issue rate | **confirmed contributor** | Stage 6 vec4 gave 1.39× |
| load/compute latency | ruled out | Stage 7, no change |

What remains is the **instruction mix of the inner loop**. At a 2×2 register
block, each `kk` issues 2 scalar A reads plus 1 vec2 B read for 4 FMAs — only
**57 % of issued instructions are FMAs**, before address arithmetic. Widening the
register block is the lever that changes that ratio:

```
2×2:  3 LDS + 4 FMA  -> 57% FMA, 0.75 LDS/FMA
4×2:  5 LDS + 8 FMA  -> 62% FMA, 0.63 LDS/FMA
4×4:  6 LDS + 16 FMA -> 73% FMA, 0.38 LDS/FMA
```

---

## 5d. The register allocator has two failure modes

Stage 8 scaled the register block from 2×2 to 4×2 and 4×4, holding the workgroup
at 256 invocations so occupancy changes could only come from register pressure.
**Both regressed.** Measured at 1024³:

```
block  GFLOP/s  vs 2×2  VGPR  waves/SIMD  scratch
2×2       1270   1.000×    64          16        0
4×2       1125   0.886×    84          12        0
4×4        999   0.787×    64          16    24576
```

The compiler statistics predicted this **before any benchmark ran**, which is
the point of collecting them.

### Where the model worked

For 4×2 the two-term model is nearly exact:

```
FMA share  50% -> 57%   = 1.14× instruction efficiency
occupancy   16 -> 12    = 0.75×
predicted             0.857×      measured 0.886×   (3% error)
```

Larger register blocks *do* improve arithmetic density, exactly as hypothesised.
The gain is simply smaller than the occupancy it costs.

### Where the model failed

For 4×4 it failed outright. The prediction was ~140 VGPRs and an occupancy
collapse. What actually happened: the allocator **capped registers at 64 and
spilled 24 KiB to scratch memory**. Occupancy stayed at 16 — there is no
occupancy term to blame — and performance fell 21 % anyway.

> **The register allocator has two failure modes, and which one it chooses is
> not predictable from a register estimate.** It can spend registers and lose
> occupancy (4×2), or cap registers and spill to scratch (4×4). A model that
> only tracks occupancy sees the first and is blind to the second.

Note that Stage 5.5 explicitly *ruled out* spilling — correctly, at the time.
`scratch = 0` held through every stage until 4×4. A resource being absent for
seven stages is not evidence it will stay absent.

The practical consequence: **`scratch_bytes` must be checked before any timing
is interpreted.** Non-zero scratch invalidates reasoning about every other
resource, because accumulator traffic has moved to memory.

---

## 5e. Waves per SIMD is not occupancy — count independent barrier domains

M3-01 enlarged the threadblock tile (32×32 → 64×32 → 64×64) while holding the register block
at 2×2, so per-thread registers could not move. Arithmetic intensity doubled, 8.0 → 16.0
FLOP/byte. **Every compiler statistic came out exactly as predicted, and every shape got
slower.**

```
tile     wg    LDS     VGPR  waves/SIMD  scratch  instr   1024³     vs 32×32
32×32   256   8 KiB      41          16        0   1124   1.807ms     1.000×
64×32   512  12 KiB      41          16        0   1130   2.141ms     0.844×
64×64  1024  16 KiB      41          16        0   1126   2.333ms     0.774×
```

This is the first regression in the project that the compiler statistics did **not** predict.
They were not wrong — they were complete and flat. The model reading them was wrong.

### The missing term

`max_waves = 16` was identical in all three cases, and the model treated that as "occupancy is
unchanged". It is not, because **the waves of one workgroup all rendezvous at the same
barrier**. They are concurrent but not independent: when a workgroup stalls at
`barrier()`, every one of its waves stalls together, and the SIMD can only hide that latency
with waves from a *different* workgroup.

The quantity that actually matters is therefore concurrent workgroups per CU — the number of
independent barrier domains:

```
tile     waves/CU   waves/wg   INDEPENDENT WORKGROUPS/CU
32×32          64          8            8
64×32          64         16            4
64×64          64         32            2
```

It halves at each step, and performance falls monotonically with it. Doubling the tile
doubled intensity; it also quartered the number of things the CU could switch to.

### Confirmation

Double buffering removes one of the two barriers per k-tile. If synchronisation is the cost,
it must help far more at the large tile — and it does:

```
              no DB      +DB     recovered
32×32, 2048³  7.093ms  6.737ms      1.05×
64×64, 2048³ 11.025ms  7.782ms      1.42×
```

A 1.42× recovery at 64×64 against 1.05× at 32×32 is the mechanism, measured directly.

### The fill curve, measured

M3-01 established the metric qualitatively; M3-02 measured it. Fill is
`tiles / (36 CUs x 8 concurrent workgroups per CU)`, against the existing 32x32 kernel:

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

Throughput is flat above ~90 % fill and collapses below it: the 4-tile case runs at **3 % of
the full-fill rate**, a 32x gap. Fill is therefore the dominant term for any shape whose tile
count is below about `2 x CU_count`.

**Fill is necessary but not sufficient.** `256x16384x256` reaches 1237.9 GFLOP/s at the same
22 % fill where `256x4096x256` reaches 762.2 — with very large K the per-workgroup prologue
amortises. Predict from fill, then check whether K is large enough to hide the prologue; do not
treat fill alone as the answer.

### Consequence for the model

Add to the occupancy analysis of §2: after computing waves/SIMD, divide waves per CU by waves
per workgroup. **If that number falls below about 4, treat the configuration as
latency-exposed regardless of what `max_waves` reports.** Growing a workgroup past ~256
invocations on this GPU costs more in lost independence than any intensity gain it buys.

Production practice agrees and is unambiguous: llama.cpp uses **128 invocations for both its
64×64 and its 128×128 tile** (`ggml-vulkan.cpp:4030-4032`), CLBlast's tuned `gfx1010` entry
uses 256 threads for 64×64, and rocBLAS's navi21 SGEMM uses 128 for a 128×64 macro tile.
None of them grows the workgroup to grow the tile; all of them grow per-thread work instead.

---

## 5f. Timestamp profiling cannot measure CONCURRENT dispatches

Split-K (M3-03) is the first vkML operation that issues several independent dispatches with no
barrier between them, so that the driver may overlap them. Measured with the profiler on, it
looked like a catastrophic regression:

```
64x16384x64      profiled GPU sum      wall clock, profiling OFF
  split-K off          2.769 ms                3.127 ms
  split-K, 8 splits    6.286 ms                1.140 ms      <-- 5.5x disagreement
```

A profiled time of 6.286 ms against a wall-clock time of 1.140 ms is impossible unless the
measurement changed the execution. It did.

**Cause.** `Recorder::end_timestamp()` writes its timestamp at
`VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT` (`vk_command.cpp:93`). That stage drains the pipeline,
so a timestamp between two dispatches forces the first to complete before the second starts.
For a chain of dependent dispatches -- everything vkML had until now -- this costs nothing,
because a barrier separates them anyway. For *independent* dispatches it destroys exactly the
overlap being measured, and then reports the serialised result as if it were the truth.

The profiled per-dispatch breakdown made this visible once looked at directly: eight partitions
of 0.788 ms each, summing to 6.30 ms, with no overlap at all.

**Rule.** Timestamp profiling measures a dispatch correctly only when that dispatch is
serialised with respect to its neighbours anyway. To measure a group of concurrent dispatches,
either disable profiling and take wall-clock across the whole submit, or place one timestamp
pair around the entire group rather than around each member.

This does not invalidate any earlier measurement in the project: every prior benchmark was a
single dispatch or a dependent chain. It does mean **`bench/gpu_bench.py` will under-report any
future multi-dispatch operation**, and the profiler should grow a group-scoped timing mode
before split-K is enabled by default.

Recorded here rather than fixed in M3-03, because changing `Recorder` is outside that stage's
scope and needs its own validation.

---

## 5g. A PREDICTIVE register model (M3-R2)

Sections 5d-5f were descriptive: they explained results after the fact. This one predicts.

A systematic sweep of `(RM, RN, STACK_LEVELS)` -- with `STACK_LEVELS` varied through K, so the
carry stack is the only thing moving -- gives, for every spill-free configuration:

```
block  L=4        L=6        L=8        L=10       L=12       L=16
2x2    33         41         49         57         65         81
4x2    57         73         89         spill      spill      spill
4x4    99         spill      spill      spill      spill      spill
```

### The law

```
VGPR = RM * RN * STACK_LEVELS + C(RM, RN)        C(2,2)=17  C(4,2)=25  C(4,4)=35
```

**The slope is exactly 1.000**, over six independent points for 2×2 alone and ten across the
three blocks, with zero residual. Each carry-stack float costs precisely one VGPR: the compiler
keeps the entire private array live, with no packing, no reuse, no rematerialisation.

That is the mechanism behind `GAP_ANALYSIS.md` §1.1. The carry stack's register cost is not
*approximately* `RM·RN·L` -- it is exactly that, and the model now says so quantitatively.

### The occupancy step function

Measured, this driver, this shader family:

```
VGPR <= 33    -> 20 waves/SIMD      (the RDNA1 wave32 maximum)
VGPR 41-57    -> 16
VGPR 65-81    -> 12
VGPR 89-99    ->  8
VGPR >~ 100   -> the allocator CAPS registers and spills to scratch
```

Stated as a measured step function rather than a derived formula. Several plausible
closed forms (waves = floor(budget / granule)) were tried against the data and **none fits all
eight points**, so no formula is claimed. The table is over-determined and predictive; a
formula would be a guess dressed as a law.

### The spill cliff, and the trap it sets

The allocator will allocate up to 99 VGPRs but not ~105. Past that it caps registers and
spills — and this is where `max_waves` becomes actively dangerous:

```
4x4, K=16384, no split-K:   VGPR=35  waves=16  scratch=40960   <- looks GOOD
4x4, K=16384, split-K:      VGPR=99  waves= 8  scratch=    0   <- looks WORSE
```

The spilling configuration reports **twice the occupancy** of the healthy one. Any tuner
ranking candidates by `max_waves` would choose the one writing 40 KiB to scratch.

> **Rule.** Reject non-zero `scratch_bytes` and `spilled_vgprs` *before* reading occupancy.
> Never rank on `max_waves` alone. Pinned by
> `test_high_occupancy_can_mean_spilling_not_efficiency`.

### Predictive test passed

M3-02 listed "does a shorter carry stack reopen the 4×4 register block?" as an open question
and declined to assume an answer. The model above answers it *before measuring*: at `L=4`,
4×4 needs `64 + 35 = 99` VGPRs, just under the cliff. Measurement:

```
4x4, K=16384, unsplit:   L=10  VGPR=35  waves=16  scratch=40960
4x4, K=16384, split-K:   L= 4  VGPR=99  waves= 8  scratch=    0
```

Exact. **This is the first quantitative prediction of compiler behaviour this project has made
and confirmed**, and it is the standard the model should be held to from here.

The block is still not worth using -- 0.884 ms against 2×2's 0.763 ms, because occupancy falls
to 8 waves. So 4×4 loses for **two independent reasons**, and removing the spill merely exposes
the second. Stage 8 saw only the first.

---

## 5h. The register laws, corrected (M3-R3)

M3-R2 proposed a register model and classified "spill cliff at ~100 VGPRs" as *strong
evidence*. **That was wrong.** Two new geometries designed to discriminate -- `2x8` and `8x2`,
identical `RM*RN` and identical carry stacks -- refuted it and produced a better law.

All figures below are at `lds_vec = 0`, forced with `VKML_GEMM_NOLDSVEC=1`, because
`lds_vec` is enabled only when `RN == 2` and would otherwise be a second variable. It is worth
+1 VGPR (M3-R2's numbers were measured with it on for two of three geometries -- a confound
found and removed in M3-R3).

### Law 1 -- register cost of the carry stack. **EXACT, 6 geometries.**

```
VGPR = RM*RN*STACK_LEVELS + C(RM, RN)          slope exactly 1.000
```

Every spill-free point, every geometry, zero residual. One carry-stack float, one VGPR.

### Law 2 -- what spilling removes. **EXACT, 5 geometries.**

```
when the stack spills:   VGPR == C(RM, RN)
```

The measured VGPR of a spilled kernel equals its Law-1 constant exactly -- 26, 23, 35, 35, 42.
The entire stack moves to scratch and *nothing else changes*. This is strong evidence that the
decomposition is structural rather than a curve fit: the two terms can be separated physically.

### Law 3 -- scratch volume. **EXACT, 15 points.**

```
scratch_bytes = stack_floats * 256              (at 256 invocations)
```

### Law 4 -- the spill threshold is the PRIVATE ARRAY SIZE, not total register pressure

```
max spill-free stack:  64 floats
min spilled stack:     80 floats        -> perfect separation, 27 configurations
```

The decisive pair:

```
8x2, L=4 :  stack 64 floats,  VGPR 106  ->  NO SPILL
4x2, L=10:  stack 80 floats,  VGPR 106  ->  SPILLS 20480 B
```

**Identical total register pressure, opposite outcomes.** A VGPR threshold cannot express this.
The allocator keeps a private array in registers up to 64 floats and spills the whole thing
beyond that, whatever the total pressure. This also explains why `2x2` *never* spills at any K:
`RM*RN = 4`, so its stack maxes out at `4 x 16 = 64` -- exactly at the limit.

The exact boundary lies in `(64, 80]`. It cannot be narrowed with the current knobs, because
`stack = RM*RN*L` with `RM*RN` in {4,8,16} and `L` bucketed to {4,6,8,10,12,16} reaches no
value between them. 64 floats = 256 bytes is a suspiciously round number but that is a
**hypothesis**, not a measurement.

### Why M3-R2 got this wrong

Every geometry sampled before M3-R3 had array size and total VGPR moving together, so the two
explanations were indistinguishable -- the exact failure mode §5e warns about: *when two
independent calculations agree, that is a warning the experiment cannot separate them.* The
warning was already written down, and the trap was still walked into. The fix was to
**design a configuration where the two disagree**, which is what `2x8` / `8x2` are for.

### C(RM,RN) is asymmetric

```
C(2,2)=18   C(4,2)=26   C(2,4)=23   C(4,4)=35   C(2,8)=35   C(8,2)=42
```

`C(4,2) != C(2,4)` and `C(2,8) != C(8,2)` -- by 3 and 7 VGPRs. **`RM` costs materially more
than `RN`.** The mechanism is visible in the shader: A is read with stride `BK`
(`As[(ty*RM + i)*BK + kk]`), so each of `RM` rows needs its own address register, while B is
contiguous and needs one base plus an index.

A linear form `C = a*RM + b*RN + RM*RN + g` fits the first four points exactly with
`a=2, b=0.5, g=9`, then **fails on the next two** (predicted C(2,8)=33 against 35, C(8,2)=42
against 42). So the asymmetry is **proven** and its direction is **explained**, but no closed
form for `C` is established. Stated as an open question rather than a law.

### Prediction scorecard

| Prediction | Predicted | Measured | Outcome |
|---|---|---|---|
| Slope = 1.000 on unseen geometry | 1.000 | 1.000 | correct |
| `scratch = stack x 256` | exact | exact | correct |
| 2x4 wave sequence | 16, 12, 8 | 16, 12, 8 | correct |
| 2x4 spills at L=10 | spill | spill | correct |
| C(8,2) | 42 | 42 | correct |
| 2x8 no spill at L=4 | no spill | no spill | correct |
| C(2,4), structural | 25 | 23 | **wrong, -2** |
| C(2,4), symmetric | 26 | 23 | **wrong, -3** |
| C(2,8) | 33 | 35 | **wrong, -2** |
| 8x2 spills at L=4 | spill | **no spill** | **wrong -- produced Law 4** |
| Instruction count within 5 % of 4x2 | +-5 % | +17 % | **wrong** |

Six correct, five wrong. The failures were the valuable half: the `8x2` miss replaced a false
VGPR-threshold law with a true array-size law, and the `C` misses established asymmetry.

---

## 6. Why resource counters are better regression signals than timings

**Timings on this machine are noisy. Resource counters are exact.**

```
1024³ GEMM wall time   2.4 – 3.4 ms across runs   (~13 % spread)
1024³ GEMM VGPR count  41                          (identical every run)
```

A 15 % regression threshold on timings produces false warnings on sub-0.1 ms
kernels, whose relative variance exceeds it. The same threshold on VGPR count is
meaningless, because **any** change in VGPR count is a real change in the
compiled kernel — there is no noise floor.

So the regression suite treats them differently:

- **Resource fields** (`vgprs`, `sgprs`, `lds_bytes`, `instructions`,
  `scratch_bytes`) — warn on *any* change; occupancy warns only on a decrease.
- **Timings** — warn beyond a percentage threshold, with kernel time and wall
  time compared separately so a transfer-path change is never mistaken for a
  kernel change.
- **Missing statistics** — never a failure. A driver without
  `pipeline_executable_properties` reports `available = false`, which means
  "unknown", not "zero".

Stage 5's regression — an optimisation that made the kernel *slower* — was a
register allocation bug. It would have been caught at pipeline-compile time by a
VGPR check, instead of by benchmarking and then a separate investigation.

---

## 7. Interpreting benchmark variance

- **Report the minimum**, with mean and standard deviation alongside. For a
  deterministic single-threaded workload the minimum is the sample least
  polluted by scheduler noise; the mean measures the machine's background load.
- **A large standard deviation invalidates the comparison, not just the number.**
  When Stage 5.75's `tiled` baseline showed sd = 0.318 ms on a 2.445 ms mean, no
  claim about a 1.0× difference against it was defensible, and none was made.
- **Warm up untimed.** Pipeline creation, first-touch page faults and GPU clock
  ramping all belong to setup.
- **Beware clock ramping.** `relu 1024×1024` measured *slower* than
  `relu 2048×2048` despite 4× less data — most likely because the larger
  workload keeps the GPU clocked up. Recorded as an open anomaly rather than
  explained away.

---

## 8. Applying this

Before optimising, establish which resource is binding:

1. Compute AI from the tile geometry. Below ~20 FLOP/byte, expect
   bandwidth-bound.
2. Read `waves_per_simd` from the pipeline statistics. Below the device ceiling
   (20 here), occupancy is limiting and register pressure is the likely cause.
3. Check `scratch_bytes` and `spilled_vgprs`. Non-zero means the register
   allocator has given up; nothing else matters until that is fixed.
4. Only then measure with timestamps, and only compare against a baseline whose
   variance is small enough to support the claim.

The recurring failure mode across M1 and M2 has not been bad implementations.
It has been **good measurements of the wrong quantity**: wall clock instead of
GPU time, relative error instead of backward error, ULP where values cross zero,
bit-equality between two different algorithms. Establish what a number measures
before trusting it.
