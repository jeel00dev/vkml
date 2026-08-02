# ADR 0011 — Geometry as specialisation constants, and the pipeline variants it buys

**Status:** accepted, implemented and measured.
**Date:** 2026-08-02
**Covers:** `shaders/im2col.comp`, `shaders/col2im.comp`, `shaders/max_pool2d.comp`, and the
general question the measurement raised: *when is a shader's cost its addressing rather than
its memory?*
**Hardware:** AMD RX 5600M (RDNA1), RADV.

---

## 1. The measurement

After `docs/adr/0010`, a CIFAR-100 step's profile put `im2col` at 10.7% and `col2im` at
5.2% — together larger than every kernel but `matmul`. Both are **pure data movement**: no
arithmetic, one load and one store per element. There is no reason for them to be slow.

Comparing each against its own bandwidth roof said they were, badly. The card peaks around
288 GB/s:

```
                        ms      GB/s   % of peak
  im2col  3ch @32     0.265     29.7      10.3
  im2col 32ch @16     0.490     42.8      14.9
  im2col 64ch @8      0.295     35.5      12.3
  col2im  3ch @32     0.132     59.6      20.7
```

**The discriminating measurement.** Percentage of peak says a kernel is slow; it does not say
*why*. `relu` over a tensor of im2col's output shape moves the same bytes with one flat index
and no addressing at all:

```
                 im2col GB/s    relu GB/s    ratio
  3ch @32x32         29.0          227.7      4.4x
  32ch @16x16        43.7          234.9      3.0x
  64ch @8x8          39.5          233.2      3.3x
```

`relu` reaches 79–82% of peak on the identical traffic. **The memory system is fine; the
addressing is the cost.**

### Why

Per output element, `im2col` performed **nine integer divisions by runtime values** — five
decoding the index into `(batch, row, position)` and then `(channel, ki, kj)`, and four more
inside `operand_offset` walking the source's strides. `col2im` was worse: four to decode the
index, then a stride test per kernel position — `top % stride_h`, `top / stride_h` and the
same across — which for a 3×3 kernel is **thirty-six more**.

A GPU has no integer divide instruction. Each becomes a multi-instruction sequence. A kernel
moving four bytes per thread can easily spend more time working out *which* four than fetching
them, and these did.

---

## 2. Two candidate fixes, and why the smaller one won

### A — magic-number division (Granlund & Montgomery)

Compute `(magic, shift)` per divisor on the host, and replace each division with a multiply
and a shift in the shader.

| | |
|---|---|
| **Benefit** | Works for any divisor, no pipeline variants, generalises to `operand_offset` and therefore to every strided kernel |
| **Cost** | A correctness-critical numeric derivation, 12 bytes of push constant per divisor against a 128-byte guaranteed budget, and code that is harder to read than `a / b` |
| **Worthwhile when** | The divisors genuinely vary per dispatch and pipeline variants would be unbounded |
| **Not worthwhile when** | They are fixed by a layer's geometry, which is exactly this case |

The derivation was checked before it was written off — the naive form (`shift = 32`,
`m = ceil(2^32/d)`) is **wrong**, failing at `d = 5, n = 2^30`, while the classic form
(`shift = 32 + ceil(log2 d)`) holds across the whole range. That is recorded because it is the
kind of thing that looks obviously right and is not.

### B — specialisation constants

Compile the divisors into the pipeline. The compiler strength-reduces each into a multiply
and shift *and* does better: it folds `patch_size = kernel_h * kernel_w`, unrolls loops whose
bounds are now constants, and where `stride == 1` it deletes the `x % stride` test entirely,
because `x % 1 == 0` is a tautology it can finally see.

| | |
|---|---|
| **Benefit** | Strictly better code than A, and a much smaller change — no push-constant pressure, no numeric derivation, the shader still reads as `a / b` |
| **Cost** | **A pipeline per distinct geometry**, and pipelines are never evicted |
| **Worthwhile when** | The geometry set is small and bounded — a model has a fixed number of layers |
| **Not worthwhile when** | Shapes are unbounded and unrepeated, where it becomes a slow leak |

**B, and the cost is measured rather than estimated.** The whole CIFAR CNN produces **three**
`im2col` variants — one per convolution layer, exactly as expected — and compiling one takes
**2 ms**. The adversarial case, 32 distinct input sizes in one process, cost 31 pipelines and
0.07 s in total.

> **The residual risk, stated:** an inference server accepting arbitrary input sizes would
> accumulate a pipeline per size, and nothing evicts them. At 2 ms and a few KB each that is
> slow rather than fatal, and it is the same shape as the exposure `gemm`'s tile variants
> already carry. If it ever matters the answer is an eviction policy on `PipelineCache`, which
> is a change to one class and not to these kernels.

**Only values the shader DIVIDES BY or ITERATES OVER are specialised.** Adding one it merely
multiplies by would cost a variant and buy nothing, so the two kernels' lists differ — they
are different shaders with different arithmetic. `SRC_CONTIGUOUS` is on the same footing: when
the source can be walked with a flat index, the four-axis stride decode is the identity and
the compiler removes it.

---

## 3. Results

Kernel throughput, minimum of 40 warm runs, submit window (rule 3), validation off (rule 5):

```
                  before            after           % of peak
  im2col  3ch @32   29.7    ->      62.7   (2.1x)   10.3 -> 21.8
  im2col 32ch @16   42.8    ->      88.1   (2.1x)   14.9 -> 30.6
  im2col 64ch @8    35.5    ->      65.0   (1.8x)   12.3 -> 22.6
  col2im  3ch @32   59.6    ->     179.6   (3.0x)   20.7 -> 62.3
  col2im 32ch @16   66.5    ->     236.3   (3.6x)   23.1 -> 82.0
  col2im 64ch @8    62.1    ->     214.2   (3.4x)   21.6 -> 74.4
```

**col2im is now at 62–82% of peak**, which is where `relu` sits — it has become
memory-bound, and there is nothing further to win from its addressing. im2col is at 22–31%
and has not.

End to end, a CIFAR-100 step, best of 5 rounds of 20 steps:

```
                     before    after
  im2col, share       10.7%     4.5%
  col2im, share        5.2%     1.6%
  GPU busy          128.4 ms  112.1 ms     per 20 steps
  step wall           8.87 ms   8.07 ms    -9.0%
```

---

## 4. What this changes about correctness, and how it is checked

Nothing about the arithmetic: both kernels gather and add in a fixed order, both are
`Kind.EXACT` in `tests/python/tolerance.py`, and both still are. What changes is that **the
geometry now reaches the shader through a second channel**, and two new failure modes come
with it:

1. **A constant wired to the wrong field.** Every one is an `int`, so nothing distinguishes
   `out_h` from `out_w` or `kernel_h` from `kernel_w`. A square image with a square kernel
   passes against a transposed pair.
2. **A pipeline cache that does not key on the constants**, which would hand the second
   distinct geometry in a process the pipeline compiled for the first — silently, and
   invisibly to any test run one geometry at a time.

Both are addressed by **asymmetry**: `test_im2col_geometry_is_wired_to_the_right_constants`
and its `col2im` twin use geometries where every extent differs from every other, so
transposing any pair changes the answer.
`test_distinct_geometries_do_not_share_a_pipeline` runs several interleaved, twice, and
corroborates on the variant count.

Verified by breaking four things — `OUT_W` fed `out_h`, `kernel_h`/`kernel_w` transposed,
`stride_h`/`stride_w` transposed, `image_h`/`image_w` transposed — each of which turns
multiple tests red. A fifth, claiming `SRC_CONTIGUOUS` unconditionally, is caught by the
existing strided-input tests in `test_layout_and_scale.py`.

---

## 4b. Applied a second time, to `max_pool2d` — and a regression that was not one

Re-profiling after §3 put `max_pool2d_backward` at 9.6% of a step, second only to `matmul`
and entirely unexamined. It is the same shape of problem: the adjoint runs a stride test per
kernel position with an argmax scan inside it, so a 2×2 pool performed roughly twenty integer
divisions per input element.

```
                        before    after     % of peak
  backward 32ch @32      51.4  ->  77.0     17.8 -> 26.7
  backward 64ch @16      30.1  ->  46.7     10.5 -> 16.2
  backward 128ch @8      24.8  ->  37.3      8.6 -> 12.9
```

End to end, `max_pool2d_backward` went **9.6% → 4.5%** of a step and GPU busy 112.1 → 105.2 ms.

`BACKWARD` and `CONTIGUOUS` were already specialisation constants here, so this adds variants
along axes the cache already split on.

### The forward looked like a 12% regression, and was not

The first measurement showed the forward at 128ch @8×8 going from 103.7 to 83 GB/s. Two
things stopped that being accepted:

- **The driver's own view.** `vulkan_pipeline_stats` reports 10 VGPRs forward and 14
  backward, **no spilling and no scratch** — so the unrolling did not cost occupancy, which
  was the obvious hypothesis. An independent producer contradicting the guess is exactly what
  §5 of `OBSERVABILITY-ARCHITECTURE.md` asks for.
- **The control.** `relu`, which did not change at all, varied by **32%** between runs on that
  same 25-microsecond dispatch. Rule 1 says an effect of that size needs repeated independent
  trials; here the noise was larger than the effect.

Measured properly — shapes large enough that the kernel dominates its own timestamp bracket,
minimum of 120 warm runs, A/B against the frozen baseline restored with `git stash` (rule 7):

```
                        before     after
  128ch @8x8            0.0350  ->  0.0338    faster
   64ch @16x16          0.0528  ->  0.0520    flat
   32ch @32x32          0.0773  ->  0.0567    1.36x faster
   32ch @64x64          0.2117  ->  0.2127    flat, 197 GB/s
   64ch @64x64          0.4076  ->  0.4082    flat, 206 GB/s
```

**There is no regression.** The forward is faster or equal at every size, and flat at the
large ones because it is already at ~200 GB/s — bandwidth-bound, with nothing left to win
from addressing. The 12% was noise on a measurement the constitution's own rules say cannot
resolve 12%.

Recorded because the near-miss is the useful part: a regression was very nearly accepted and
documented as a trade, on a number that did not exist.

---

## 5. Consequences

- **`im2col` is still 3.6× off its roof** while `col2im` has reached it. The remaining cost
  is the gather's access pattern, not its addressing: consecutive threads read consecutive
  *output* positions, which map to consecutive input columns only within a row. That is the
  next thing to measure there, and it is a different problem from this one.
- **Implicit GEMM would delete the kernel rather than speed it up.** `EXTENSIBILITY-ROADMAP`
  §4a P2 proposes fusing the im2col addressing into the GEMM's operand load so the expansion
  is never written. This ADR makes the expansion cheaper; that would make it free. The two do
  not conflict — this is the version that exists.
- **The technique generalises and is being generalised in measured order.** Applied to
  `im2col`/`col2im` (§3) and then to `max_pool2d` (§4b), in the order the profile named them.
  `reduce`, `binary`, `unary` and the rest all call `operand_offset` and all would benefit
  from a `SRC_CONTIGUOUS` fast path — but each is a pipeline-variant trade to be justified
  where it is measured, not a blanket change to every kernel.
- **Two kernels are now at the bandwidth roof and one is not.** `col2im` reaches 62–82% of
  peak and `max_pool2d` forward ~200 GB/s; both are memory-bound and done. `im2col` is at
  22–31% and its residual is the gather's access pattern, which is a different problem from
  the one this ADR solves.
