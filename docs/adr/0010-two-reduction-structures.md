# ADR 0010 — Two reduction structures, and a fold order that is allowed to move

**Status:** accepted, implemented and measured.
**Date:** 2026-08-02
**Covers:** `shaders/reduce.comp`, the selection rule in `vulkan_backend.cpp`, and the
question the constitution requires be asked before any of it: *may the arithmetic change?*
**Hardware:** AMD RX 5600M (RDNA1), RADV. Every measurement below was run on it.

---

## 1. The measurement this exists for

Attributing a CIFAR-100 training step put `sum` at **14.9%** — second only to `matmul` — with
no obvious reason. Reductions read a tensor once and write far less; 14.9% of a step is not
what that should cost.

The shapes said why. The largest single reduction in the step is a convolution's weight
gradient: `(64, 128, 576) -> (1, 1, 128, 576)`, folding 64 samples into 73,728 values.
`reduce.comp` launches **one workgroup per output element**, so that is 73,728 workgroups of
256 lanes, each performing 64 loads and then eight barrier steps to combine them.

Holding the element count fixed at 18 MiB and sliding the split between outputs and reduction
length:

```
  elements per output       4      16      64     256    1024    4096
  workgroups          1179648  294912   73728   18432    4608    1152
  GB/s, reduced axis contiguous
                          1.6     6.4    21.6    62.3    96.2   107.5
  GB/s, reduced axis strided
                          1.4     4.2     6.6     6.7    26.7    33.2
```

**Throughput tracks the workgroup count, not the bytes.** The kernel is launch-bound below
roughly a thousand elements per output, by up to 67×. The card's peak is ~288 GB/s; the
weight-gradient case was running at 6.6.

Two independent causes, and the sweep separates them:

- **Launch.** 73,728 workgroups to move 18 MiB, with 192 of every 256 lanes idle.
- **Coalescing.** Within a workgroup, lane *l* and lane *l+1* walk the REDUCTION, so when the
  reduced axis is strided they read `n_out` floats apart — 288 KB for this shape. Every lane
  touches its own cache line.

---

## 2. The second structure

A lane-per-output kernel inverts both: one **lane** per output element, each walking its own
reduction alone. No shared memory, no barriers, and `ceil(n_out / WG)` workgroups instead of
`n_out`.

It also fixes coalescing **in exactly the case that was worst**. Reducing a strided axis
leaves the contiguous axes as the output, so adjacent lanes — adjacent `out_id` — read
adjacent addresses.

The two are mirror images, and each is catastrophic in the other's case. At 64 elements per
output:

| | wide (workgroup per output) | tall (lane per output) |
|---|---|---|
| reduced axis strided | 6.6 GB/s | **132.2 GB/s** |
| reduced axis contiguous | **21.6 GB/s** | 8.8 GB/s |

So this is not "replace the kernel". It is a second structure with a selection rule.

---

## 3. The rule, and the term that was missing

Three terms, and only the first two were obvious:

1. **Coalescing.** Use the tall kernel when the kept axes are contiguous — equivalently, when
   the reduced axis is not the innermost one.
2. **The launch floor.** Below ~32 elements per output the wide kernel spends everything on
   workgroup launch *whatever* the layout — 1.6 GB/s at 4 elements per output. The tall
   kernel wins there even against its own bad case, because the stride between adjacent lanes
   is only `n_red` floats, short enough to stay inside a cache line.
3. **Occupancy.** The tall kernel has exactly `n_out` threads, so a reduction with few outputs
   leaves the device empty however well it coalesces.

**Term 3 was found by re-running the same sweep after the rule changed, not by reasoning.**
With only the first two terms, 1,152 outputs of 4,096 elements ran at **4.3 GB/s against the
wide kernel's 33.2 — a 7.7× regression** introduced by an optimisation. The floor is one full
workgroup per compute unit, which is a statement about the hardware rather than a fitted
constant: at half of it the two tie (29.7 against 26.7) and at twice it the tall path wins by
12×, so the boundary is not sharp and does not need to be.

`shader_core_count` comes from `VK_AMD_shader_core_properties2` and is 0 elsewhere.
`MEASUREMENT-AUDIT.md` §6 requires every consumer to treat 0 as a reason to decline rather
than guess — so this declines to a conservative floor of 8 units instead of declining to
choose, because the wrong value here costs throughput on one kernel and cannot cost
correctness.

### Result across the whole sweep

```
                    reduced axis strided        reduced axis contiguous
  elts/output      before    after              before    after
       4              1.4    116.3  (83x)          1.6    132.5  (83x)
      16              4.2    149.6  (36x)          6.4     21.5  (3.4x)
      64              6.6    132.2  (20x)         21.6     21.7  (=)
     256              6.7     83.7  (12x)         62.3     61.0  (-2%)
    1024             26.7     26.6  (=)           96.2     96.6  (=)
    4096             33.2     33.3  (=)          107.5    105.3  (-2%)
```

No cell regresses beyond measurement noise.

---

## 4. The part that needed this document: the fold order moves

Both structures fold **pairwise** — the tall path reuses the wide path's carry stack, blocks
of 32 merged by binary-carry propagation — so both satisfy the `(B + log2(n/B))*eps` backward
bound that `tests/python/tolerance.py` derives for `Kind.BACKWARD`.

**They do not associate identically.** The wide kernel splits a reduction across lanes and
combines them in a shared-memory tree; the tall kernel walks it in one lane. For the same
input, the last bits differ. A shape that switches structures — because it grew, or because
it ran on a device with a different compute-unit count — produces a different float.

This is a change to the numerical contract and the constitution forbids making it silently.
The re-derivation:

- **`sum` and `mean` are `Kind.BACKWARD`**, and the policy's own note is *"pairwise
  summation"* — a bound, not an association. Both structures meet it. `tolerance.py` needed
  no change, which is the check that the policy was written about the right thing.
- **`max`, `min`, `argmax`, `argmin` are `Kind.EXACT` and stay exact.** They select rather
  than accumulate, so no reordering can move a bit. The tie-breaking rule (`strict >` with
  ascending *k*, so the FIRST extremum wins) and NaN absorption are reimplemented in the tall
  path and tested there specifically.
- **Determinism is unaffected**, and this is the distinction that matters: determinism is
  *reproducibility*, not immutability across versions. Both structures fix the assignment of
  elements to lanes and the shape of the combining tree from the shape alone, so the same
  input on the same device gives bit-identical output, run after run. What changed is which
  fixed order runs for a given shape.
- **The choice is published as a Decision** (`site = "reduce.structure"`), because it changes
  which arithmetic order executes and is not derivable from the inputs — the test in
  `OBSERVABILITY-ARCHITECTURE.md` §3. A consumer comparing two runs bit for bit can see which
  ran, and the attribution report now prints `sum:reduce_lane_per_output` as a kernel in its
  own right.

### What is NOT claimed

The tall path's pairwise carry stack is **defensive rather than load-bearing today**, and
saying so is more useful than implying coverage. Replacing it with a flat sequential
accumulator was tried as a negative control and **no test noticed** — because the rule sends
large `n_red` to the wide path, so the tall path only ever folds short reductions, where
sequential and pairwise differ by less than the tolerance's safety factor. The stack is there
so that a future change to the selection rule does not silently cross the bound. It is not
evidence that the bound is being exercised.

---

## 5. End-to-end

The number that decides whether any of this mattered — a CIFAR-100 training step, best of 5
rounds of 20 steps:

```
                       before     after
  sum, share of step    14.9%      4.7%
  GPU busy            152.1 ms   128.4 ms      per 20 steps
  step wall            10.18 ms    8.87 ms     -12.9%
```

`sum` fell from the second-largest line in the table to sixth. `matmul` is now 25.6% and
`im2col` 10.7%, which is where the next measurement should start.

---

## 6. Consequences

- A reduction's result depends on its shape and on the device's compute-unit count. Anything
  comparing floats across devices bit for bit must account for that; the tolerance policy
  already does, and the Decision makes it visible when it matters.
- `reduce.comp` now has two paths. `LANE_PER_OUTPUT` is a specialisation constant rather than
  a push constant so the compiler drops the unused half — the tall path keeps no shared
  memory and takes no barriers, and a runtime branch would leave both costs in the binary.
- **Split-K for reductions remains unbuilt and is now better scoped.** It addresses the
  opposite corner from the tall kernel's: few outputs, many elements each. The sweep shows
  that corner running at 33 GB/s, which is the worst remaining cell.
- The selection rule is three terms and each was measured. If a fourth is needed, the sweep
  in `scripts/`-adjacent form is four lines of Python and is the thing to re-run — the third
  term exists because it was re-run once.
