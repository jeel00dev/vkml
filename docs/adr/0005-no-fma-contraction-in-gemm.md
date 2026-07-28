# ADR 0005 — Products are rounded: no FMA contraction in the GEMM accumulators

**Status:** accepted
**Date:** 2026-07-29
**Covers:** why `precise` appears on five shader accumulators, and what may not be
changed without re-deriving the bound in THEORY.md

---

## Context

`matmul` of a matrix of alternating `+1e7` and `-1e7` has an exact answer of zero. The
CPU backend and RADV both return exactly `0.0`. MoltenVK returned `1130496.0`.

This surfaced the first time the suite ran on a third driver, on hardware nobody working
on the project owns. It is the case the cross-driver CI job exists to find.

## Measurements

Everything below was computed or read, not estimated.

`1e7` is exactly representable in fp32. `1e14` is not:

```
fl(1e14) - 1e14                     = 376832        (call it delta)
MoltenVK residue where exact = 0    = 1130496 = 3 x delta   exactly
one ULP at 1e14                     = 8388608      (so the residue is sub-ULP)
```

Simulating both arithmetics against the shader's real structure — 32-element sequential
blocks folded by the pairwise carry stack, `K=96` giving three blocks — reproduces both
observed results bit-exactly:

```
separate mul + add   blocks=[0, 0, 0]                  total=0          (CPU, RADV, lavapipe)
FMA contracted       blocks=[376832, 376832, 376832]   total=1130496    (MoltenVK)
```

Without contraction every product rounds to `fl(1e14) = 1e14 + delta`; the alternating
signs cancel the deltas along with the values, so each block is exactly zero. With
contraction the product is never rounded, so one delta survives per block. **The factor
of three is `K / 32`, not a coincidence.**

A memory or bounds error cannot produce an exact integer multiple of a rounding error,
which is what rules out the other explanation for a GPU-only divergence.

## Why this is a contract violation, not a strict test

`gemm_naive.comp` documents the K-loop error bound as `(B + log2(n/B)) * eps` for the
pairwise fold. **That derivation assumes each product is rounded.** A driver that
contracts changes the per-product rounding the bound is built on, so the bound is not
merely pessimistic there — it does not describe the kernel that ran.

Determinism is part of correctness in this project (P1). Fixed pairwise trees, no global
atomics, and a split-K partitioning proven bit-identical to the unsplit kernel were all
paid for to get reproducible results. Contraction reintroduces exactly the cross-driver
variability those choices removed, and does it invisibly.

## Decision

The GEMM family declares its product accumulators `precise`, which emits SPIR-V
`NoContraction` on the multiply-add. Five accumulators: `gemm_naive`, `gemm_tiled`,
`gemm_reg`, `gemm_db`, `gemv`.

**Per-product rounding is now part of the numerical contract.** A kernel that accumulates
`a * b` into a running sum must not permit contraction unless THEORY.md's bound is
re-derived for the fused form.

## Consequences

**Cost on RADV: none, measured.** Pipeline statistics are identical with and without the
decoration — `vgpr=41 sgpr=35 instr=1124 scratch=0 lds=8192`. The compiled machine code
does not change, which independently confirms RADV was never contracting, consistent with
it returning exactly zero.

Statistics rather than timings because the machine's noise floor swamped the effect: the
recorded `rx5600m` baseline reported `+472%` on `relu 1024x1024` *with* this change and
`+485%` *without* it, on kernels the change does not touch. That baseline is stale and
could not answer the question; the reverted control run is what established this.

**Cost on Metal: real, and not yet measured.** Forbidding contraction there replaces one
`fma` with a separate multiply and add in the hot loop. MoltenVK went from 4 failures to
1 with this change, so the remedy works; its price is unknown.

Verified unchanged elsewhere: 1181 Python tests, 96 doctest cases, and the lavapipe job,
which passes with the decoration in place — a third driver confirming the change is inert
where contraction was not happening.

## Revisit trigger

Measure GEMM throughput on Apple hardware. If contraction is worth a material fraction of
Metal GEMM performance, the options are a per-driver decision or a re-derived bound for
the fused form — both of which need that number first, and neither of which may quietly
drop the guarantee. Until then this decision is binding.
