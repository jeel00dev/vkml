#!/usr/bin/env python3
"""Every GEMM accumulator must forbid FMA contraction.

ADR 0005 makes per-product rounding part of the numerical contract: a kernel
accumulating `a * b` into a running sum must not permit the compiler to fuse the
multiply and the add, because THEORY.md's error bound is derived for the rounded
form and a fused multiply-add rounds once instead of twice.

GLSL expresses that as `precise`, which emits SPIR-V NoContraction.

WHY A GATE. The decoration is invisible in behaviour on the hardware this was
measured on -- RADV produces byte-identical machine code with and without it,
`vgpr=41 sgpr=35 instr=1124 scratch=0 lds=8192` either way -- so a new GEMM
variant that omits it would pass every test here and violate the contract only
on a driver that does contract. That is the same shape as the f16 rounding bug
(#3) and the push-constant budget (#2): correct locally, wrong elsewhere, and
silent in between.

Checked by RULE rather than by a list, so a sixth GEMM variant is covered the
day it is added rather than the day someone remembers this ADR.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHADERS = ROOT / "shaders"

# A multiply-accumulate: `x += a * b`, or `x = x + a * b`, or fma().
MULADD = re.compile(r"\+=\s*[^;]*\*|=\s*\w+\s*\+\s*[^;]*\*|\bfma\s*\(")


def main() -> int:
    problems, checked = [], 0
    for path in sorted(SHADERS.glob("*.comp")):
        # The rule targets the GEMM family: those are the accumulators whose
        # bound THEORY.md derives. A reduction that only sums cannot contract.
        if not (path.stem.startswith("gemm") or path.stem.startswith("gemv")):
            continue

        text = path.read_text()
        body = "\n".join(ln for ln in text.split("\n") if not ln.strip().startswith("//"))
        has_muladd = bool(MULADD.search(body))
        has_precise = "precise" in body

        checked += 1
        state = ("muladd" if has_muladd else "sum-only")
        print(f"  {path.stem:22} {state:9} precise={'yes' if has_precise else 'NO'}")

        if has_muladd and not has_precise:
            problems.append(
                f"{path.name} accumulates a product but does not declare `precise`; "
                "ADR 0005 requires NoContraction on GEMM accumulators")

    print()
    print(f"  {checked} GEMM-family shaders checked")
    if problems:
        print("  FAILED:")
        for p in problems:
            print(f"    {p}")
        return 1
    print("  every product accumulator forbids contraction")
    return 0


if __name__ == "__main__":
    sys.exit(main())
