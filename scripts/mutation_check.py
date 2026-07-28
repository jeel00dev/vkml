#!/usr/bin/env python3
"""Mutation campaign: break each kernel, confirm the suite notices.

docs/MEASUREMENT-AUDIT.md rule 10 says to check every gate for vacuity before
trusting a pass. A green suite proves the tests ran, not that they can fail --
and a test that cannot fail is worse than none, because it manufactures
confidence. This is how that rule is checked rather than asserted.

Each entry applies one semantically MEANINGFUL mutation to a kernel -- an
off-by-one, a dropped guard, a reversed fold -- rebuilds, runs the tests that
should catch it, and reports:

    KILLED    the suite failed. The tests detect this defect.
    SURVIVED  the suite passed. Something is untested; investigate.

A syntax error would prove nothing, so every mutation compiles.

MAINTENANCE COST, stated plainly: each mutation is a literal string from the
source, so a refactor silently stops it applying. That failure is reported as
PATTERN-MISSING rather than passing quietly, but it does mean this file has to
be updated alongside the kernels it targets. Adding a kernel without adding a
mutation here leaves a gap this script will not notice.

Usage:  python scripts/mutation_check.py [path-substring]
Exit:   0 if every mutation was killed, 1 otherwise.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/jeel/Projects/vkml")
PY = ROOT / ".venv/bin/python"

# (label, file, find, replace, test selector)
# Each mutation changes MEANING, not syntax -- a compile error would prove
# nothing about the tests.
MUTATIONS = [
    # --- shaders -----------------------------------------------------------
    ("tri: diagonal off-by-one", "shaders/tri.comp",
     "offset >= p.diagonal : offset <= p.diagonal",
     "offset > p.diagonal : offset <= p.diagonal", "test_tri"),

    ("cat: reuse output extent for the source index", "shaders/cat.comp",
     "const uint extent = from_a ? p.a_extent : p.b_extent;",
     "const uint extent = p.out_extent;", "test_cat"),

    ("index_select: ignore the index vector", "shaders/index_select.comp",
     "const uint j = uint(clamp(raw, int64_t(0), int64_t(p.src_extent) - 1));",
     "const uint j = k % p.src_extent;", "test_index_select"),

    ("scatter_add: reverse the fold order", "shaders/scatter_add.comp",
     "for (uint k = 0; k < p.src_extent; ++k) {",
     "for (uint k = p.src_extent; k-- > 0;) {", "test_scatter_add"),

    ("im2col: drop the padding bounds check", "shaders/im2col.comp",
     "if (h >= 0 && h < p.image_h && x >= 0 && x < p.image_w) {",
     "if (h >= 0 && h < p.image_h && x >= 0 && x < p.image_w || true) {", "test_im2col"),

    ("col2im: drop the stride-boundary test", "shaders/col2im.comp",
     "if (top < 0 || (top % p.stride_h) != 0) { continue; }",
     "if (top < 0) { continue; }", "test_col2im"),

    ("max_pool2d: tie rule picks the last maximum", "shaders/max_pool2d.comp",
     "if (v > best) {", "if (v >= best) {", "test_max_pool2d or test_conv2d"),

    ("rand: nine Philox rounds instead of ten", "shaders/rand.comp",
     "const int  kRounds = 10;", "const int  kRounds = 9;", "test_rand or test_dropout"),

    # --- CPU oracle --------------------------------------------------------
    ("philox(cpu): nine rounds instead of ten", "src/backend/cpu/philox.h",
     "inline constexpr int kPhiloxRounds = 10;",
     "inline constexpr int kPhiloxRounds = 9;", "test_rand or test_dropout"),

    ("k_col2im: drop the stride-boundary test", "src/backend/cpu/kernels_movement.cpp",
     "if (top < 0 || top % p.stride_h != 0) {\n                continue;\n            }",
     "if (top < 0) {\n                continue;\n            }", "test_col2im"),

    ("k_max_pool2d: tie rule picks the last maximum", "src/backend/cpu/kernels_movement.cpp",
     "            if (v > best) {", "            if (v >= best) {",
     "test_max_pool2d or test_conv2d"),

    ("k_cat: reuse output extent for the source index", "src/backend/cpu/kernels_movement.cpp",
     "const int64_t extent = from_a ? a_extent : b_extent;",
     "const int64_t extent = out_extent;", "test_cat"),
]


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)


def build() -> bool:
    r = run(["cmake", "--build", "build/release", "-j8"])
    return r.returncode == 0


def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    results = []

    for label, rel, find, replace, selector in MUTATIONS:
        if only and only not in rel:
            continue
        path = ROOT / rel
        original = path.read_text()

        if find not in original:
            results.append((label, "PATTERN-MISSING"))
            print(f"  !! {label}: pattern not found -- mutation not applied", flush=True)
            continue

        try:
            path.write_text(original.replace(find, replace, 1))
            if not build():
                results.append((label, "BUILD-FAILED"))
                print(f"  !! {label}: did not compile", flush=True)
                continue

            r = run([str(PY), "-m", "pytest", "tests/python", "-x", "-q", "-k", selector])
            killed = r.returncode != 0
            results.append((label, "KILLED" if killed else "SURVIVED"))
            mark = "ok " if killed else "!! "
            print(f"  {mark}{label}: {'KILLED' if killed else 'SURVIVED'}", flush=True)
        finally:
            path.write_text(original)

    build()  # leave the tree as we found it

    print()
    survived = [r for r in results if r[1] != "KILLED"]
    print(f"{len(results) - len(survived)}/{len(results)} mutations killed")
    for label, status in survived:
        print(f"  SURVIVING: {label} [{status}]")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
