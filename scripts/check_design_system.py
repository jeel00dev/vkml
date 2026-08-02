#!/usr/bin/env python3
"""The documentation stylesheet must stay a system, not an accumulation.

WHY THIS EXISTS. Left alone, a stylesheet does not drift in any one place; it
accretes. Each component picks a size that looks right next to the thing beside
it, nobody is wrong, and the whole slowly stops agreeing with itself. Measured
before this check existed, vkml.css had:

    39 distinct font sizes across 62 uses
    18 of them between 12.8px and 15.2px -- .84rem, .845rem, .85rem, .855rem,
       .86rem, .87rem, .875rem, .88rem, .885rem, .89rem ...
    6 font weights, two of which (550, 650) are not standard and are either
       synthesised or rounded by most families
    6 colour literals outside the palette

None of that is visible as a bug. It is visible as a site that feels slightly
loose and cannot be pointed at.

Font size is checked strictly rather than by a budget: the scale covers every
case in use, so a literal is a deviation by definition, and a budget with
headroom would wave through the first new value -- which is precisely the step
that starts the accumulation. The original 39 arrived one defensible value at a
time.
"""
from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSS = ROOT / "web" / "theme" / "vkml.css"

# ANY absolute font size outside the scale is a deviation, so this is not a
# count threshold. A budget with headroom lets the first new size through
# silently, which is exactly the step that starts the accumulation -- the
# original 39 arrived one defensible value at a time.
ALLOWED_WEIGHTS = {"400", "500", "600", "700"}
# Sizes relative to their PARENT are a different thing from a scale step, and
# are correct: inline code is .875em of whatever contains it, so it tracks the
# heading or paragraph it sits in rather than fixing itself.
RELATIVE_OK = True


def strip_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/", "", text, flags=re.S)


def root_spans(css: str):
    return [m.span() for m in re.finditer(r":root[^{]*\{[^}]*\}", css, re.S)]


def main() -> int:
    if not CSS.exists():
        print(f"missing {CSS}")
        return 1
    css = strip_comments(CSS.read_text())
    spans = root_spans(css)
    outside = lambda i: not any(a <= i < b for a, b in spans)
    problems: list[str] = []

    # --- font sizes ------------------------------------------------------
    literal = collections.Counter()
    for m in re.finditer(r"font-size:\s*([\d.]+(?:rem|em|px))", css):
        v = m.group(1)
        if not (RELATIVE_OK and v.endswith("em") and not v.endswith("rem")):
            literal[v] += 1
    for m in re.finditer(r"(?<![-\w])font:(?:\s*\d{3})?\s*([\d.]+(?:rem|em|px))", css):
        v = m.group(1)
        if not (RELATIVE_OK and v.endswith("em") and not v.endswith("rem")):
            literal[v] += 1
    tokens = set(re.findall(r"var\(--fs-\d+\)", css))
    total = len(literal) + len(tokens)
    if literal:
        problems.append(
            f"{len(literal)} absolute font size(s) not on the scale: "
            f"{', '.join(f'{v} x{n}' for v, n in sorted(literal.items()))}. "
            f"Use one of the --fs-* steps, or add a step deliberately.")

    # --- weights ---------------------------------------------------------
    weights = set(re.findall(r"font-weight:\s*(\d+)", css))
    weights |= set(re.findall(r"(?<![-\w])font:\s*(\d{3})\s", css))
    stray = weights - ALLOWED_WEIGHTS
    if stray:
        problems.append(
            f"non-standard font weights {sorted(stray)}; use "
            f"{sorted(ALLOWED_WEIGHTS)} or the --fw-* tokens. Most families do "
            f"not ship 550 or 650 and will synthesise or round them.")

    # --- colour ----------------------------------------------------------
    hexes = [m.group(0) for m in re.finditer(r"#[0-9a-fA-F]{3,8}\b", css)
             if outside(m.start())]
    if hexes:
        problems.append(
            f"{len(hexes)} colour literal(s) outside the :root palette "
            f"({', '.join(sorted(set(hexes)))}). A colour that is not a token "
            f"is a colour a theme cannot restate.")
    rgba = [m.group(0) for m in re.finditer(r"rgba?\([^)]*\)", css)
            if outside(m.start())]
    if rgba:
        problems.append(f"{len(rgba)} rgb()/rgba() literal(s) outside the palette")

    # --- tokens nobody spends -------------------------------------------
    #
    # The same defect as an unused rule, one level down, and it arrives the same
    # way: a token outlives the thing it coloured and nothing says so. `--t-ty`
    # coloured a `tok-t` span the syntax highlighter has never emitted -- its
    # token groups are c/s/n/f/w and there is no `t` -- so the value sat in both
    # themes being maintained by whoever edited the palette next.
    # SCALES ARE EXEMPT, and the distinction is the point rather than a
    # loophole. `--fs-*`, `--s-*` and `--fw-*` are the steps of a declared
    # scale: they exist to be CHOSEN FROM, a complete scale with a gap in it is
    # worse than one with an unused step, and deleting `--s-1` only guarantees
    # that whoever next needs the smallest spacing invents a different value
    # for it. A token naming one specific thing has no such defence.
    scale = re.compile(r"^--(fs|s|fw)-")
    declared = {m.group(1) for m in re.finditer(r"(--[\w-]+)\s*:", css)}
    referenced = {m.group(1) for m in re.finditer(r"var\(\s*(--[\w-]+)", css)}
    orphans = sorted(t for t in declared - referenced if not scale.match(t))
    if orphans:
        problems.append(
            f"{len(orphans)} custom propert(y/ies) declared and never used: "
            f"{', '.join(orphans)}. Delete them, or the palette accumulates "
            f"values nothing spends. Scale steps (--fs-* --s-* --fw-*) are "
            f"exempt: they exist to be chosen from.")

    print(f"  {total} font sizes ({len(tokens)} tokens), "
          f"{len(weights)} weights, {len(hexes) + len(rgba)} stray colours")
    if problems:
        print(f"  {len(problems)} problem(s):")
        for p in problems:
            print(f"    {p}")
        return 1
    print("  the stylesheet is on the system")
    return 0


if __name__ == "__main__":
    sys.exit(main())
