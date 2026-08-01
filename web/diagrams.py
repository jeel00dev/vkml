"""Diagrams generated from the source tree.

WHY GENERATED. A hand-drawn architecture diagram is a claim about the code that
nothing checks. It is right on the day it is drawn and slowly stops being right,
and because it is a picture nobody greps it -- so it goes stale more quietly
than prose does. Every diagram here is derived from the tree at build time, so
the question "is this still true" is answered by rebuilding.

The layer diagram is the strongest case: it is built from the SAME extraction
`scripts/check_layering.py` uses to fail CI. The picture and the gate cannot
disagree, because they read the same includes with the same parser. If the
diagram is wrong the build is broken, not the documentation.

WHY SVG rather than an image. It is text, so a change shows up in a diff; it
scales; it costs a couple of kilobytes against tens for a PNG; it can carry
<title> and <desc> for a screen reader; and inlined in the page it inherits the
theme's CSS variables, so it is correct in both themes without a second file.

WHAT IS NOT GENERATED is said so, in place, rather than left for the reader to
assume.
"""
from __future__ import annotations

import collections
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def _layer_data():
    """Layers, file counts and real include edges, from the gate's own parser."""
    import check_layering as CL

    files: collections.Counter = collections.Counter()
    edges: collections.Counter = collections.Counter()
    for path in CL.iter_sources():
        try:
            rel = path.relative_to(CL.ROOT / "include" / "vkml").as_posix()
        except ValueError:
            rel = path.relative_to(CL.ROOT / "src").as_posix()
        own = CL.layer_of(rel)
        if own is None:
            continue
        files[own] += 1
        for line in path.read_text().splitlines():
            m = CL.INCLUDE_RE.match(line)
            if not m:
                continue
            dep = CL.layer_of(m.group(1))
            if dep and dep != own:
                edges[(own, dep)] += 1
    return CL.LAYERS, files, edges


def layer_stack() -> str:
    """The dependency stack: what may include what, drawn from what does.

    THE QUESTION THIS ANSWERS: "if I add a file here, what am I allowed to use?"

    Levels run bottom to top so that "depends on" reads downward, which is the
    direction the rule is stated in and the direction every arrow must point.
    That is the whole invariant, and a reader should be able to see it without
    reading a sentence: if any arc curved upward, the build would be failing.
    """
    LAYERS, files, edges = _layer_data()

    order = sorted(LAYERS, key=lambda n: (-LAYERS[n], n))
    row_h, top, box_w, left = 42, 26, 190, 210
    height = top + row_h * len(order) + 24
    y_of = {name: top + i * row_h for i, name in enumerate(order)}

    out = [
        f'<svg class="dia" viewBox="0 0 640 {height}" role="img" '
        f'aria-labelledby="ls-t ls-d" xmlns="http://www.w3.org/2000/svg">',
        '<title id="ls-t">vkML layer dependencies</title>',
        '<desc id="ls-d">Nine layers stacked by level. Every arrow points '
        'downward: a layer may include only from a layer below it, which '
        'scripts/check_layering.py enforces on every build.</desc>',
    ]

    # The arcs first, so the boxes sit on top of them.
    for (src, dst), n in sorted(edges.items(), key=lambda kv: -kv[1]):
        if src not in y_of or dst not in y_of:
            continue
        y1, y2 = y_of[src] + 13, y_of[dst] + 13
        # Bow width scales with the distance skipped, so a long-range dependency
        # is visibly a bigger claim than one on the layer directly below.
        span = abs(LAYERS[src] - LAYERS[dst])
        bow = left + box_w + 18 + span * 22
        out.append(
            f'<path class="dia-edge" d="M{left + box_w} {y1} '
            f'C{bow} {y1} {bow} {y2} {left + box_w} {y2}" '
            f'stroke-width="{min(1 + n / 6, 3):.1f}"/>')

    for name in order:
        y = y_of[name]
        n = files.get(name, 0)
        empty = ' dia-box-empty' if n == 0 else ''
        out.append(
            f'<rect class="dia-box{empty}" x="{left}" y="{y}" '
            f'width="{box_w}" height="26" rx="4"/>'
            f'<text class="dia-lv" x="{left - 12}" y="{y + 18}" '
            f'text-anchor="end">{LAYERS[name]}</text>'
            f'<text class="dia-name" x="{left + 12}" y="{y + 18}">'
            f"{html.escape(name)}</text>"
            f'<text class="dia-meta" x="{left + box_w - 12}" y="{y + 18}" '
            f'text-anchor="end">{n or "—"}</text>')

    out.append(f'<text class="dia-cap" x="{left - 12}" y="{top - 8}" '
               f'text-anchor="end">level</text>')
    out.append(f'<text class="dia-cap" x="{left + box_w - 12}" y="{top - 8}" '
               f'text-anchor="end">files</text>')
    out.append("</svg>")
    return "\n".join(out)


def layer_facts() -> dict:
    """Numbers the prose beside the diagram can quote without retyping them."""
    LAYERS, files, edges = _layer_data()
    return {
        "layers": len(LAYERS),
        "files": sum(files.values()),
        "edges": len(edges),
        "includes": sum(edges.values()),
        "empty": sorted(n for n in LAYERS if not files.get(n)),
        "widest": max(edges.items(), key=lambda kv: kv[1]) if edges else None,
    }
