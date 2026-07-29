#!/usr/bin/env python3
"""Enforce the layering rule from docs/ARCHITECTURE.md §4.1.

    "each layer may include only from layers strictly below it"

The architecture document draws the stack as a conceptual data-flow diagram.
This script encodes the actual *dependency* DAG, which differs in one place:
`core` sits below `plan` and `dispatch` rather than above them, because
Shape/Storage/DType are foundations that those layers build on; and `autograd`
sits ABOVE `api`, because backward rules are written in terms of forward tensor
operations (that is the central design choice) while `api` never calls into
autograd. The rule the
document actually states -- and its two worked examples ("backend/vulkan must
not know what an nn.Linear is", "autograd must not know Vulkan exists") -- are
both preserved.

Layer of a header : path under include/vkml/
Layer of a source : path under src/

Exit code 0 if clean, 1 if any violation is found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Lower index == lower layer. A file may include from a strictly lower index,
# or from its own layer. Equal-index layers are siblings and may NOT include
# each other (e.g. backend/cpu must not reach into backend/vulkan).
LAYERS: dict[str, int] = {
    "util": 0,
    "core": 1,
    "graph": 2,
    "backend/api": 3,
    "backend/cpu": 4,
    "backend/vulkan": 4,
    "dispatch": 5,
    "plan": 5,
    "api": 6,
    "autograd": 7,
}

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]vkml/([^">]+)[">]')

ROOT = Path(__file__).resolve().parent.parent


def layer_of(rel: str) -> str | None:
    """Longest matching layer prefix for a path like 'backend/cpu/ops.h'."""
    best = None
    for name in LAYERS:
        if rel == name or rel.startswith(name + "/"):
            if best is None or len(name) > len(best):
                best = name
    return best


def iter_sources():
    for base, strip in ((ROOT / "include" / "vkml", None), (ROOT / "src", None)):
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix in (".h", ".hpp", ".cpp", ".cc"):
                yield path


def main() -> int:
    violations: list[str] = []
    checked = 0

    for path in iter_sources():
        # as_posix(), not str(): layer names are written with forward slashes,
        # and str() renders a WindowsPath with backslashes, so `rel` never
        # matched, `own` was always None, and every file took the continue
        # below. The script then reported "layering check passed (0 files)" and
        # exited 0 -- a gate that examined nothing and said yes (issue #11).
        try:
            rel = path.relative_to(ROOT / "include" / "vkml").as_posix()
        except ValueError:
            rel = path.relative_to(ROOT / "src").as_posix()

        own = layer_of(rel)
        if own is None:
            continue  # umbrella header or a file outside the layered tree
        checked += 1

        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            m = INCLUDE_RE.match(line)
            if not m:
                continue
            dep = layer_of(m.group(1))
            if dep is None:
                continue  # umbrella header include, always allowed
            if dep == own:
                continue
            if LAYERS[dep] >= LAYERS[own]:
                rel_path = path.relative_to(ROOT)
                violations.append(
                    f"{rel_path}:{lineno}: layer '{own}' (level {LAYERS[own]}) "
                    f"must not include from '{dep}' (level {LAYERS[dep]})\n"
                    f"    {line.strip()}"
                )

    if violations:
        print(f"layering check FAILED -- {len(violations)} violation(s):\n", file=sys.stderr)
        for v in violations:
            print(v, file=sys.stderr)
        return 1

    # A gate that examined nothing must not report success. On Windows this
    # script found all 56 sources, resolved none of them to a layer, and printed
    # "passed (0 files)" with exit code 0 -- the count was right there and the
    # word "passed" is what anyone reads (issue #11).
    #
    # The floor is deliberately crude. Its job is to separate "checked the tree"
    # from "checked nothing", not to track how many files the project has, which
    # would make it a second thing to update on every commit.
    MINIMUM_FILES = 20
    if checked < MINIMUM_FILES:
        print(
            f"layering check INCONCLUSIVE -- examined only {checked} file(s), expected at "
            f"least {MINIMUM_FILES}.\nThe sources were found and then not resolved to a "
            f"layer, so nothing was actually checked. This is a bug in the checker, not a "
            f"clean tree.",
            file=sys.stderr,
        )
        return 1

    print(f"layering check passed ({checked} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
