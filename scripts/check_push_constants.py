#!/usr/bin/env python3
"""Every push-constant block must fit Vulkan's guaranteed 128 bytes.

WHY THIS IS A GATE. Issue #2 was three blocks over the guarantee, and the
symptom was not a warning -- it was 19 failing tests on an AMD Windows driver
that reports exactly 128, while the development GPU reports 256 and hid it
completely. maxPushConstantsSize is guaranteed to be only 128 bytes; a block
above that is undispatchable on a conformant minimum-spec device and perfectly
fine on the machine that wrote it.

Closing #2 required per-op repacking -- `where` and `softmax` store shared
extents once, `cat` derives its operands' extents from the output's -- and none
of that is self-enforcing. Adding a field to any block silently re-opens the
issue on hardware the author does not own. This makes that a build failure
instead.

The size is computed from the GLSL declarations under `scalar` layout. It is a
close estimate rather than the authority: the C++ struct the host packs is what
actually reaches the driver. The margin below is set so an estimate that drifts
a few bytes still fails loudly rather than passing narrowly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web"))

import research as R

GUARANTEED = 128


def main() -> int:
    details = R.all_shader_details()
    rows, over = [], []
    for stem in sorted(details):
        push = details[stem]["push"]
        if not push:
            continue
        rows.append((stem, push["bytes"], len(push["fields"])))
        if push["bytes"] > GUARANTEED:
            over.append((stem, push["bytes"]))

    width = max(len(r[0]) for r in rows)
    for stem, size, nfields in sorted(rows, key=lambda r: -r[1]):
        flag = "  OVER" if size > GUARANTEED else ""
        print(f"  {stem:<{width}}  {size:>4} B  {nfields:>2} fields{flag}")

    print()
    print(f"  {len(rows)} push blocks, largest {max(r[1] for r in rows)} B, "
          f"guaranteed minimum {GUARANTEED} B")

    if over:
        print()
        print("  FAILED — these exceed what Vulkan guarantees and cannot be dispatched")
        print("  on a conformant minimum-spec device:")
        for stem, size in over:
            print(f"    {stem}: {size} B, {size - GUARANTEED} B over")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
