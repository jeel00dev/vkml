#!/usr/bin/env python3
"""Every internal link and anchor in the built site must resolve.

WHY THIS IS SEPARATE from the other two link gates, which look redundant and are
not:

    this file              does the target PAGE and ANCHOR exist?
    check_source_links.py  does the target LINE still hold the symbol its label
                           names? A file:line link goes stale silently -- insert
                           ten lines at the top of a header and every link below
                           points elsewhere while still resolving.
    check_docs_references  does a path CITED IN PROSE exist in the tree?

Three different failure modes, so three checks rather than one.

THIS GATE COULD NOT FAIL FOR ITS ENTIRE LIFE. It printed whatever it found and
then fell off the end of the file, so its exit status was always 0 -- it reported
PASS in CI and in every local sweep regardless of what it detected. Verified by
pointing a page at `does-not-exist.html`: it printed "missing page
does-not-exist.html" and exited 0.

That is exactly the failure docs/DOCUMENTATION-PRINCIPLES.md 9 exists for. It
was written, it looked right, it ran green, and nobody ever broke something on
purpose to check that green meant anything.

It also resolved `web/_site` relative to the working directory, so it silently
found no pages at all when run from anywhere but the repository root -- a second
way to report success without looking.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "web" / "_site"


def main() -> int:
    if not SITE.exists():
        print(f"{SITE} does not exist; run python web/build.py first")
        return 1

    pages = {p.name for p in SITE.glob("*.html")}
    anchors = {p.name: set(re.findall(r'id="([^"]+)"', p.read_text(errors="ignore")))
               for p in SITE.glob("*.html")}
    if not pages:
        print(f"no pages found under {SITE}; nothing was checked")
        return 1

    bad: list[str] = []
    for p in sorted(SITE.glob("*.html")):
        for href in re.findall(r'href="([^"]+)"', p.read_text(errors="ignore")):
            # Strip a cache-busting query string. The stylesheet ships as
            # theme/vkml.css?v=<hash> so a changed file is a different URL and
            # browsers cannot serve a stale one; the file on disk is still
            # theme/vkml.css, and treating the query as part of the path made
            # this gate report all 58 pages as broken.
            href = href.split("?", 1)[0]
            if href.startswith(("http", "#", "mailto")):
                continue
            f, _, frag = href.partition("#")
            if f and f not in pages and not (SITE / f).exists():
                bad.append(f"{p.name}: missing page {f}")
            elif frag and f in anchors and frag not in anchors[f]:
                bad.append(f"{p.name}: {f}#{frag} does not exist on that page")

    print(f"  {len(pages)} pages, {sum(len(a) for a in anchors.values())} anchors")
    if bad:
        print(f"  {len(bad)} broken link(s):")
        for b in bad[:20]:
            print(f"    {b}")
        if len(bad) > 20:
            print(f"    ... and {len(bad) - 20} more")
        return 1
    print("  every internal link and anchor resolves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
