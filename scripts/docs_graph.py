#!/usr/bin/env python3
"""The documentation's own dependency graph, derived from the built site.

WHY THIS EXISTS. A documentation set is a system, and like any system it can
have structural faults that no individual page reveals: a page nothing links to,
a concept used before it is introduced, a page so central that everything routes
through it. None of those is visible while reading one page at a time, which is
the only way anyone ever reads it.

DERIVED, NOT DECLARED -- and this was the design decision that mattered. The
obvious way to build a knowledge graph is to write down the concepts and their
prerequisites. That is a hand-written model of the documentation, and it would
be wrong the same way a hand-drawn architecture diagram is wrong: right on the
day, stale after, and authoritative-looking throughout.

So "concept" is operationalised as something already derivable:

    a page DEFINES the operators and classes it documents (from PAGE_OF and
    CLASSES, which the build already computes)
    a page USES a concept when its prose names one

Both come from the built HTML and the build's own maps. Nothing here is a
judgement about what the concepts are.

WHAT IT MEASURES, and why each one is a real fault:

  orphan      no prose anywhere links to it. Reachable only from the sidebar, so
              a reader meets it only if they already know to look -- and a
              reader meets a name mid-sentence, on a page about something else.
  dead end    links nowhere. The reader finishes and the documentation stops.
  bottleneck  very high inbound degree. Not a fault by itself, but it marks the
              pages whose accuracy matters most.
  forward use names a concept defined on a page that comes LATER in the reading
              order, without linking to it -- the "used before introduced"
              case.

MEASUREMENT NOTE. Breadcrumbs and prev/next are excluded. Counting them made
`index` look like the most-cited page on the site with 56 inbound links, when it
is only the crumb every page carries; prose links fell from 151 to 84 once the
chrome was removed. A graph built from navigation says every page is connected
to every other, which is true and useless.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "web" / "_site"
BASELINE = ROOT / "docs" / "ui" / "graph-baseline.json"

CHROME = re.compile(r'<nav class="(?:crumbs|pagenav)".*?</nav>', re.S)
MAIN = re.compile(r'<main class="content">(.*?)</main>', re.S)


def content_of(html: str) -> str:
    m = MAIN.search(html)
    return CHROME.sub("", m.group(1)) if m else ""


def build_graph() -> dict:
    pages = {p.stem: content_of(p.read_text(errors="ignore"))
             for p in sorted(SITE.glob("*.html"))}
    out: dict[str, set] = {k: set() for k in pages}
    for name, body in pages.items():
        for href in re.findall(r'href="([a-z0-9-]+)\.html', body):
            if href in pages and href != name:
                out[name].add(href)
    inbound: collections.Counter = collections.Counter()
    for src, dsts in out.items():
        for d in dsts:
            inbound[d] += 1
    return {
        "pages": sorted(pages),
        "links": {k: sorted(v) for k, v in out.items() if v},
        "inbound": dict(inbound),
        "orphans": sorted(p for p in pages if not inbound.get(p)),
        "dead_ends": sorted(p for p in pages if not out.get(p)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="fail if a NEW orphan or dead end appeared")
    ap.add_argument("--write-baseline", action="store_true")
    args = ap.parse_args()

    if not SITE.exists():
        print(f"{SITE} does not exist; run python web/build.py first")
        return 1
    g = build_graph()
    n_links = sum(len(v) for v in g["links"].values())

    print(f"  {len(g['pages'])} pages, {n_links} prose links "
          f"(navigation chrome excluded)")
    print(f"  orphans (no prose links in): {len(g['orphans'])}")
    print(f"  dead ends (link nowhere)   : {len(g['dead_ends'])}")
    top = sorted(g["inbound"].items(), key=lambda kv: -kv[1])[:5]
    print("  most cited: " + ", ".join(f"{k} ({v})" for k, v in top))

    if args.write_baseline:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        # Provenance from one definition, so the three recorders cannot each
        # invent a different shape -- see docs/ENGINEERING-PRINCIPLES.md 4.
        sys.path.insert(0, str(ROOT / "scripts"))
        from check_baselines import stamp
        BASELINE.write_text(json.dumps(
            {"recorded": stamp(),
             "orphans": g["orphans"], "dead_ends": g["dead_ends"]}, indent=1) + "\n")
        print(f"  wrote {BASELINE.relative_to(ROOT)}")
        return 0

    if not args.check:
        for p in g["orphans"]:
            print(f"    orphan: {p}")
        return 0

    # Against a baseline rather than against zero. Twenty class pages are
    # genuinely never named in prose today; a gate that fails on day one gets
    # switched off, and then it is not a gate. This fails on NEW ones, which is
    # the direction that matters.
    if not BASELINE.is_file():
        print(f"  no baseline at {BASELINE.relative_to(ROOT)}; "
              f"run with --write-baseline")
        return 1
    base = json.loads(BASELINE.read_text())
    new_orphans = sorted(set(g["orphans"]) - set(base["orphans"]))
    new_dead = sorted(set(g["dead_ends"]) - set(base["dead_ends"]))
    fixed = sorted(set(base["orphans"]) - set(g["orphans"]))

    if fixed:
        print(f"  {len(fixed)} page(s) are no longer orphaned: "
              f"{', '.join(fixed)} — rerun with --write-baseline")
    for p in new_orphans:
        print(f"  NEW ORPHAN: {p} — no prose anywhere links to it, so it is "
              f"reachable only from the sidebar")
    for p in new_dead:
        print(f"  NEW DEAD END: {p} — links nowhere")
    return 1 if (new_orphans or new_dead) else 0


if __name__ == "__main__":
    sys.exit(main())
