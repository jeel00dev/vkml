#!/usr/bin/env python3
"""Screenshot the documentation site at several viewports.

WHY THIS EXISTS. A documentation site is judged by how it looks and how it
reads, and neither can be assessed from the generator's source. Every UI claim
in the tracker -- "the sidebar is cramped at 1280", "code blocks overflow on a
phone" -- has to be observed, and observed again after the change, or it is an
opinion.

Renders with the same engine most readers use, at the widths where layouts
actually break rather than at round numbers:

    390   iPhone-class portrait; the narrowest that matters
    768   tablet portrait, and the usual sidebar breakpoint
    1280  the most common laptop, where two-column layouts get tight
    1920  desktop, where an unconstrained measure gets too wide

Usage:
    python scripts/shoot_docs.py OUTDIR [--pages a,b,c] [--theme dark|light|both]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "web" / "_site"

# One page per LAYOUT, not one per topic: the site has 55 pages built from a
# handful of templates, and shooting all of them mostly produces duplicates.
DEFAULT_PAGES = [
    "index",                 # landing
    "api",                   # the operator index: every name on one page
    "api-linear-algebra-nn", # a category page: full operator entries
    "class-vkml-tensor",     # class reference, many members
    "arch-graph",            # architecture prose, long form
    "get-started",           # guide with REPL transcripts
    "reference-env",         # a wide table
]

VIEWPORTS = [(390, 844, "phone"), (768, 1024, "tablet"),
             (1280, 800, "laptop"), (1920, 1080, "desktop")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--pages", default=",".join(DEFAULT_PAGES))
    ap.add_argument("--theme", default="dark", choices=["dark", "light", "both"])
    ap.add_argument("--full", action="store_true",
                    help="whole page rather than the first viewport")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed: pip install playwright "
              "&& python -m playwright install chromium")
        return 1

    if not SITE.exists():
        print(f"{SITE} does not exist; run python web/build.py first")
        return 1

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    pages = [p.strip() for p in args.pages.split(",") if p.strip()]
    themes = ["dark", "light"] if args.theme == "both" else [args.theme]

    shots = 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for w, h, label in VIEWPORTS:
            for theme in themes:
                ctx = browser.new_context(viewport={"width": w, "height": h},
                                          device_scale_factor=2)
                # Set the theme the way the site does, BEFORE any document loads,
                # so the first paint is the themed one and no screenshot catches
                # the flash of the default.
                ctx.add_init_script(
                    f"try{{localStorage.setItem('vkml-theme','{theme}')}}catch(e){{}}")
                page = ctx.new_page()
                for name in pages:
                    f = SITE / f"{name}.html"
                    if not f.exists():
                        print(f"  missing: {f.name}")
                        continue
                    page.goto(f.as_uri())
                    page.wait_for_load_state("networkidle")
                    suffix = "-full" if args.full else ""
                    dest = out / f"{name}__{label}-{theme}{suffix}.png"
                    page.screenshot(path=str(dest), full_page=args.full)
                    shots += 1
                ctx.close()
        browser.close()

    print(f"  {shots} screenshots -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
