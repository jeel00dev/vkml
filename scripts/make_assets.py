#!/usr/bin/env python3
"""Derive the site's brand assets from the master logo.

WHY THIS EXISTS. `assets/vkml_logo.png` is 1536x1024 and 2.4 MB, and the site
was serving it three ways, all wrong: as the favicon, which browsers render at
16-32px; in the topbar at 25.6px; and in the hero at 120px. Against 54.7 kB of
HTML, CSS and JavaScript, the logo was 97.8% of every page load.

Derived rather than hand-exported so the master stays the single source. Run it
after changing the logo:

    python scripts/make_assets.py

Sizes are 2x their rendered size, which is what a high-density display asks for
and the point past which more pixels are invisible. The OG image is 1200x630
because that is what Slack, Discord, X and LinkedIn crop to.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "assets" / "vkml_logo.png"
OUT = ROOT / "assets" / "derived"

# (filename, longest edge in px, palette colours or None for full RGBA, why)
#
# The palette entry is only set where full colour costs more than the 80 kB
# per-asset budget check_docs_references enforces on the built site. At 880 px
# the logo is 235 kB as RGBA and 47 kB quantised to 256 colours, and the two are
# indistinguishable: alpha survives (corners still measure 0, 96 distinct
# levels), 0.47% of visible pixels differ by more than 32/255, and the parts
# that carry the brand -- the metallic gradient in the wordmark, the brush ring,
# the particle squares -- are unchanged. Below 256 px there is nothing to gain,
# so those stay RGBA rather than being quantised on principle.
SIZES = [
    ("favicon-32.png", 32, None, "browser tab"),
    ("apple-touch-icon.png", 180, None, "iOS home screen, the one size it asks for"),
    ("logo-64.png", 64, None, "topbar, rendered at 25.6px"),
    ("logo-256.png", 256, None, "hero, rendered at 120px"),
    ("logo-880.png", 880, 256, "README, rendered at 440px"),
]

OG = (1200, 630)
OG_BG = (19, 20, 26)          # --bg, the dark theme page background


def main() -> int:
    try:
        from PIL import Image
    except ImportError:
        print("needs Pillow: pip install pillow")
        return 1

    if not MASTER.exists():
        print(f"missing {MASTER}")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    master = Image.open(MASTER).convert("RGBA")
    print(f"  master {master.width}x{master.height}, "
          f"{MASTER.stat().st_size / 1e6:.2f} MB")

    total = 0
    for name, edge, colors, why in SIZES:
        im = master.copy()
        im.thumbnail((edge, edge), Image.LANCZOS)
        if colors:
            # FASTOCTREE rather than the default: it is the one Pillow method
            # that quantises RGBA, and alpha is not optional here.
            im = im.quantize(colors=colors, method=Image.FASTOCTREE)
        dst = OUT / name
        # Alpha is preserved: the corners measure 0 and the glow is part of the
        # artwork, so flattening onto a background would put a box round it.
        im.save(dst, optimize=True)
        total += dst.stat().st_size
        print(f"  {name:22} {im.width:>4}x{im.height:<4} "
              f"{dst.stat().st_size / 1000:>6.1f} kB   {why}"
              f"{'' if not colors else f' ({colors} colours)'}")

    # The social card. Composed on the brand background because transparency
    # renders unpredictably in preview cards -- some clients put it on white,
    # some on black, and the logo's glow only reads on dark.
    card = Image.new("RGB", OG, OG_BG)
    mark = master.copy()
    mark.thumbnail((int(OG[0] * 0.42), int(OG[1] * 0.62)), Image.LANCZOS)
    card.paste(mark, ((OG[0] - mark.width) // 2, (OG[1] - mark.height) // 2), mark)
    dst = OUT / "og-card.png"
    card.save(dst, optimize=True)
    total += dst.stat().st_size
    print(f"  {'og-card.png':22} {OG[0]:>4}x{OG[1]:<4} "
          f"{dst.stat().st_size / 1000:>6.1f} kB   link previews")

    print(f"\n  {len(SIZES) + 1} files, {total / 1000:.1f} kB total "
          f"(master was {MASTER.stat().st_size / 1000:.0f} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
