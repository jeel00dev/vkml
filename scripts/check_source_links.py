#!/usr/bin/env python3
"""Every generated file:line link must point at what its label says.

WHY LINE NUMBERS ARE THE FRAGILE PART. The documentation's implementation tables
are extracted, so they cannot claim a file that does not exist -- an earlier gate
covers that. But a line NUMBER goes stale differently and more quietly: inserting
ten lines at the top of a header silently moves every link below it, the file
still exists, the link still resolves, and it now points at the wrong
declaration.

This checks the association rather than the existence. For every
`<a href=".../path#Lnnn">path:nnn</a>` in the built site, it opens the file and
looks for the label's subject near that line. What counts as the subject is taken
from the surrounding anchor -- an operator page's entry for `erfc` linking to
ops.h:105 should find `erfc` there.

A WINDOW rather than an exact line, because the extractor records the last line
of a joined multi-line declaration while a reader would call the first line its
location. Both are correct; a window of a few lines accepts either without
accepting a link that has drifted to a different symbol entirely.
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "web" / "_site"

# <a href="https://github.com/.../blob/main/PATH#LNNN">…</a>
LINK = re.compile(r'<a href="[^"]*?/blob/main/([^"#]+)#L(\d+)"[^>]*>(.*?)</a>', re.S)

# The nearest preceding entry anchor names what the link is about.
SUBJECT = re.compile(r'<h[234] id="([\w.:~-]+)"')

WINDOW = 12


def subject_at(text: str, pos: int) -> str | None:
    """The id of the nearest heading before `pos`."""
    best = None
    for m in SUBJECT.finditer(text, 0, pos):
        best = m.group(1)
    return best


def main() -> int:
    if not SITE.exists():
        print("  ERROR: web/_site not built; run python web/build.py first")
        return 1

    cache: dict[str, list[str]] = {}
    checked = missing = drifted = 0
    problems: list[str] = []

    for page in sorted(SITE.glob("*.html")):
        text = page.read_text()
        for m in LINK.finditer(text):
            rel, line_s = m.group(1), int(m.group(2))
            checked += 1

            target = ROOT / rel
            if not target.exists():
                missing += 1
                problems.append(f"{page.name}: links to {rel}, which does not exist")
                continue

            if rel not in cache:
                cache[rel] = target.read_text(errors="ignore").split("\n")
            lines = cache[rel]

            if line_s > len(lines):
                missing += 1
                problems.append(
                    f"{page.name}: {rel}#L{line_s} but the file has {len(lines)} lines")
                continue

            subject = subject_at(text, m.start())
            if not subject:
                continue
            # Member ids are prefixed; operator ids are the bare name.
            name = subject[2:] if subject.startswith("m-") else subject
            name = name.lstrip("~")
            if not name or not re.fullmatch(r"[\w.]+", name):
                continue

            # An operator is `scatter_add` on the page and `ScatterAdd` in the
            # OpKind enum, so both spellings are accepted. Without this the gate
            # reported every autograd link as drifted -- line 194 really is
            # `case OpKind::Sub:`, and only the CASE differed from the label.
            camel = "".join(part.capitalize() for part in name.split("_"))
            lo = max(0, line_s - 1 - WINDOW)
            hi = min(len(lines), line_s + 2)
            window = " ".join(lines[lo:hi])
            if name not in window and camel not in window:
                drifted += 1
                problems.append(
                    f"{page.name}: {rel}#L{line_s} is labelled for {name!r}, "
                    f"which does not appear within {WINDOW} lines of it")

    print(f"  {checked} source links checked across {len(list(SITE.glob('*.html')))} pages")
    if problems:
        print(f"  {missing} unresolvable, {drifted} pointing at the wrong place")
        for p in problems[:25]:
            print(f"    {p}")
        if len(problems) > 25:
            print(f"    … and {len(problems) - 25} more")
        return 1
    print("  every link resolves and names something present at its line")
    return 0


if __name__ == "__main__":
    sys.exit(main())
