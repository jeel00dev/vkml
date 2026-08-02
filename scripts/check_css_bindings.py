#!/usr/bin/env python3
"""The markup and the stylesheet must agree about which classes exist.

WHY THIS EXISTS. A class in the HTML that no rule defines renders as a bare
div. A rule whose selector matches nothing is dead weight that still has to be
read, maintained and reasoned about. Neither shows up in a build, a link check,
a design-system check or a test -- the page renders, every link resolves, and
the result is simply wrong-looking.

This is not hypothetical. Two admonitions were added to the performance page
using `admonition` and `admonition-title`, which is Sphinx's convention and not
this project's; the correct one is `admon warn|note` with a `label` span and a
`body` div. The build printed nothing, `check_docs_links` passed, and
`check_design_system` passed, because none of them compares the two files.

`docs/HANDOFF.md` had already recorded "a CSS selector matching zero elements"
as a gate that ought to exist and did not. This is that gate, in both
directions, because the direction that actually bit was the other one.

WHAT COUNTS AS USED. Class attributes in the built pages, plus every class name
the site's script adds, removes, toggles, tests or selects on -- `active`,
`open`, `sel` and the rest are applied at runtime and are no less real for it.
Reading them out of the built site rather than from an exemption list is what
keeps this from needing one.

    python scripts/check_css_bindings.py
    python scripts/check_css_bindings.py --list
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSS = ROOT / "web" / "theme" / "vkml.css"
SITE = ROOT / "web" / "_site"

# Classes a rule may define for a state the site does not currently reach, with
# the reason. Kept empty until something earns a place: an exemption list is the
# part of a gate that rots, and every entry has to be re-justified when the
# stylesheet changes.
EXEMPT_UNUSED: dict[str, str] = {}

# Classes that intentionally have no rule, with what styles the element instead.
# These make the ELEMENT check pass where it would otherwise fire on markup that
# renders correctly.
EXEMPT_UNDEFINED: dict[str, str] = {
    "prev": "a positional marker inside .pagenav; `.pagenav a` styles it and only "
            "its sibling `.next` needs an override",
}


def strip_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/", "", text, flags=re.S)


def defined_classes(css: str) -> set[str]:
    """Class names appearing in a selector, ignoring declaration bodies."""
    selectors = re.sub(r"\{[^{}]*\}", "{}", css)
    return set(re.findall(r"\.(-?[_a-zA-Z][\w-]*)", selectors))


def class_groups(html: str) -> list[list[str]]:
    """One list per `class="..."` attribute — the classes ON one element.

    The unit matters. An element carrying `rz rz-architecture` is styled by the
    `.rz` rule; `.rz-architecture` deliberately has no rule of its own, because
    the base is neutral and only two of the family carry a hue. Asking of each
    NAME whether a rule defines it calls that a defect. Asking of each ELEMENT
    whether ANY of its classes does is the question that means something.
    """
    return [value.split() for value in re.findall(r'class="([^"]*)"', html)]


def used_in_markup(html: str) -> set[str]:
    used: set[str] = set()
    for group in class_groups(html):
        used.update(group)
    return used


def used_in_script(js: str) -> set[str]:
    """Classes the script applies or selects on.

    `contains` counts as a use: a class only ever tested still has to be put
    there by something, and styled by something.
    """
    used: set[str] = set()
    for name in re.findall(r"""classList\.(?:add|remove|toggle|contains)\(\s*['"]([^'"]*)['"]""",
                           js):
        used.add(name)
    for value in re.findall(r"""className\s*=\s*['"]([^'"]*)['"]""", js):
        used.update(value.split())
    for selector in re.findall(
            r"""(?:querySelector|querySelectorAll|matches|closest)\(\s*['"]([^'"]*)['"]""", js):
        used.update(re.findall(r"\.(-?[_a-zA-Z][\w-]*)", selector))
    return used


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print both sets")
    args = ap.parse_args()

    if not CSS.exists():
        print(f"missing {CSS}", file=sys.stderr)
        return 1
    pages = sorted(SITE.glob("*.html"))
    if not pages:
        print("no built pages; run `python web/build.py` first", file=sys.stderr)
        return 1

    defined = defined_classes(strip_comments(CSS.read_text()))

    used: set[str] = set()
    where: dict[str, str] = {}
    unstyled: list[tuple[str, str]] = []       # (page, the whole class attribute)
    for page in pages:
        text = page.read_text()
        for name in used_in_markup(text):
            used.add(name)
            where.setdefault(name, page.name)
        for group in class_groups(text):
            if group and not any(name in defined or name in EXEMPT_UNDEFINED
                                 for name in group):
                unstyled.append((page.name, " ".join(group)))

    scripts = sorted(SITE.rglob("*.js"))
    for script in scripts:
        text = script.read_text()
        # Both extractors: the script applies classes through classList AND
        # builds markup by concatenating `class="..."` into innerHTML, and the
        # search results are entirely the second kind.
        for name in used_in_script(text) | used_in_markup(text):
            used.add(name)
            where.setdefault(name, script.name)

    unused = sorted(defined - used - set(EXEMPT_UNUSED))

    if args.list:
        print(f"\n  {len(defined)} classes defined, {len(used)} used\n")
        for name in sorted(defined | used):
            mark = "both" if name in defined and name in used else (
                "CSS only" if name in defined else "HTML only")
            print(f"    {name:<28} {mark}")

    print(f"  {len(pages)} pages and {len(scripts)} script(s), "
          f"{len(used)} classes used, {len(defined)} defined")

    if unstyled:
        seen: dict[str, str] = {}
        for page, group in unstyled:
            seen.setdefault(group, page)
        print(f"\n  {len(seen)} element(s) whose classes are ALL undefined -- they render "
              f"unstyled:", file=sys.stderr)
        for group, page in sorted(seen.items()):
            print(f"    class=\"{group}\"   in {page}", file=sys.stderr)
        print("\n  Either add a rule, or use the class the stylesheet already has.\n",
              file=sys.stderr)

    if unused:
        print(f"\n  {len(unused)} rule(s) whose selector matches nothing on the site:",
              file=sys.stderr)
        for name in unused:
            print(f"    .{name}", file=sys.stderr)
        print("\n  Delete them, or record why they are kept in EXEMPT_UNUSED.\n",
              file=sys.stderr)

    if unstyled or unused:
        return 1
    print("  every element is styled by something, and every rule has a user")
    return 0


if __name__ == "__main__":
    sys.exit(main())
