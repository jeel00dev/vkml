#!/usr/bin/env python3
"""Version numbers stated in prose must match the file that decides them.

WHY THIS EXISTS. The get-started page said "CMake 3.24+" while
CMakeLists.txt requires 3.25. A reader with 3.24 would have followed the
documentation and failed to configure, with an error naming CMake rather than
the page that misled them. README had it right, so two documents disagreed and
nothing noticed.

A handful of version facts live in exactly one authoritative place each. Every
other mention is a copy, and a copy drifts. This extracts each fact from its
source and checks every restatement against it.

Deliberately narrow: it only checks numbers it can attribute to a single
authority. A version mentioned in prose that has no source here is left alone
rather than guessed at, because a gate that flags correct statements gets
switched off -- which is the lesson from the reference gate's `vkml.json` false
positive.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Where each fact is DECIDED, and how to read it.
SOURCES: dict[str, tuple[Path, str]] = {
    "cmake": (ROOT / "CMakeLists.txt", r"cmake_minimum_required\(\s*VERSION\s+([\d.]+)"),
    "cxx_standard": (ROOT / "CMakeLists.txt", r"set\(CMAKE_CXX_STANDARD\s+(\d+)\)"),
    "python": (ROOT / "pyproject.toml", r'requires-python\s*=\s*">=\s*([\d.]+)"'),
    "clang_format": (ROOT / ".github" / "workflows" / "ci.yml",
                     r"clang-format==([\d.]+)"),
}

# How each fact appears in prose. A match's group(1) is compared to the source.
MENTIONS: dict[str, list[str]] = {
    "cmake": [r"CMake\s+([\d.]+)\s*(?:\+|or newer|or later)"],
    "cxx_standard": [r"C\+\+(\d\d)\b"],
    "python": [r"Python\s+([\d.]+)\s*\+"],
    "clang_format": [r"clang-format[^\n]{0,20}?([\d.]{4,})"],
}

DOCS = (
    [ROOT / n for n in ("README.md", "CONTRIBUTING.md", "CLAUDE.md")]
    + sorted((ROOT / "docs").glob("*.md"))
    + sorted((ROOT / "web" / "content").glob("*.py"))
)


def read_sources() -> dict[str, str]:
    out = {}
    for key, (path, pattern) in SOURCES.items():
        if not path.exists():
            print(f"  ERROR: source file missing: {path.relative_to(ROOT)}")
            continue
        m = re.search(pattern, path.read_text())
        if not m:
            print(f"  ERROR: could not read {key} from {path.relative_to(ROOT)}")
            continue
        out[key] = m.group(1)
    return out


def main() -> int:
    truth = read_sources()
    if len(truth) != len(SOURCES):
        return 1

    print("  authoritative values:")
    for key, (path, _) in SOURCES.items():
        print(f"    {key:14} {truth[key]:<10} from {path.relative_to(ROOT)}")

    problems, checked = [], 0
    for doc in DOCS:
        if not doc.exists():
            continue
        text = doc.read_text(errors="ignore")
        for i, line in enumerate(text.split("\n"), 1):
            for key, patterns in MENTIONS.items():
                for pattern in patterns:
                    for m in re.finditer(pattern, line):
                        checked += 1
                        if m.group(1) != truth[key]:
                            problems.append(
                                f"{doc.relative_to(ROOT)}:{i}: says {key} "
                                f"{m.group(1)!r}, but {SOURCES[key][0].name} "
                                f"says {truth[key]!r}")

    print()
    print(f"  {checked} version mentions checked across {len(DOCS)} documents")
    if problems:
        print(f"  {len(problems)} disagreements:")
        for p in problems:
            print(f"    {p}")
        return 1
    print("  all agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
