#!/usr/bin/env python3
"""clang-format, as a gate that can be run before the commit that breaks it.

WHY THIS EXISTS. This check was a shell one-liner inside ci.yml for its whole
life, and it went red at f595d73 and stayed red for fifteen commits. Nothing
noticed, because a `run:` line in a workflow file is invisible to every
mechanism this project built to stop exactly that:

    check_gate_coverage.py  asks which SCRIPTS CI runs; a shell step is not one
    verify_gates.py         breaks a thing and checks the GATE goes red; it had
                            no gate to name
    PRE-COMMIT-CHECKLIST    names it, in prose, as a find | xargs to type by hand

A checklist item is not a mechanism. This is the second formatting backlog the
project has shipped for that reason -- 42c1d33 cleared 2,046 violations -- and
both times the code was fine and the process was missing a command.

THE VERSION IS PART OF THE CHECK, not an installation detail. clang-format
changes its mind between releases, so a differently-versioned binary disagrees
about untouched code and the gate's verdict starts depending on the date.
CONTRIBUTING.md, ci.yml and this file must name the same version; the check
below refuses to run on any other rather than reporting a difference of opinion
as a defect.

    python scripts/check_format.py           # check, and name what differs
    python scripts/check_format.py --fix     # rewrite the offending files
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Keep in step with .github/workflows/ci.yml and CONTRIBUTING.md.
REQUIRED_VERSION = "18.1.8"

#: The trees CI formats. Not `find .`: third_party/ is vendored, and build
#: directories hold generated headers nobody edits.
TREES = ("include", "src", "tests/cpp", "bench/cpp", "bindings")


def sources() -> list[Path]:
    out: list[Path] = []
    for tree in TREES:
        base = ROOT / tree
        if not base.is_dir():
            continue
        out += [p for p in base.rglob("*.h")]
        out += [p for p in base.rglob("*.cpp")]
    return sorted(out)


def find_clang_format() -> Path | None:
    """The project venv first: the system one is usually a different version."""
    candidates = [ROOT / ".venv" / "bin" / "clang-format",
                  ROOT / ".venv" / "Scripts" / "clang-format.exe"]
    for c in candidates:
        if c.exists():
            return c
    found = shutil.which("clang-format")
    return Path(found) if found else None


def version_of(binary: Path) -> str:
    text = subprocess.run([str(binary), "--version"], capture_output=True, text=True).stdout
    # "clang-format version 18.1.8" or "... 18.1.8 (https://...)"
    for word in text.split():
        if word[:1].isdigit():
            return word
    return text.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true", help="rewrite files instead of reporting them")
    args = ap.parse_args()

    binary = find_clang_format()
    if binary is None:
        print("clang-format not found. Install the pinned version:", file=sys.stderr)
        print(f"    pip install clang-format=={REQUIRED_VERSION}", file=sys.stderr)
        return 1

    version = version_of(binary)
    if version != REQUIRED_VERSION:
        print(f"clang-format {version} at {binary}, but this project pins "
              f"{REQUIRED_VERSION}.", file=sys.stderr)
        print("Different releases disagree about UNCHANGED code, so a mismatched "
              "binary reports style churn as a defect. Install the pinned one:",
              file=sys.stderr)
        print(f"    pip install clang-format=={REQUIRED_VERSION}", file=sys.stderr)
        return 1

    files = sources()
    if not files:
        print("no C++ sources found; the tree list is wrong", file=sys.stderr)
        return 1

    if args.fix:
        subprocess.run([str(binary), "-i", *[str(p) for p in files]], check=True)
        print(f"formatted {len(files)} files with clang-format {version}")
        return 0

    # One invocation per file, so the report names files rather than a wall of
    # column offsets. The whole tree takes about a second either way.
    offenders: list[Path] = []
    for path in files:
        done = subprocess.run([str(binary), "--dry-run", "--Werror", str(path)],
                              capture_output=True, text=True)
        if done.returncode != 0:
            offenders.append(path)

    if offenders:
        print(f"\n  clang-format {version} — {len(offenders)} of {len(files)} files differ\n",
              file=sys.stderr)
        for path in offenders:
            print(f"    {path.relative_to(ROOT)}", file=sys.stderr)
        print("\n  python scripts/check_format.py --fix\n", file=sys.stderr)
        return 1

    print(f"clang-format {version}: {len(files)} files, all formatted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
