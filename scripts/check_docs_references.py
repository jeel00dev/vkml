#!/usr/bin/env python3
"""Every file, symbol and constant the documentation cites must actually exist.

WHY THIS EXISTS. Writing the reduction page I stated that `pairwise_sum` lives
in `src/backend/cpu/iterate.h`. It lives in `src/backend/cpu/reduce.h`. The
source comment I was reading from was correct; I mis-attributed it while
summarising, and nothing would have caught that -- the page rendered, the links
resolved, the examples ran, and the sentence was simply false.

Prose is where documentation rots fastest, because unlike a link there is
nothing structural to break. This walks every backticked path, `path:line`
reference and named C++ constant in web/content/ and checks it against the tree:

  path.h              -> the file exists
  path.cpp:123        -> the file exists AND has at least that many lines
  `kSomeConstant`     -> the identifier appears somewhere in src/ or include/

The constant check is deliberately loose. It cannot tell a real citation from a
coincidence, and it is not trying to: it catches the case that actually happens,
which is a name that has been renamed or removed entirely.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))

from content import CLASSES, PROSE  # noqa: E402

# A backticked token that looks like a repository path, optionally with :line.
#
# Two forms are accepted and the distinction matters. A token containing a
# separator (`src/api/ops.cpp`) is unambiguously repo-relative. A bare filename
# is only treated as a repository reference when it carries a SOURCE extension,
# because documentation legitimately names files that are not in the tree --
# `vkml.json` is a member inside a checkpoint archive, and the first version of
# this gate reported it as missing. A gate that cries wolf gets switched off.
PATH_RE = re.compile(
    r"`("
    r"(?:[\w.-]+/)+[\w.-]+\.(?:h|cpp|comp|glsl|py|json|md|txt)"   # has a separator
    r"|[\w.-]+\.(?:h|cpp|comp|glsl|py)"                            # bare, source only
    r")(?::(\d+))?`")
# A backticked identifier in the project's constant style.
# Matches the identifier at the START of the backticked span, so
# `kPairwiseBlock = 32` is checked as kPairwiseBlock rather than skipped.
CONST_RE = re.compile(r"`(k[A-Z]\w+)\b")

SEARCH_DIRS = ["src", "include", "shaders", "python", "scripts", "docs", "tests", "web"]


def prose_fields(entry: dict):
    for key in ("summary", "detail", "note", "warning", "tip", "returns"):
        if entry.get(key):
            yield key, entry[key]
    for name, ptype, desc in entry.get("params", []):
        yield f"param {name}", desc


def resolve(rel: str) -> Path | None:
    """A cited path may be repo-relative or a bare basename."""
    direct = ROOT / rel
    if direct.exists():
        return direct
    if "/" not in rel:
        for d in SEARCH_DIRS:
            hits = list((ROOT / d).rglob(rel))
            if hits:
                return hits[0]
    return None


def main() -> int:
    sources = ""
    for d in ("src", "include", "shaders"):
        for p in (ROOT / d).rglob("*"):
            if p.is_file() and p.suffix in {".h", ".cpp", ".comp", ".glsl"}:
                sources += p.read_text(errors="ignore")

    problems, checked_paths, checked_consts = [], 0, 0

    for op, entry in sorted(PROSE.items()):
        for where, text in prose_fields(entry):
            for rel, line in PATH_RE.findall(text):
                checked_paths += 1
                target = resolve(rel)
                if target is None:
                    problems.append(f"{op} ({where}): no such file `{rel}`")
                elif line:
                    n = target.read_text(errors="ignore").count("\n") + 1
                    if int(line) > n:
                        problems.append(
                            f"{op} ({where}): `{rel}:{line}` but the file has {n} lines")

            for const in CONST_RE.findall(text):
                checked_consts += 1
                if const not in sources:
                    problems.append(f"{op} ({where}): `{const}` appears nowhere in the sources")

    print(f"  {checked_paths} path references, {checked_consts} constant references")
    problems += unlisted_native_members()
    if problems:
        print(f"  {len(problems)} problems:")
        for p in problems:
            print(f"    {p}")
        return 1
    print("  all resolve")
    return 0


def unlisted_native_members():
    """Every public member of a nanobind class must appear in a group on its page.

    The page for a class defined in C++ is built by introspecting the BUILT
    module, so a newly bound member appears on it the moment it is bound --
    but only in the catch-all "Other members" bucket, silently and with no
    prose. That is how the Python `Tensor` came to have six members documented
    nowhere at all (`T`, `astype`, `max`, `min`, `numpy`, `view`) while the
    page still looked complete.

    Checked against `dir()` on the real class rather than against a list kept
    here, because a list kept here would need the same discipline it is meant
    to enforce.
    """
    import importlib
    out = []
    for name, spec in sorted(CLASSES.items()):
        if spec.get("lang") != "native":
            continue
        try:
            cls = getattr(importlib.import_module("vkml"), spec.get("symbol", name))
        except (ImportError, AttributeError):
            out.append(f"{name}: cannot import to verify its members")
            continue
        grouped = {m for _, members in spec.get("groups", []) for m in members}
        for attr in sorted(dir(cls)):
            if attr.startswith("_") and attr not in spec.get("extra_members", ()):
                continue
            if attr not in grouped:
                out.append(f"{name}: `{attr}` is bound but listed in no group "
                           f"-- add it to a group in web/content/classes.py")
    return out


if __name__ == "__main__":
    sys.exit(main())
