#!/usr/bin/env python3
"""Every recorded baseline must carry the conditions that make it comparable.

WHY. A 38.6% regression was tracked, investigated and refuted: the benchmark had
been measured on a GPU parked at its idle clock, and the baseline it was compared
against was taken warm. `bench/baselines/rx5600m.json` records GPU model,
subgroup sizes and memory -- and no clock state, no driver version, no timestamp
period, no commit. Nothing in it could rule the explanation in or out, so the
investigation had to reconstruct conditions that should have been written down
when the number was taken.

A baseline is a claim that two measurements are comparable. Without the
conditions, that claim cannot be checked, and the number is worse than no
number because it looks authoritative.

SCOPE, established by measurement rather than assumed. Six recorded artifacts
exist; three are compared against and are in scope. The three
`examples/*_summary.json` files are write-only -- nothing reads them back -- so
they are outputs, not baselines, and requiring provenance of them would be
ceremony.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# path -> what a reader needs to judge whether it is still comparable.
REQUIRED: dict[str, set[str]] = {
    "bench/baselines/rx5600m.json": {"commit", "recorded_at", "driver", "warmed"},
    "docs/coverage-baseline.json": {"commit", "recorded_at"},
    "docs/ui/graph-baseline.json": {"commit", "recorded_at"},
}


def provenance(doc: dict) -> dict:
    return doc.get("recorded") if isinstance(doc.get("recorded"), dict) else {}


def main() -> int:
    problems: list[str] = []
    for rel, needed in sorted(REQUIRED.items()):
        p = ROOT / rel
        if not p.is_file():
            problems.append(f"{rel}: missing")
            continue
        try:
            doc = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            problems.append(f"{rel}: not valid JSON ({e})")
            continue
        if not isinstance(doc, dict):
            problems.append(f"{rel}: top level is not an object, so it cannot "
                            f"carry a `recorded` block")
            continue
        have = provenance(doc)
        missing = sorted(needed - set(have))
        if missing:
            problems.append(
                f"{rel}: `recorded` is missing {', '.join(missing)}. "
                f"A baseline without its conditions cannot be compared against "
                f"-- see docs/ENGINEERING-PRINCIPLES.md 4.")

    print(f"  {len(REQUIRED)} baselines checked")
    if problems:
        for p in problems:
            print(f"    {p}")
        return 1
    print("  every baseline records the conditions it was taken under")
    return 0


def stamp() -> dict:
    """The provenance block a recorder should embed. One definition, so the
    three recorders cannot each invent a different shape."""
    import datetime
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, cwd=ROOT).stdout.strip()
    return {"commit": commit or "unknown",
            "recorded_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}


if __name__ == "__main__":
    sys.exit(main())
