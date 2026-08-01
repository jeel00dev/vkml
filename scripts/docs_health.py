#!/usr/bin/env python3
"""Documentation health, measured the way test coverage is measured.

WHY THESE METRICS. The question a documentation set has to keep answering is not
"is it finished" but "how much of it could quietly become false". A page written
by hand from a source of truth is correct today and unguarded tomorrow; a page
derived from that source is correct whenever it builds. So the headline number
here is the share of content that is DERIVED rather than typed -- it is the
share that cannot rot.

Every metric is computed from the tree or the built site. None is entered by
hand, because a hand-maintained health metric is the first thing to go stale and
would be a joke at this file's expense.

Deliberately NOT measured: page views, time on page, search queries. They are
about an audience this has none of yet, and optimising them would mean guessing.

    python scripts/docs_health.py            # the report
    python scripts/docs_health.py --json     # machine-readable
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "web" / "_site"
CONTENT = ROOT / "web" / "content"


def _run(script: str, *args: str) -> str:
    """A gate's own output is the measurement. Re-deriving it here would be a
    second implementation of the thing the gate already computes."""
    exe = ROOT / ".venv" / "bin" / "python"
    proc = subprocess.run(
        [str(exe if exe.exists() else sys.executable), str(ROOT / "scripts" / script), *args],
        capture_output=True, text=True, cwd=ROOT)
    return proc.stdout


def collect() -> dict:
    sys.path.insert(0, str(ROOT / "web"))
    sys.path.insert(0, str(ROOT / "python"))
    import build as B          # noqa: E402
    import research as R       # noqa: E402

    m: dict = {}

    # --- how much of the site is derived rather than typed -------------------
    #
    # Counted in PAGES, not in bytes: a page is the unit a reader meets and the
    # unit that can be wrong. Operator, class and capability pages are built
    # from extraction; guide and architecture pages are prose.
    pages = sorted(p.stem for p in SITE.glob("*.html"))
    generated = [p for p in pages
                 if p.startswith(("api", "class-")) or p in
                 {"capabilities", "compatibility", "limitations"}]
    m["pages_total"] = len(pages)
    m["pages_generated"] = len(generated)
    m["pct_generated"] = round(100 * len(generated) / max(len(pages), 1))

    # --- claims with a machine-checked backing -------------------------------
    ex = _run("check_docs_examples.py")
    mm = re.search(r"(\d+) statements checked", ex)
    m["examples_executed"] = int(mm.group(1)) if mm else 0
    m["examples_failing"] = int(re.search(r"(\d+) mismatches", ex).group(1)) if "mismatch" in ex else 0

    sl = _run("check_source_links.py")
    mm = re.search(r"(\d+)", sl)
    m["source_links_verified"] = int(mm.group(1)) if mm else 0

    dr = _run("check_docs_references.py")
    mm = re.search(r"(\d+) path references", dr)
    m["path_references_verified"] = int(mm.group(1)) if mm else 0

    # --- operators with an extracted implementation trail --------------------
    facts = B.FACTS
    m["operators"] = len(facts)
    m["operators_with_tests"] = sum(1 for f in facts.values() if f.tests)
    m["operators_with_cpu_kernel"] = sum(1 for f in facts.values() if f.cpu_kernel)
    m["operators_with_gradient"] = sum(1 for f in facts.values() if f.has_grad)
    m["operators_with_adr"] = sum(1 for n in facts if n in B.ADR_MENTIONS)
    m["pct_operators_tested"] = round(
        100 * m["operators_with_tests"] / max(m["operators"], 1))

    # --- diagrams ------------------------------------------------------------
    svgs = sum(len(re.findall(r'<svg class="dia"', p.read_text(errors="ignore")))
               for p in SITE.glob("*.html"))
    m["diagrams_generated"] = svgs
    m["diagrams_handdrawn"] = len(list((ROOT / "docs" / "ui").glob("*.svg")))

    # --- structure, from the link graph -------------------------------------
    g = _run("docs_graph.py")
    for key, pat in (("prose_links", r"(\d+) prose links"),
                     ("orphans", r"orphans \(no prose links in\): (\d+)"),
                     ("dead_ends", r"dead ends \(link nowhere\)\s*: (\d+)")):
        mm = re.search(pat, g)
        m[key] = int(mm.group(1)) if mm else None

    # --- declared exceptions, which are debt with a reason -------------------
    try:
        from content.capabilities import BACKEND_REASONS, GRADIENT_REASONS
        m["declared_gaps"] = len(GRADIENT_REASONS) + len(BACKEND_REASONS)
        m["real_gaps"] = sum(1 for v in GRADIENT_REASONS.values()
                             if v[0] not in ("by-design",))
    except Exception:                                   # noqa: BLE001
        m["declared_gaps"] = m["real_gaps"] = None
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not SITE.exists():
        print(f"{SITE} does not exist; run python web/build.py first")
        return 1
    m = collect()
    if args.json:
        print(json.dumps(m, indent=1))
        return 0

    def line(label: str, value, note: str = "") -> None:
        print(f"  {label:<34} {str(value):>8}   {note}")

    print("\n  DERIVED vs TYPED")
    line("pages built from extraction",
         f"{m['pages_generated']}/{m['pages_total']}", f"{m['pct_generated']}%")
    line("diagrams generated from source", m["diagrams_generated"],
         "hand-drawn: %s" % m["diagrams_handdrawn"])

    print("\n  CLAIMS WITH A MACHINE-CHECKED BACKING")
    line("examples executed on every build", m["examples_executed"],
         "failing: %s" % m["examples_failing"])
    line("source links verified", m["source_links_verified"])
    line("path references verified", m["path_references_verified"])

    print("\n  THE OPERATOR SURFACE")
    line("operators documented", m["operators"])
    line("with tests found by name",
         f"{m['operators_with_tests']}/{m['operators']}",
         f"{m['pct_operators_tested']}%")
    line("with a CPU kernel located", m["operators_with_cpu_kernel"])
    line("with a gradient rule", m["operators_with_gradient"])
    line("with an ADR explaining them", m["operators_with_adr"])

    print("\n  STRUCTURE")
    line("prose links between pages", m["prose_links"])
    line("pages no prose links to", m["orphans"])
    line("pages whose PROSE links nowhere", m["dead_ends"],
         "prev/next is excluded: it is chrome")

    print("\n  DECLARED GAPS (debt carrying a reason)")
    line("gaps with a declared reason", m["declared_gaps"])
    line("of those, genuinely missing work", m["real_gaps"],
         "the rest are by design")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
