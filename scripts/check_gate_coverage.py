#!/usr/bin/env python3
"""Every gate must be run by something. This checks which ones are.

WHY THIS EXISTS. A gate has three ways to be useless, and this project has now
hit all three:

    it cannot fail            check_docs_links.py exited 0 for its entire life
    it tests the wrong thing  the mutation campaign rebuilt one extension and
                              ran another
    nothing runs it           VKML_MIN_SPEC, documented for five months, present
                              in zero scripts and zero workflows

verify_gates.py covers the first. The audit that found the third also found two
gates in the same state: `docs_graph.py --check` was run by nothing at all, and
`check_docs_links.py` was reachable only as a side effect of verify_gates
restoring its own damage -- so a genuinely broken link would have failed the
build under the message "the gate stayed red after the damage was reverted",
which names the wrong cause.

Neither was visible by reading either file. Both are obvious the moment the two
sets are put side by side, which is all this does.

WHY IT REPLACES A LIST. docs/PRE-COMMIT-CHECKLIST.md section 6 is headed "the
gates CI runs anyway" and enumerates them by hand. That heading is a claim about
another file, and nothing checked it: three gates CI runs were missing from it.
A hand-maintained list of what is automated is the least likely list to stay
true, because it has to be edited on exactly the days when somebody is busy
adding automation.

EXEMPTIONS ARE DECLARED, NOT INFERRED. A script that is not a CI gate says so
here with a reason. That keeps the question answerable -- "why is this not in
CI?" has an answer at the point of the exemption rather than in someone's
memory -- and it makes adding one a deliberate act.

    python scripts/check_gate_coverage.py
    python scripts/check_gate_coverage.py --list
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ci import run_blocks  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
WORKFLOWS = ROOT / ".github" / "workflows"

# Scripts that are not CI gates, and why. A reason of the form "it is an X"
# should say what runs it instead, or that nothing needs to.
EXEMPT: dict[str, str] = {
    "_ci.py": "a shared helper, not a script; exercised by its callers' controls",
    "check_baselines.py": "run by CI, but as part of the Baselines step -- listed "
                          "here only because it is also imported for stamp()",
    "coverage_matrix.py": "needs a VKML_COVERAGE trace from a full suite run; CI "
                          "invokes it in the coverage job with that argument",
    "tracker.py": "a report, not a gate -- the classification is a judgement "
                  "about work, and a CI job cannot adjudicate it",
    "docs_health.py": "a report, not a gate -- it has no failing condition and "
                      "is meant to be read",
    "measure_docs.py": "an instrument: fetches twelve reference sites over the "
                       "network, so it must not run per-push",
    "shoot_docs.py": "an instrument: drives headless Chromium to produce "
                     "screenshots, run when the UI changes",
    "make_assets.py": "a generator, run when the source images change",
    "hardware_report.py": "a diagnostic: prints what this machine reports, which "
                          "is different everywhere and asserts nothing",
    "mutation_check.py": "the full campaign rebuilds the extension per mutation "
                         "and takes about an hour; CI runs --patterns, and the "
                         "full run is scheduled rather than per-push",
    "check_cpu_only_build.py": "reproduces a configuration CI builds for real in "
                               "three jobs, so CI covers the defect directly; the "
                               "script exists to reproduce it locally in a minute",
    "verify_gates.py": "run by CI, and it is the harness the others are measured "
                       "by -- listed for completeness",
}

# Gates whose CI invocation carries an argument, so the bare name is not enough.
REQUIRE_ARGS: dict[str, str] = {
    "docs_graph.py": "--check",
    "mutation_check.py": "--patterns",
}


def gate_scripts() -> list[Path]:
    """Scripts that can fail on purpose -- the ones for which "does CI run it?" means something."""
    out = []
    for p in sorted(SCRIPTS.glob("*.py")):
        text = p.read_text()
        # A gate is a script with a deliberate non-zero exit. Checking for the
        # exit rather than the name, because naming is a convention and this
        # question is about behaviour.
        if re.search(r"sys\.exit\(1|sys\.exit\(main\(\)\)|return 1\b", text):
            out.append(p)
    return out


def ci_commands() -> list[str]:
    return [b for wf in sorted(WORKFLOWS.glob("*.yml")) for b in run_blocks(wf)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="show every gate and its state")
    args = ap.parse_args()

    commands = ci_commands()
    gates = gate_scripts()
    rows, missing = [], []

    for p in gates:
        name = p.name
        needed = REQUIRE_ARGS.get(name)
        if needed:
            run = any(f"scripts/{name}" in c and needed in c for c in commands)
            label = f"{name} {needed}"
        else:
            run = any(f"scripts/{name}" in c for c in commands)
            label = name
        if run:
            rows.append(("ci  ", label, "run by a CI step"))
        elif name in EXEMPT and name not in REQUIRE_ARGS:
            # A script in REQUIRE_ARGS cannot be exempted by its bare name. The
            # first version allowed it, and docs_graph.py -- the gate whose
            # absence prompted this file -- came out EXEMPT while its own reason
            # string said the --check form was required. An exemption that
            # cancels the requirement it explains is worse than no exemption.
            rows.append(("--  ", label, EXEMPT[name]))
        else:
            rows.append(("!!  ", label, "NOTHING RUNS THIS"))
            missing.append(label)

    if args.list or missing:
        print(f"\n  GATE COVERAGE — {len(gates)} scripts that can fail\n")
        for mark, label, why in rows:
            print(f"  {mark}{label}")
            print(f"        {why}")
        print()

    n_ci = sum(1 for m, _, _ in rows if m == "ci  ")
    print(f"  {n_ci} run by CI, {len(gates) - n_ci - len(missing)} exempt with a "
          f"recorded reason, {len(missing)} unaccounted for")
    for label in missing:
        print(f"  FAIL  nothing runs {label} — it can fail, and no job gives it "
              f"the chance. Add it to a workflow, or add an exemption saying why "
              f"it is not a gate.")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
