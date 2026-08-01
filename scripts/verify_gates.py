#!/usr/bin/env python3
"""The verification system, verified. Every gate proves it can fail.

WHY THIS EXISTS. `check_docs_links.py` printed every broken link it found and
then fell off the end of the file, so its exit status was always 0. It reported
PASS in CI and in every local sweep for its entire life. Two other gates were
found doing the same kind of nothing: `check_docs_examples` reported PASS over
class-page examples it had never opened, and the mutation campaign reported
SURVIVED for seventeen mutations it never executed because it rebuilt one
extension and tested another.

None of that is visible from a green run. A gate that has never been shown to
fail has never been shown to work.

SO THIS DOES NOT ASK. It breaks something on purpose, runs the gate, and records
the exit status. The "verified" column is a test result, not a claim -- a
hand-maintained column saying "yes, we checked" would be exactly the assertion
this file exists to distrust.

HOW A CONTROL IS WRITTEN. Each names a gate, a file to damage, a precise edit,
and the class of defect that edit represents. The edit is applied to a copy on
disk, the gate is run, and the original is restored in a `finally` -- so an
interrupted run cannot leave a mutated tree, which is a mistake the mutation
campaign made for its whole life before it was fixed.

A gate with no control is REPORTED AS UNVERIFIED rather than omitted. An absent
control is the finding.

    python scripts/verify_gates.py            # run every control
    python scripts/verify_gates.py --list     # the dashboard without running
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "bin" / "python"
PY = PY if PY.exists() else Path(sys.executable)


@dataclass
class Control:
    """One negative control: damage this, and that gate must notice."""
    gate: str
    guards: str                     # the CLASS of defect, not the instance
    source_of_truth: str
    target: str | None = None       # file to damage, relative to the repo
    find: str | None = None
    replace: str | None = None
    defect: str = ""                # what the edit represents
    rebuild: bool = False           # rebuild the site before running the gate
    manual: str = ""                # why no automated control exists
    artifacts: str = ""             # what generated output depends on this gate


CONTROLS: list[Control] = [
    Control(
        gate="check_docs_links.py",
        guards="internal links and anchors that do not resolve",
        source_of_truth="the built site",
        target="web/_site/concepts.html",
        find="<p>", replace='<p><a href="does-not-exist.html">x</a></p><p>',
        defect="a link to a page that is not there",
        artifacts="every page's cross-references",
    ),
    Control(
        gate="check_docs_references.py",
        guards="cited paths, constants, heading ids, asset budgets and declared "
               "capability gaps that disagree with the code",
        source_of_truth="the tree, plus web/content declarations",
        target="web/content/capabilities.py",
        find='    "Prod": ("not-yet"',
        replace='    "Matmul": ("not-yet", "stale claim"),\n    "Prod": ("not-yet"',
        defect="a reason declared for a gap that no longer exists -- the site "
               "telling a reader something is unsupported after it was built",
        artifacts="capabilities, compatibility and limitations pages",
    ),
    Control(
        gate="check_design_system.py",
        guards="type, weight and colour drifting off the design system",
        source_of_truth="web/theme/vkml.css",
        target="web/theme/vkml.css",
        find="}\n", replace="}\n.control-probe { font-size: .8375rem; }\n",
        defect="a font size that is not on the nine-step scale",
        artifacts="every page's typography",
    ),
    Control(
        gate="check_layering.py",
        guards="a layer including from a layer above it",
        source_of_truth="the include graph in include/ and src/",
        target="src/core/storage.cpp",
        find='#include "vkml/core/storage.h"',
        replace='#include "vkml/core/storage.h"\n#include "vkml/api/tensor.h"',
        defect="an upward dependency, which is what the layer order forbids",
        artifacts="the generated layer diagram on arch-overview",
    ),
    Control(
        gate="check_versions.py",
        guards="a version stated in prose disagreeing with the file that decides it",
        source_of_truth="CMakeLists.txt, pyproject.toml",
        target="web/content/guide.py",
        find="CMake 3.25 or newer", replace="CMake 3.24 or newer",
        defect="documentation naming a version the build does not require",
        artifacts="the get-started page",
    ),
    Control(
        gate="check_push_constants.py",
        guards="a push-constant block over the 128 bytes Vulkan guarantees",
        source_of_truth="the GLSL in shaders/",
        target="shaders/reduce.comp",
        find="layout(push_constant, scalar) uniform Params {",
        replace="layout(push_constant, scalar) uniform Params {\n"
                "    uvec4 control_probe[8];   // 128 bytes on its own",
        defect="a block that fits the development GPU and not the guaranteed floor",
        artifacts="none -- this one guards the code",
    ),
    Control(
        gate="check_gate_coverage.py",
        guards="a gate that can fail but which nothing ever runs",
        source_of_truth="scripts/ that exit non-zero, against the workflows' shell lines",
        target=".github/workflows/ci.yml",
        # Damages a gate that ONLY ci.yml runs. `docs_graph.py --check` was the
        # original target and silently stopped being a control the moment
        # pages.yml started running it too: removing it from one workflow left
        # it accounted for by the other, so the gate stayed green and this
        # control reported FAILED -- correctly, because it could no longer
        # detect what it guards. A control can rot the same way a gate can.
        find="        run: python scripts/check_layering.py",
        replace="        run: true",
        defect="a gate quietly dropped from CI, or added and never wired in -- "
               "the state check_docs_links.py and docs_graph.py were both in",
        artifacts="PRE-COMMIT-CHECKLIST section 6, which it makes checkable",
    ),
    Control(
        gate="check_min_spec.py",
        guards="the Vulkan guaranteed-floor facility silently ceasing to be "
               "exercised, or to lower anything",
        source_of_truth="the clamps in vk_device.cpp, and a live device",
        target=".github/workflows/ci.yml",
        find="VKML_MIN_SPEC=1 VKML_TEST_DEVICE",
        replace="VKML_TEST_DEVICE",
        defect="the suite no longer running at the floor, while the step that "
               "used to do it is still there and still green",
        artifacts="none -- it guards the test matrix's only limits axis",
    ),
    # The device half of that gate -- source declares a floor the running binary
    # does not apply -- has an equally cheap control (edit 128U to 64U in
    # vk_device.cpp; verified, exits 1 with the two numbers) but it needs a
    # Vulkan device, and this dashboard runs in a job without one. It would
    # report a missing GPU as a gate that cannot fail, which is a lie in the
    # more dangerous direction.
    Control(
        gate="docs_graph.py --check",
        guards="a page becoming unreachable from prose, or linking nowhere",
        source_of_truth="the built site's link graph",
        manual="The control has to remove every prose link to one page across "
               "57 files and then restore them. Doing that in place risks "
               "leaving the site damaged if interrupted; it was verified by "
               "hand against arch-graph and the result recorded here.",
        artifacts="docs/ui/graph-baseline.json",
    ),
    Control(
        gate="check_docs_examples.py",
        guards="a published example whose output does not match what the code prints",
        source_of_truth="the imported module",
        manual="Damaging an example means editing web/content and rebuilding, "
               "which the bytecode cache makes non-deterministic inside one "
               "run -- an earlier attempt reported a mismatch on restored "
               "content until __pycache__ was cleared. Verified by hand: "
               "claiming 7 for an x.size of 6 fails it.",
        artifacts="91 executed statements across the site",
    ),
    Control(
        gate="mutation_check.py",
        guards="tests that cannot fail -- a green suite that proves nothing",
        source_of_truth="the kernels themselves",
        manual="This IS a negative-control harness: it breaks each kernel and "
               "confirms the suite notices. Running it from here would nest a "
               "control inside a control, rebuild the extension ~17 times and "
               "take an hour. Its own last full run was 30/30 killed.",
        artifacts="none -- it validates the test suite",
    ),
    Control(
        gate="check_source_links.py",
        guards="a file:line link that still resolves but points at the wrong symbol",
        source_of_truth="the source tree",
        manual="The natural control -- insert lines at the top of a header so "
               "every link below shifts -- would need a rebuild to regenerate "
               "the links, and the gate reads the BUILT site. Left manual; "
               "the drift it guards is real and was the reason it was written.",
        artifacts="440 verified deep links",
    ),
    Control(
        gate="check_precise_gemm.py",
        guards="a GEMM accumulator losing its NoContraction decoration (ADR 0005)",
        source_of_truth="the GLSL in shaders/",
        manual="The decoration is invisible in behaviour on RADV -- byte-identical "
               "machine code either way -- so a control here proves the gate "
               "reads the shader, which is all the gate claims.",
        artifacts="none -- guards a numerical contract",
    ),
    Control(
        gate="check_file_ownership.py",
        guards="root-owned artifacts left in the tree by a container running as root",
        source_of_truth="the filesystem",
        manual="Creating a root-owned file needs privileges this does not have. "
               "It was verified when written, by running it against the 551 "
               "files the CI container had already left behind.",
        artifacts="none -- guards the working tree",
    ),
]


def run_gate(gate: str) -> int:
    parts = gate.split()
    return subprocess.run([str(PY), str(ROOT / "scripts" / parts[0]), *parts[1:]],
                          capture_output=True, text=True, cwd=ROOT).returncode


def build_site() -> None:
    subprocess.run([str(PY), str(ROOT / "web" / "build.py")],
                   capture_output=True, cwd=ROOT)


def exercise(c: Control) -> tuple[str, str]:
    """Break it, run the gate, restore. Returns (verdict, detail)."""
    if c.manual:
        return "manual", c.manual.split(".")[0] + "."

    target = ROOT / c.target
    if not target.is_file():
        return "ERROR", f"{c.target} does not exist"
    original = target.read_text()
    if c.find not in original:
        return "STALE", f"the control's anchor is no longer in {c.target}"

    try:
        target.write_text(original.replace(c.find, c.replace, 1))
        if c.rebuild:
            build_site()
        code = run_gate(c.gate)
    finally:
        target.write_text(original)
        if c.rebuild:
            build_site()

    if code == 0:
        return "FAILED", "the gate did NOT notice — it cannot detect what it guards"
    clean = run_gate(c.gate)
    if clean != 0:
        return "FAILED", "the gate stayed red after the damage was reverted"
    return "verified", "fails on the defect, passes when restored"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="the dashboard, without running")
    args = ap.parse_args()

    print(f"\n  VERIFICATION DASHBOARD — {len(CONTROLS)} gates\n")
    bad = 0
    for c in CONTROLS:
        verdict, detail = ("not run", "") if args.list else exercise(c)
        if verdict in {"FAILED", "ERROR", "STALE"}:
            bad += 1
        mark = {"verified": "ok  ", "manual": "man ", "not run": "    "}.get(verdict, "!!  ")
        print(f"  {mark}{c.gate}")
        print(f"        guards      {c.guards}")
        print(f"        truth       {c.source_of_truth}")
        if c.artifacts:
            print(f"        artifacts   {c.artifacts}")
        if not args.list:
            print(f"        control     {verdict} — {detail}")
        print()

    auto = sum(1 for c in CONTROLS if not c.manual)
    print(f"  {auto} with an automated control, {len(CONTROLS) - auto} verified by hand "
          f"with the reason recorded.")
    if bad:
        print(f"  {bad} PROBLEM(S) — a gate that cannot fail is not a gate.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
