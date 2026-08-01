#!/usr/bin/env python3
"""VKML_MIN_SPEC: does it still lower the limits, and does anything still run it?

WHY THIS EXISTS. This project's most common bug has one shape -- code written
against what the development GPU reports rather than what Vulkan guarantees.
Push constants (#2), workgroups per dispatch (#20) and workgroup invocations
(#21) were each that bug, and each was found by somebody else's hardware.

VKML_MIN_SPEC=1 exists to make that class reproducible without owning the
hardware, and CLAUDE.md tells readers to run it before claiming a limit is
satisfied. An audit found it referenced by zero scripts and zero workflow files.
One test used it, in a subprocess, covering matmul and softmax. The instruction
was five months old and had never been carried out by anything but a person
remembering to.

TWO FAILURE MODES, and the second is the one worth having a gate for.

  1. Nothing runs it.       Then it is documentation, and documentation is
                            followed until the first busy week.
  2. It runs and does       If the environment variable stopped being read --
     nothing.               renamed, refactored, lost in a merge -- the CI arm
                            would keep passing, because it would be an ordinary
                            suite run wearing a min-spec label. Green, and
                            proving nothing. That is exactly the shape of the
                            three dead gates verify_gates.py was written for.

DERIVED, NOT RETYPED. The floors come from parsing the clamps out of
vk_device.cpp, which is the code that applies them. Writing `128` here would be
a second copy of the Vulkan Required Limits table with nothing able to compare
the two -- category 3 in docs/ENGINEERING-PRINCIPLES.md, and the failure it
warns about is that the copy drifts silently and the drift always looks correct.

    python scripts/check_min_spec.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ci import runs_command  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEVICE_CPP = ROOT / "src" / "backend" / "vulkan" / "vk_device.cpp"
CI = ROOT / ".github" / "workflows" / "ci.yml"

# The C++ field names that the Python capability dict exposes under a different
# name. Everything else maps to itself. Kept explicit and tiny: a clever
# normaliser would silently match the wrong pair.
ALIASES = {"max_shared_memory": "max_shared_memory_bytes"}

CLAMP = re.compile(r"info\.(\w+)(?:\[\d\])?\s*=\s*std::min(?:<\w+>)?\("
                   r"\s*info\.\w+(?:\[\d\])?\s*,\s*(\d+)U?\s*\)")

# Exit 3 means the module could not be imported AT ALL, which is a broken
# environment rather than a machine without a GPU. The two have to be told
# apart: a gate that treats "I could not run this" the same as "there is
# nothing to check" reports green for both, and the second is the one that
# happens on a developer's machine with the wrong interpreter.
PROBE = ("import sys, json\n"
         "try:\n"
         "    import vkml as V\n"
         "except BaseException as e:\n"
         "    print(f'import failed: {e!r}', file=sys.stderr); sys.exit(3)\n"
         "V.set_log_level(V.LogLevel.ERROR)\n"
         "V.init_vulkan(0)\n"
         "print(json.dumps(V.vulkan_capabilities(0)))\n")


def declared_floors() -> dict[str, int]:
    """The clamps VKML_MIN_SPEC applies, read from the function that applies them."""
    src = DEVICE_CPP.read_text()
    start = src.find('env_flag("VKML_MIN_SPEC"')
    if start < 0:
        print(f"  FAIL  no VKML_MIN_SPEC block in {DEVICE_CPP.relative_to(ROOT)} — "
              f"the facility this gate guards has been removed or renamed")
        return {}
    block = src[start:src.find("VKML_LOG_INFO", start)]
    out: dict[str, int] = {}
    for field, value in CLAMP.findall(block):
        # Indexed limits (workgroup_count[0..2]) clamp to different values per
        # axis; keep the loosest, since the check is an upper bound.
        out[field] = max(out.get(field, 0), int(value))
    return out


class Unusable(Exception):
    """The probe could not run. Carries whether that is fatal."""

    def __init__(self, why: str, fatal: bool):
        super().__init__(why)
        self.fatal = fatal


def capabilities(min_spec: bool) -> dict:
    env = dict(os.environ, PYTHONPATH="python", VKML_MIN_SPEC="1" if min_spec else "0")
    proc = subprocess.run([sys.executable, "-c", PROBE],
                          capture_output=True, text=True, env=env, cwd=ROOT)
    if proc.returncode == 3:
        raise Unusable(f"{sys.executable} cannot import vkml — "
                       f"{proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else 'no detail'}",
                       fatal=True)
    if proc.returncode != 0:
        raise Unusable("no usable Vulkan device", fatal=False)
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise Unusable("the probe printed something that is not capabilities JSON",
                       fatal=True) from None


def check_ci() -> int:
    """Something must actually run the suite at the floor."""
    if not CI.is_file():
        print("  no CI workflow; skipping the enforcement check")
        return 0
    # SHELL LINES ONLY, via _ci. The first version of this checked whole steps,
    # and the negative control caught it: deleting VKML_MIN_SPEC=1 from the
    # command left the step's own comment explaining VKML_MIN_SPEC=1, so the
    # gate stayed green over a step that no longer ran it.
    runs_it = runs_command(CI, "VKML_MIN_SPEC=1", "pytest")
    if not runs_it:
        print("  FAIL  no CI step runs the suite under VKML_MIN_SPEC=1.")
        print("        The guaranteed-floor limits are then only checked when "
              "somebody remembers to,")
        print("        which is the state this gate was written to end. See "
              "CLAUDE.md, issue #21.")
        return 1
    print("  ok    a CI step runs the suite at the guaranteed floor")
    return 0


def check_device(floors: dict[str, int]) -> int:
    try:
        real = capabilities(False)
        floor = capabilities(True)
    except Unusable as e:
        if e.fatal:
            # Not the same as having no GPU. Reporting it as one would make a
            # broken environment indistinguishable from a CPU-only CI job, and
            # this project's own notes warn that `import vkml` may resolve
            # somewhere other than the tree you just built.
            print(f"  FAIL  {e}")
            print(f"        This is a broken environment, not a machine without "
                  f"a GPU. Use the project .venv,")
            print(f"        or check where it resolves: python -c "
                  f"'import vkml; print(vkml._vkml_core.__file__)'")
            return 1
        print(f"  skip  {e}; the mechanism check reads limits from a live device")
        return 0

    observable = {f: v for f, v in floors.items()
                  if ALIASES.get(f, f) in real and ALIASES.get(f, f) in floor}
    if not observable:
        print(f"  FAIL  none of the {len(floors)} clamped limits is exposed to "
              f"Python, so nothing here can be checked")
        return 1

    bad = 0
    lowered = []
    for field, limit in sorted(observable.items()):
        key = ALIASES.get(field, field)
        got, was = floor[key], real[key]
        if got > limit:
            print(f"  FAIL  {key}: reports {got} under VKML_MIN_SPEC=1, above "
                  f"the {limit} declared in vk_device.cpp")
            bad += 1
        elif got < was:
            lowered.append(f"{key} {was}->{got}")

    if not bad and not lowered:
        # Every observable limit already sits at or below the floor. Correct on
        # a genuinely minimum-spec device, and NOT evidence the variable is read
        # -- so it is reported rather than counted as a pass.
        print("  ??    this device is already at the floor for every observable "
              "limit, so this run cannot tell whether the variable is read at all")
    elif not bad:
        print(f"  ok    VKML_MIN_SPEC=1 lowers {', '.join(lowered)}")

    unobservable = len(floors) - len(observable)
    if unobservable:
        # Not a failure and not hidden: push constants, workgroup counts and
        # workgroup sizes are clamped but absent from vulkan_capabilities(), so
        # no test can assert on them. Printed every run so the gap stays visible.
        print(f"  note  {len(observable)} of {len(floors)} clamped limits are "
              f"observable from Python; {unobservable} "
              f"({', '.join(sorted(set(floors) - set(observable)))}) are not "
              f"exposed, so nothing can check them")
    return bad


def main() -> int:
    print("\n  VKML_MIN_SPEC — the Vulkan 1.3 guaranteed floor\n")
    floors = declared_floors()
    if not floors:
        return 1
    print(f"  {len(floors)} limits clamped by vk_device.cpp: "
          f"{', '.join(f'{k}={v}' for k, v in sorted(floors.items()))}")
    bad = check_device(floors) + check_ci()
    print()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
