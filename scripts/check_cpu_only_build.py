#!/usr/bin/env python3
"""Run the Python suite against a build compiled WITHOUT the Vulkan backend.

Three CI jobs build that way -- "Linux / Python + PyTorch validation",
"Linux / Python suite under AddressSanitizer" and "Windows / MSVC" -- because
`cmake --preset release` does not set VKML_VULKAN and none of them passes it.
A developer with a GPU never sees that configuration by accident, so a test
that assumes `has_vulkan` passes locally and fails on three platforms at once.

That is not hypothetical. Two tests added on 2026-07-29 asserted a Vulkan-only
error message and called `init_vulkan`, which is only bound when the backend is
compiled in. Both passed locally and broke all three jobs.

WHY THIS IS A SCRIPT AND NOT A SECOND BUILD DIRECTORY. Both configurations link
their extension to the SAME path, `python/vkml/_vkml_core*.so` -- deliberately,
because that is where `import vkml` looks and where scikit-build-core stages a
wheel (bindings/CMakeLists.txt). So the two builds overwrite each other, and
after running this the tree must be put back. Doing that by hand is a trap:
`cmake --build build/release` afterwards sees the timestamp and decides there
is nothing to do, leaving the CPU-only extension in place while every later
Vulkan test skips itself and the run looks fine.

This script therefore saves the extension, swaps it, and restores it in a
`finally`. Exit code 0 if the suite passes, 1 otherwise.

    python scripts/check_cpu_only_build.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "python" / "vkml"
BUILD_DIR = REPO / "build" / "cpu-only"

# The interpreter to build and test against.
#
# NOT simply sys.executable. This gate is normally invoked as
# `python scripts/check_cpu_only_build.py`, which on a machine with a project
# virtualenv is usually the SYSTEM python -- and the system python has no
# nanobind, so CMake refused to configure and the gate failed with
# "nanobind not found" rather than anything about the CPU-only build. A gate
# that goes red for a reason unrelated to what it checks is a gate people learn
# to ignore.
#
# Prefer the project virtualenv when there is one, matching what
# scripts/mutation_check.py does, and fall back to the running interpreter.
_VENV_CANDIDATES = (REPO / ".venv/bin/python", REPO / ".venv/Scripts/python.exe")
PYTHON = str(next((p for p in _VENV_CANDIDATES if p.is_file()), Path(sys.executable)))


def extension_files() -> list[Path]:
    """The built extension, whatever this interpreter names it.

    Globbed rather than spelled out: the suffix carries the Python version and
    platform (`_vkml_core.cpython-314-x86_64-linux-gnu.so`), and hard-coding one
    would silently save nothing on a different interpreter -- which would look
    like success right up to the point the tree was left broken.
    """
    return sorted(PACKAGE.glob("_vkml_core*"))


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(str(c) for c in command)}", flush=True)
    return subprocess.run(command, cwd=REPO, **kwargs)


def build_cpu_only() -> bool:
    """Configures and builds with the Vulkan backend off. True on success."""
    configure = run([
        "cmake", "-B", str(BUILD_DIR),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DVKML_BUILD_PYTHON=ON",
        "-DVKML_BUILD_TESTS=OFF",
        "-DVKML_BUILD_BENCH=OFF",
        # Pinned to PYTHON above. Without it CMake picks whichever Python it
        # finds first, which on a machine with a virtualenv is usually the wrong
        # one, and the build then fails on a missing nanobind that is installed
        # in the venv it did not choose.
        f"-DPython_EXECUTABLE={PYTHON}",
    ])
    if configure.returncode != 0:
        return False
    return run(["cmake", "--build", str(BUILD_DIR), "-j"]).returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true",
                        help="leave the CPU-only extension in place instead of restoring")
    args = parser.parse_args()

    saved = extension_files()
    if not saved:
        print("no built extension in python/vkml -- build one first", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as stash:
        for path in saved:
            shutil.copy2(path, Path(stash) / path.name)
        print(f"saved {len(saved)} extension file(s); they will be restored afterwards\n")

        try:
            if not build_cpu_only():
                print("\nCPU-only BUILD failed", file=sys.stderr)
                return 1

            # Assert the swap actually happened. Without this the suite could run
            # against the Vulkan extension and report a green that means nothing
            # -- the exact vacuous pass this script exists to prevent.
            probe = run([PYTHON, "-c",
                         "import sys;sys.path.insert(0,'python');import vkml;"
                         "print(vkml.has_vulkan)"],
                        capture_output=True, text=True)
            if probe.stdout.strip() != "False":
                print(f"\nexpected a CPU-only extension, got has_vulkan="
                      f"{probe.stdout.strip()!r}{probe.stderr}", file=sys.stderr)
                return 1

            print("\nrunning the Python suite against the CPU-only build\n")
            passed = run([PYTHON, "-m", "pytest", "tests/python", "-q"]).returncode == 0
        finally:
            if args.keep:
                print("\n--keep: leaving the CPU-only extension in place")
            else:
                for path in extension_files():
                    path.unlink()
                for name in (Path(stash)).iterdir():
                    shutil.copy2(name, PACKAGE / name.name)
                print("\nrestored the original extension")

    print("\nCPU-only suite PASSED" if passed else "\nCPU-only suite FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
