#!/usr/bin/env python3
"""Run the Python validation suite against an AddressSanitizer build.

WHY THIS EXISTS
---------------
`ctest --preset asan` sanitises `tests/cpp`, and most of the operator surface is
not tested there — it is tested from Python. That gap is not hypothetical: a
use-after-free in `cat`'s shape inference (`src/api/ops.cpp`, where `shape()`
returns by value and iterators into two dead temporaries were being used)
reached the Python suite as garbage output — "rank 43020 exceeds kMaxDims=4" —
and ASan never saw it, because ASan was not in the build the Python suite loads.

This closes that: the same suite, against an instrumented extension.

THE PRELOAD IS NOT OPTIONAL
---------------------------
Python itself is not instrumented, so an ASan-built extension links against a
runtime that nothing has loaded and the import fails with

    undefined symbol: __asan_option_detect_stack_use_after_return

Preloading the runtime before the interpreter starts is the standard remedy and
the only one that does not require rebuilding CPython.

WHAT THIS COVERS, and what it does not. The CPU backend and every host-side
layer above it — graph, shape, api, autograd, the bindings — which is exactly
where the motivating bug was. **Vulkan is deliberately absent**: CI has no GPU,
so those tests would skip regardless, and pointing ASan at a graphics driver
produces noise about code that is not ours. A memory bug inside a compute
shader is not something ASan can see in any configuration.

Usage:
    python scripts/asan_python.py [pytest args...]

Every build writes its extension into python/vkml/ so that an in-tree
`import vkml` works, so this one necessarily displaces whatever was there. It
puts the original back when it finishes, including on failure -- telling the
caller to "just rebuild" does not work, because an unchanged tree gives CMake
nothing to relink and the sanitised extension silently stays in place.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build" / "asan"


def asan_runtime() -> str:
    """Absolute path to the compiler's ASan runtime shared library."""
    compiler = os.environ.get("CXX", "clang++")
    probe = shutil.which(compiler) or shutil.which("clang++") or "clang++"

    result = subprocess.run([probe, "-print-file-name=libclang_rt.asan-x86_64.so"],
                            capture_output=True, text=True, check=False)
    path = result.stdout.strip()
    if result.returncode != 0 or not path or not Path(path).is_file():
        sys.exit(
            f"could not locate the ASan runtime via '{probe}'.\n"
            "It ships with clang; install the matching compiler-rt package.\n"
            f"(asked for libclang_rt.asan-x86_64.so, got {path!r})")
    return path


def symbolizer() -> str | None:
    """`llvm-symbolizer`, if this machine has one.

    Without it ASan still detects and still aborts, but every frame is a raw
    offset into the extension rather than a function and line. Detected rather
    than required, because the detection is the useful half and a missing
    symbolizer should degrade the report, not fail the run.
    """
    found = shutil.which("llvm-symbolizer")
    if found:
        return found
    # clang ships one beside itself on most distributions.
    probe = shutil.which(os.environ.get("CXX", "clang++")) or shutil.which("clang++")
    if probe:
        candidate = Path(probe).resolve().parent / "llvm-symbolizer"
        if candidate.is_file():
            return str(candidate)
    return None


def build() -> None:
    """Configure and build the sanitised extension for THIS interpreter.

    `sys.executable` rather than whatever CMake finds first: the extension has
    to match the interpreter that will import it, and a machine with several
    Pythons will otherwise build against the wrong one and fail at import with
    a message about ABI rather than about configuration.
    """
    configure = ["cmake", "--preset", "asan",
                 "-DVKML_BUILD_PYTHON=ON",
                 f"-DPython_EXECUTABLE={sys.executable}"]
    subprocess.run(configure, cwd=ROOT, check=True)
    subprocess.run(["cmake", "--build", str(BUILD), f"-j{os.cpu_count() or 4}"],
                   cwd=ROOT, check=True)


def run_suite(extra_args: list[str]) -> int:
    env = dict(os.environ)
    env["LD_PRELOAD"] = asan_runtime()

    # detect_leaks=0 deliberately. CPython, numpy and torch all leave
    # allocations live at interpreter exit by design, so LeakSanitizer reports a
    # wall of findings that are not ours and would bury a real one. What this
    # gate is for is use-after-free, buffer overflow and invalid free -- the
    # class that produced garbage output rather than a crash.
    env["ASAN_OPTIONS"] = os.environ.get(
        "ASAN_OPTIONS", "detect_leaks=0:abort_on_error=1:print_stacktrace=1")

    symbols = symbolizer()
    if symbols:
        env["ASAN_SYMBOLIZER_PATH"] = symbols
    else:
        print("note: no llvm-symbolizer found, so any report will give offsets "
              "rather than function names. Detection is unaffected.", flush=True)

    # The suite is noisy at debug level and ASan makes it slower; keep the log
    # to real problems so an ASan report is not lost in it.
    env.setdefault("VKML_LOG_LEVEL", "error")

    # -s is not optional. ASan writes its report to stderr as the process dies,
    # and pytest's capture swallows it -- the run then aborts with exit 134 and
    # no diagnosis at all, which is the least useful possible failure. Verified:
    # with capture on, stderr arrives empty.
    command = [sys.executable, "-m", "pytest", "tests/python", "-q", "-s",
               "-p", "no:cacheprovider", *extra_args]
    print(f"$ LD_PRELOAD={env['LD_PRELOAD']} {' '.join(command)}", flush=True)
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


def installed_extension() -> Path | None:
    """The extension currently in the package directory, if any."""
    found = sorted((ROOT / "python" / "vkml").glob("_vkml_core*.so"))
    return found[0] if found else None


def main() -> int:
    # POSIX only, and deliberately so rather than by oversight. The whole design
    # rests on LD_PRELOAD putting the ASan runtime in front of an uninstrumented
    # CPython, and Windows has no equivalent: MSVC's /fsanitize=address links a
    # runtime DLL that must be resolved at load time, which is a different
    # mechanism needing a different script, not a path fix.
    #
    # Fail here, saying that, rather than three steps later on a clang probe
    # that reports something unrelated (issue #8). NOT SUPPORTED, NOT BROKEN:
    # ctest --preset asan still sanitises the C++ suite on Windows.
    if os.name != "posix":
        print(f"scripts/asan_python.py does not support {sys.platform}: it needs "
              "LD_PRELOAD to\ninsert the sanitiser runtime ahead of an "
              "uninstrumented CPython, which Windows has\nno equivalent for.\n\n"
              "  C++ suite under ASan, all platforms:  ctest --preset asan\n"
              "  Python suite under ASan:              Linux, or WSL",
              file=sys.stderr)
        return 2

    # Preserve whatever is installed, so a developer's fast extension survives a
    # sanitiser run. Restored in `finally`: an ASan run that ends in a report is
    # exactly when the tree must not be left holding a 20x slower build.
    original = installed_extension()
    saved = None
    if original is not None:
        saved = original.with_suffix(original.suffix + ".preasan")
        shutil.copy2(original, saved)

    try:
        build()
        return run_suite(sys.argv[1:])
    finally:
        current = installed_extension()
        if saved is not None and saved.is_file():
            shutil.move(str(saved), str(current or original))
            print(f"\nrestored {(current or original).name} to the build it "
                  "replaced.")
        elif current is not None:
            # Nothing was installed before, so leaving the sanitised build in
            # place would be a surprise of our own making.
            current.unlink()
            print(f"\nremoved {current.name}; there was no extension installed "
                  "before this ran.")


if __name__ == "__main__":
    sys.exit(main())
