# Pre-commit validation checklist

What to run before pushing, and — more importantly — **why each item is on the
list**. Every entry here exists because something got through without it.

The question this list is built around is not *"does my change work?"* but
**"what existing functionality could this change unintentionally affect?"**

---

## 1. Both build configurations, not just yours

```sh
# Linux and macOS
cmake --build build/release -j$(nproc)          # your usual build
./build/release/bin/vkml_tests

# Windows
cmake --build build/msvc --config Release --parallel
.\build\msvc\bin\Release\vkml_tests.exe

python -m pytest tests/python -q
python scripts/check_cpu_only_build.py          # the one you never see
```

The test binary sits under a per-configuration subdirectory on a multi-config
generator, which is why the Windows path has `Release\` in it and the POSIX one
does not.

**Why.** `cmake --preset release` does **not** set `VKML_VULKAN` — the presets
set only the build type and warning flags (`docs/adr/0007`, and the comment in
`CMakeLists.txt`). Three CI jobs build that way: *Linux / Python + PyTorch
validation*, *Linux / Python suite under AddressSanitizer*, and
*Windows / MSVC*. A developer with a working GPU never meets that configuration
by accident.

**What it caught.** Two tests added on 2026-07-29 asserted a Vulkan-only error
message and called `init_vulkan`, which is only bound when the backend is
compiled in. Both passed locally and broke all three jobs at once. The script
reproduces it in about a minute.

**A rebuild is not necessarily what you then test.** `cmake --build` writes
`python/vkml/_vkml_core*`, but after `pip install -e .` — which README.md tells
you to run — a scikit-build-core meta-path finder resolves `vkml._vkml_core`
from site-packages instead, and the two are different binaries. So a C++ change
can be compiled, the suite rerun, and the *previous* extension tested, with
nothing saying so. The failure looks like a fix that does not work, or worse a
broken change that appears to pass.

Check which one you are testing whenever a result surprises you:

```sh
python -c "import vkml; print(vkml._vkml_core.__file__)"
```

If that is not the file you just built, rerun `pip install -e .` before
believing the suite. `check_cpu_only_build.py` asks this question itself and
swaps both copies (issue #29), but the suite does not.

**Where this bites, specifically.** Almost every Vulkan-touching test is marked
`requires_vulkan`, and that marker calls `vulkan_ready()`, which checks
`has_vulkan and vulkan_available()` — so it already screens out a CPU-only
build. 25 call sites across `test_invariants.py` and `test_vulkan_kernels.py`
are safe for free.

**`tests/python/test_device_report.py` is the exception, deliberately.** It
carries no such marker anywhere, because its contract is to answer *"on a device
the backend cannot use, on a machine with no GPU, and on a build with no Vulkan
at all"* — its own docstring. Anything added there has to handle both builds
explicitly. Both tests that broke CI were added to that file, which is the one
place the safety net does not extend.

`check_cpu_only_build.py` swaps the extension and restores it afterwards.
Do not do that swap by hand: both configurations link to the same
`python/vkml/_vkml_core*.so`, and rebuilding the Vulkan one afterwards sees the
newer timestamp and does nothing — leaving a CPU-only extension in place while
every Vulkan test quietly skips and the run still looks green.

## 2. What depends on the code you changed

Not just the file you edited. Ask who reads it.

* Changed an **error message**? Something asserts on it. `grep` the string.
* Changed a **predicate or a public name**? Check both backends, the bindings,
  and the examples.
* Changed **anything in `serialize`, `optim` or `nn`**? The examples use all
  three at once, which is what section 3 is for.

## 3. The examples, end to end

```sh
python examples/mnist/train.py    --epochs 1 --train-size 2000 --no-compare
python examples/mnist/train.py    --model cnn --epochs 1 --train-size 2000 --no-compare
python examples/cifar100/train.py --epochs 1 --train-size 2000 --no-compare
```

**Why.** They exercise the whole stack in one go — device selection, autograd,
the optimiser, checkpoint save — in combinations no unit test covers. They are
also the first thing a new user runs, so a break here is the most visible kind.

Their `.vkml` outputs are gitignored, so this does not dirty the tree.
CIFAR-100 needs its dataset downloaded first (see README).

## 4. Benchmarks, if performance could have moved

Only when performance is the point, and then to `docs/MEASUREMENT-AUDIT.md` §7 —
frozen baseline as the control arm, minimum not mean, validation off.

**On this machine, take the minimum across several PROCESS runs.** The GPU parks
at 400 MHz of a possible 1500, the clock state persists for a whole process, and
a single run of either arm can land in the wrong mode. A within-process minimum
is not enough; this produced an apparent 2.4x regression in code that could not
have caused one (`docs/adr/0006` §7).

## 5. Documentation, examples, public API

* Did an error message you **quoted in a doc** change? Re-run it and paste the
  real text. Three README outputs drifted from the code this way in one day,
  each caught only by running the command.
* Did you add a public function? The README and the module docstring are where
  someone looks first.
* Did you change a **default**? Say so where the default is documented.

## 6. The gates CI runs anyway

```sh
python scripts/check_layering.py
python scripts/check_push_constants.py
python scripts/check_docs_references.py
python scripts/check_docs_examples.py     # needs a built extension
python scripts/check_versions.py
python web/build.py && python scripts/check_source_links.py

# POSIX shells, including Git Bash on Windows. CONTRIBUTING.md sec8 has the
# PowerShell equivalent -- `find` and `xargs` do not exist there.
find include src tests/cpp bench/cpp bindings -name '*.h' -o -name '*.cpp' \
  | xargs clang-format --dry-run --Werror
```

`check_push_constants.py` reads the GLSL and fails if any push-constant block
exceeds the 128 bytes Vulkan guarantees. It is the cheapest gate here -- no
compiler, no Vulkan, no device -- and it guards the defect class of issue #2,
where three blocks over the limit produced 19 failing tests on a driver
reporting exactly 128 and nothing at all on a development GPU reporting 256.
The C++ structs carry `static_assert`s against the same bound; this covers the
shader side, and the two can drift apart.

`check_docs_references.py` verifies that every file, `path:line` and constant
the documentation cites still exists. It was written after a page claimed
`pairwise_sum` lives in `iterate.h` when it lives in `reduce.h` -- the page
rendered, the links resolved, the examples ran, and the sentence was false.

`check_docs_examples.py` executes every `>>>` transcript in `web/content/` and
compares the printed output to what the page claims. It needs a built
extension, so run it after building rather than before.

`check_versions.py` checks every version number stated in prose against the file
that decides it -- the CMake minimum against `cmake_minimum_required`, the Python
minimum against `requires-python`, and so on. It was written after the
get-started page claimed CMake 3.24 while `CMakeLists.txt` required 3.25, with
README stating the right figure and nothing noticing that the two disagreed.

`mutation_check.py` needs to know where your build is when it rebuilds a mutated
kernel. It defaults to `build/release`, so on Windows point it at the build the
instructions above create:

```powershell
$env:VKML_BUILD_DIR = "build/msvc"; $env:VKML_BUILD_CONFIG = "Release"
python scripts/mutation_check.py
```

Without that every compiled mutation reports NOT-TESTED, and the campaign covers
only the Python half.

## 7. After pushing

Watch the run. A platform-specific failure is a real failure — it means the
change depends on something your machine happens to provide.

```sh
gh run list --limit 1
gh run view <id> --log-failed
```

**Every job in `ci.yml` is expected to pass.** There are no known-red jobs to
read past — macOS was the last one and it now lives in
`.github/workflows/macos-moltenvk.yml`, off the push pipeline and run by hand
(`gh workflow run macos-moltenvk.yml`). If something in `ci.yml` is red, it is
telling you something.

**A green job is only evidence about what it compiled.** Both Windows jobs used
to pass `-DVKML_BUILD_BENCH=OFF`, so `bench/cpp` was the one directory MSVC
never saw. `harness.h` used GCC inline asm for months: the build README.md
documents failed on a clean checkout while CI stayed green, because the job was
not building the target the instructions name (issue #13, then #16).

So when a job turns an option off, that option's code is untested rather than
tested-and-fine. Before trusting a green run, check that the flags it passes
still match the ones the documentation tells a contributor to use — and if a job
must diverge, another job should cover what it drops.

---

## When something gets through anyway

Add the check that would have caught it, rather than only fixing the instance.
That is how sections 1, 4 and 5 above got here.
