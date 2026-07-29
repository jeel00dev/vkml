# Pre-commit validation checklist

What to run before pushing, and — more importantly — **why each item is on the
list**. Every entry here exists because something got through without it.

The question this list is built around is not *"does my change work?"* but
**"what existing functionality could this change unintentionally affect?"**

---

## 1. Both build configurations, not just yours

```sh
cmake --build build/release -j$(nproc)          # your usual build
python -m pytest tests/python -q
./build/release/bin/vkml_tests

python scripts/check_cpu_only_build.py          # the one you never see
```

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
find include src tests/cpp bench/cpp bindings -name '*.h' -o -name '*.cpp' \
  | xargs clang-format --dry-run --Werror
```

## 7. After pushing

Watch the run. A platform-specific failure is a real failure — it means the
change depends on something your machine happens to provide.

```sh
gh run list --limit 1
gh run view <id> --log-failed
```

Two macOS failures are **expected and pre-existing**, and the job is
`continue-on-error` for that reason: the runner's "Apple Paravirtual device"
reports valid timestamp bits and then never advances them. See the comment on
`test_profiler_reports_nonzero_for_a_real_dispatch`, which deliberately does not
skip, because a skip would blind it to the regression it was written for.

---

## When something gets through anyway

Add the check that would have caught it, rather than only fixing the instance.
That is how sections 1, 4 and 5 above got here.
