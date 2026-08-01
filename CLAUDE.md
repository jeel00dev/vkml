# vkML — orientation

A Vulkan-first machine-learning framework in C++20, with a Python API. Compute
shaders only, no graphics. Two backends — CPU and Vulkan — and the CPU one is
the correctness oracle the GPU is checked against, not a fallback.

This file is a **map**, not a manual. Everything here is either a command you
can run or a pointer to where the real answer lives. It is deliberately short,
because it is loaded into every session and long files crowd out the work.

---

## Read these first

| If you are | Read |
|---|---|
| About to change anything | `.claude/skills/cpp_spec/SKILL.md` — the engineering constitution: how to decide, what to verify, how to report. Its `references/handbook.md` has the detailed C++/testing/performance rules |
| Contributing | `CONTRIBUTING.md` — issues, commits, PRs, review |
| About to push | `docs/PRE-COMMIT-CHECKLIST.md` — every item exists because something got through without it |
| Wondering why the design is like this | `docs/ARCHITECTURE.md`, then `docs/adr/` |
| Touching numerics | `docs/THEORY.md` — error bounds, pairwise summation, stability |
| Claiming a speedup | `docs/MEASUREMENT-AUDIT.md` — **not optional**; it lists the instruments that lie |
| Touching the docs site | `docs/DOCUMENTATION-PRINCIPLES.md` — how it is built, and the three categories: generated, observed, reimplemented |
| Adding or changing a test | `docs/TESTING-STRATEGY.md` — why a green suite missed 26 bugs, and the four rules that came out of it |

**The skill and the docs are the source of truth.** If this file disagrees with
them, they win and this file is stale — say so.

---

## Build and run

```sh
# Linux and macOS. Presets: debug release relwithdebinfo asan
cmake --preset release -DVKML_VULKAN=ON
cmake --build build/release -j$(nproc)

# Windows. The presets set CMAKE_BUILD_TYPE, which a multi-config generator
# ignores -- `cmake --preset release` succeeds there and then builds Debug.
cmake -B build/msvc -DVKML_VULKAN=ON -DVKML_BUILD_PYTHON=ON
cmake --build build/msvc --config Release --parallel

pip install -e .
```

```sh
ctest --preset release                        # C++ suite -- Linux and macOS
ctest --test-dir build/msvc -C Release        # C++ suite -- Windows
python -m pytest tests/python -q              # Python + PyTorch validation
python scripts/hardware_report.py             # what this machine's GPU reports
```

Shader compilation needs `glslc` or `glslangValidator` from the Vulkan SDK.
PyTorch is a test-only dependency: it is the oracle, never a runtime one.

---

## Layout

```
include/vkml/   public headers          shaders/   24 compute shaders (GLSL)
src/            implementation          bindings/  nanobind Python extension
python/vkml/    the Python package      tests/     cpp/ and python/
docs/           design + decisions      scripts/   gates and tooling
bench/          measurement harness     examples/  mnist, cifar100 (end-to-end)
```

### Layering is enforced, not advisory

`scripts/check_layering.py` fails CI if a lower layer includes a higher one:

```
util 0 · core 1 · graph 2 · backend/api 3 · backend/cpu 4 · backend/vulkan 4
dispatch 5 · plan 5 · api 6 · autograd 7
```

---

## Traps that have already cost time

Each of these produced a real bug or a wasted afternoon. They are the reason
this section exists.

- **`cmake --preset release` does NOT enable Vulkan.** The presets set build
  type and warnings only. Three CI jobs build CPU-only, so a test that assumes
  `has_vulkan` passes locally and breaks three jobs at once. Check with
  `python scripts/check_cpu_only_build.py`.
- **clang-format is pinned to 18.1.8** (`pip install clang-format==18.1.8`).
  CI runs that version; your distro's is probably different and *will* disagree.
- **Scripts prefer the project `.venv`.** The system Python has no nanobind, so
  running a gate with the wrong interpreter fails for reasons unrelated to what
  it checks.
- **This machine's limits are not the contract.** The largest single source of
  bugs in this project: push-constant budgets, subgroup ranges and f16 rounding
  were all written against what the development GPU reports. Assert against what
  **Vulkan guarantees**, and if you cannot, say which device you verified on.
- **pytest captures stderr.** A `fprintf` probe that prints nothing is not
  evidence of anything until you have run it with `-s` and seen it fire.

---

## Verification

Beyond the two suites, these exist and are worth knowing about:

```sh
python scripts/check_layering.py           # layer dependencies (CI gate)
python scripts/check_cpu_only_build.py     # the config CI builds and you don't
python scripts/mutation_check.py           # are the tests capable of failing?
python scripts/coverage_matrix.py          # operator coverage vs baseline
```

`mutation_check.py` is the one people skip and shouldn't. A green suite proves
the tests ran, not that they *can* fail — see `docs/MEASUREMENT-AUDIT.md` rule
10. Breaking the thing a new test guards, and watching it go red, is the
standard here rather than an extra.

Determinism is a hard invariant: identical inputs give identical outputs, bit
for bit, on the same device — and across drivers where the contract says so.
This is why f32→f16 narrowing is done in software rather than left to
`OpFConvert`, whose rounding mode SPIR-V leaves implementation-defined.

---

## Where the work is

Current: 115 C++ test cases, ~1402 Python tests, 11 CI jobs green.

Open, in rough priority:

1. Deferred performance work — see `docs/M3_ROADMAP.md`.

**Issue #2 is closed.** Every push-constant block now fits the guaranteed 128
bytes, by per-op repacking rather than ADR 0009 §3's device buffer: `where` and
`softmax` store shared extents once, `cat` derives its operands' extents from
the output's. §3 stays unimplemented and un-needed — see `docs/adr/0009` §2a for
why that is a reprieve rather than a solution to the general problem.

**Issue #21 is closed.** The general workgroup width is now
`min(256, maxComputeWorkGroupInvocations)` and `matmul` falls back to the naive
kernel where the blocked ones do not fit, so a minimum-spec device can train
rather than fail at pipeline creation. **`VKML_MIN_SPEC=1` reports the Vulkan
1.3 Required Limits on any device**, which is how that is now tested — run it
before claiming a limit is satisfied, because it is the cheapest way to find the
next instance of this project's most common bug.

**A rebuild is not necessarily what `import vkml` gives you.** `cmake --build`
writes `python/vkml`; after `pip install -e .` the extension resolves from
site-packages instead. `check_cpu_only_build.py` now asks where it resolves and
swaps both (issue #29), but the suite does not — so if a change appears to have
no effect, check `python -c "import vkml; print(vkml._vkml_core.__file__)"`
before believing it.

Development happens on RADV (AMD, Mesa). **Reports from other drivers are the
main way device-specific assumptions get found**, and most recent portability
fixes exist because somebody ran it on hardware this project does not have.

---

## Conventions

Commits, PRs and issue hygiene are in `CONTRIBUTING.md` §5–6. The short version:
one logical change per commit, explain *why* in the body, say what you verified
and what you could not, and keep tool attribution out of the history.
