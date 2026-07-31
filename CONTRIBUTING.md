# Contributing to vkML

vkML is a Vulkan-first machine-learning framework in C++20, meant to stay
correct and maintainable for a long time. That goal, more than any style
preference, is what the rules below are for.

This document is about **process** — how to report something, how to propose a
change, what a reviewable commit looks like. The engineering standards
themselves live in `docs/`, and this file points at them rather than restating
them, because a duplicated rule is a rule that will drift.

| You want to | Read |
|---|---|
| Report a bug | [§1](#1-reporting-a-bug) |
| Propose a feature or a change of direction | [§2](#2-proposing-a-change) |
| Get set up and build | [§3](#3-building) |
| Know what to run before pushing | [`docs/PRE-COMMIT-CHECKLIST.md`](docs/PRE-COMMIT-CHECKLIST.md) |
| Write a commit | [§5](#5-commits) |
| Open a pull request | [§6](#6-pull-requests) |
| Claim a performance improvement | [`docs/MEASUREMENT-AUDIT.md`](docs/MEASUREMENT-AUDIT.md) — **not optional** |
| Write or change a test | [`docs/TESTING-STRATEGY.md`](docs/TESTING-STRATEGY.md) — the failure classes this project actually has, and how to test against them |
| Change an invariant or a public contract | [§7](#7-decisions-that-need-an-adr) |

---

## 1. Reporting a bug

The most useful thing you can tell us is **what you ran and what happened**. A
report we can reproduce gets fixed; a report we cannot only gets questions.

Please include:

1. **What you ran.** The smallest script that shows it. If it needs a model or a
   dataset, say so — often it does not, and finding that out is half the work.
2. **What you expected, and what happened instead.** Both. "It doesn't work" and
   an unlabelled traceback each leave the interesting half unsaid.
3. **The exact error**, copied whole. Not paraphrased, not screenshotted.
4. **Your device report**:

   ```sh
   python scripts/hardware_report.py
   ```

   This is worth more than any other single item. vkML runs on whatever Vulkan
   device is present, and most portability bugs are one device reporting a limit
   or a capability differently from another. The report names the driver, the
   Vulkan version, the subgroup range, the push-constant budget and the memory
   heaps, which is usually enough to see the cause without owning the hardware.
5. **Your platform and how you installed** — OS and version, Python version,
   whether you built from source or installed a wheel.

If you can, run it once with per-dispatch tracing on. It names the kernel and
the shape it was given, which frequently turns "wrong numbers" into a specific
dispatch:

```sh
VKML_VULKAN_DEBUG=1 python your_script.py
```

If you suspect the bug is about a device *limit* rather than a driver, you can
reproduce a minimum-spec device on the hardware you have:

```sh
VKML_MIN_SPEC=1 python your_script.py
```

That reports the Vulkan 1.3 Required Limits — 128 workgroup invocations, 16 KiB
of shared memory, 128 bytes of push constants — instead of what your GPU
actually offers, and runs the whole stack against them. It only ever reports
*less* than the hardware has, so anything that works under it genuinely works.
Every portability defect this project has had was a limit written against one
machine, and this is the cheapest way to find the next one.

### What makes a report especially valuable

- **A device we do not have.** vkML is developed on RADV (AMD, Mesa). Reports
  from NVIDIA, Intel, Apple/MoltenVK, Windows AMD or Adreno are the main way
  device-specific assumptions get found — several fixes exist only because
  someone ran it somewhere we could not.
- **A comparison against PyTorch**, when the complaint is numerical. `vkml gives
  0.31, torch gives 0.29` is actionable; `the numbers look wrong` is not.
- **Whether the CPU backend agrees.** Run the same thing with
  `device=vkml.cpu`. If both backends agree, the bug is above the backends; if
  they disagree, it is in one of them. That single line often halves the search.

### Security issues

Do not open a public issue. Contact the maintainer directly.

---

## 2. Proposing a change

**For anything larger than a bug fix, open an issue before writing code.** Not
bureaucracy — it is cheaper for everyone to disagree about an approach in a
paragraph than in a branch. Say what problem you are solving, not only what you
want to build, because the best solution is often not the one either of us
thought of first.

A proposal is easier to accept when it names both sides:

- **Benefit** — what improves, and by how much if that is knowable.
- **Cost** — what gets worse. Complexity, compile time, a new path to test, an
  option foreclosed.
- **When it is not worth it.** If you cannot name a case, the trade-off has
  probably not been examined yet.

Some changes will be turned down even though they work, and it is fairer to say
so up front than after you have written them:

- **Anything that trades determinism for speed.** Identical inputs must produce
  identical outputs, bit for bit, on the same device — and where the contract
  says so, across devices too. This is why `f32 → f16` narrowing is done in
  software rather than left to the driver.
- **A tolerance loosened to make a test pass.** A mismatch is a bug until it has
  an error analysis saying otherwise. If a tolerance genuinely must move, the
  derivation goes in the test next to it.
- **An abstraction with one implementation and no second one in sight.** Every
  abstraction costs maintenance forever; it should remove real duplication,
  simplify reasoning, or make a wanted extension easier.
- **An optimisation without a profile.** Seeing a loop is not evidence.

---

## 3. Building

```sh
git clone <repo> && cd vkml

# Linux and macOS
cmake --preset release
cmake --build build/release -j$(nproc)

# Windows -- do NOT use the presets; see below
cmake -B build/msvc -DVKML_VULKAN=ON -DVKML_BUILD_PYTHON=ON
cmake --build build/msvc --config Release --parallel

pip install -e .
```

Presets are `debug`, `release`, `relwithdebinfo` and `asan`.

**The presets are for single-config generators only.** They set
`CMAKE_BUILD_TYPE`, which Visual Studio and Ninja Multi-Config ignore in favour
of `--config` at build time. `cmake --preset release` therefore *succeeds* on
Windows and prints `build type ....... Release`, and then `cmake --build` with no
`--config` builds **Debug** — a wrong build that announced itself as the right
one. Use the explicit form above there.

**One thing that surprises everybody:** the presets set the build type and
warning flags only — they do **not** enable Vulkan. `-DVKML_VULKAN=ON` is what
builds the GPU backend, and several CI jobs deliberately build without it. If a
change touches anything Vulkan-adjacent, build both ways; the script in
`scripts/check_cpu_only_build.py` does it for you.

Shader compilation needs `glslc` or `glslangValidator` from the Vulkan SDK.

---

## 4. Before you push

Run [`docs/PRE-COMMIT-CHECKLIST.md`](docs/PRE-COMMIT-CHECKLIST.md). It is short,
and every item on it is there because something got through without it.

The three gates CI will run regardless:

```sh
python scripts/check_layering.py      # layer dependencies
python -m pytest tests/python -q      # Python + PyTorch validation

ctest --preset release                # C++ suite -- Linux and macOS
ctest --test-dir build/msvc -C Release            # C++ suite -- Windows
```

The C++ line differs by platform for the same reason the build command does: the
`release` preset points at `build/release`, which the Windows instructions never
create, and a multi-config generator needs `-C` to know which configuration to
run. See §3.

If your change could affect numerics or performance, the checklist has more —
including running the MNIST and CIFAR-100 examples end to end, which catch a
class of breakage no unit test does.

---

## 5. Commits

**One logical change per commit.** A commit that fixes a bug, renames a
variable and reorganises a header is three commits pretending to be one, and it
cannot be reviewed, reverted or bisected as a unit. Fix the abstraction in one
commit, then build on it in the next.

### The message

```
component: what changed, in the imperative

Why it changed. What was wrong before, or what became possible. This is the
part that is worth writing -- the diff already says what.

What you verified, and what you could not. Numbers if you have them.
```

- **Subject** — lower-case component prefix (`vulkan:`, `shaders:`, `scripts:`,
  `docs:`, `build:`), imperative mood, no trailing full stop, ideally under 72
  characters.
- **Body** — wrapped at 72 characters. Explain the *why*; the diff covers the
  *what*. Reference issues as `#12`.
- **Say what you verified.** "102 test cases, MNIST unchanged at 96.12%" tells a
  reviewer where to look and what not to re-derive. So does "could not test on
  Windows" — an honest gap is worth more than an implied guarantee.
- **Include the measurement** if you claim a performance change, to the rules in
  `docs/MEASUREMENT-AUDIT.md`. Minimum, never mean. Across process runs. Warm
  pipelines. Validation off. A frozen baseline arm.

### Keep the history clean

Commit messages and metadata should read as the project's engineering record.
No tool attribution, no generated trailers, no signatures, no noise — the
history is documentation, and it is read years later by people reconstructing
why something is the way it is.

---

## 6. Pull requests

- **Branch off `main`**, keep the branch focused on one thing.
- **Say what you verified and on what hardware.** Include your device report if
  the change touches the Vulkan backend. Reviewers cannot check every device and
  will not assume you did.
- **Tests come with the change, not after it.** A bug fix should include the
  test that fails without it — write it, watch it fail, then fix it. A test
  never seen to fail is not yet known to test anything.
- **Expect questions about the cost**, not only the benefit.
- **Review your own diff first**, as if someone else wrote it. Most review
  comments are things the author would have caught by reading it once more.

### If a review disagrees with your premise

It happens, and it is not a rejection of the work. Reading the code sometimes
shows that the problem is somewhere else, or already solved, or that the fix
would not achieve the stated goal. When that happens the finding *is* the
contribution — say so rather than completing the original plan so it can be
marked done.

---

## 7. Decisions that need an ADR

Anything that changes an **invariant**, a **public contract**, or something
**expensive to reverse** gets a short architecture decision record in
`docs/adr/`. Read `docs/adr/0009` for the shape: what the defect is, what was
measured before choosing, the decision with its cost, and what implementation
has to handle.

An ADR is not a formality. It is what stops the same question being re-litigated
in a year by someone who cannot tell which of two reasonable options was chosen
deliberately.

---

## 8. Style

**Use the pinned clang-format**, not whatever your distro ships — different
major versions format the same code differently, and CI checks one of them:

```sh
pip install clang-format==18.1.8

# Linux, macOS, and Git Bash on Windows
find include src tests/cpp bench/cpp bindings -name '*.h' -o -name '*.cpp' \
  | xargs clang-format -i
```

`find` and `xargs` are POSIX tools. In PowerShell or a Developer Command Prompt
neither is available — Windows' own `find` is a text search, not a file finder —
so use Git Bash, or PowerShell's equivalent:

```powershell
Get-ChildItem include, src, tests/cpp, bench/cpp, bindings -Recurse -Include *.h, *.cpp `
  | ForEach-Object { clang-format -i $_.FullName }
```

`.clang-format` governs C++ layout and CI enforces it; run `clang-format -i` on
what you touch. Beyond formatting, the standard the codebase is held to is that
**good code does not make the reader think** — their attention should go on the
algorithm, not on decoding the code.

In practice:

- Follow what the surrounding file already does. Two good styles in one codebase
  are worse than either alone.
- One level of abstraction per function.
- Comments explain **why**, not what. A comment restating the code is noise; a
  comment recording why the obvious approach was rejected is worth more than the
  code beside it.
- Names carry one meaning throughout. Vocabulary drift hurts readability faster
  than bad formatting.

And when you touch a file, leave it a little better — a lying comment fixed, a
dead branch deleted. Just not inside an unrelated commit (§5).
