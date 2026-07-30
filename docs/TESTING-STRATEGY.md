# Testing strategy — why a green suite missed 26 bugs, and what changed

Two rounds of testing on a Windows machine with an AMD driver produced 26 issues
against a codebase whose entire suite was green. That is not 26 unrelated
defects, and treating it that way would fix 26 things and prevent none.

This is the classification, the meta-cause, and the rules that follow from it.
It is written from the actual issue list rather than from principle, so every
claim here points at something that really happened.

---

## 1. The classification

| | Class | Issues | Count |
|---|---|---|---|
| **A** | A limit of the DEVELOPMENT MACHINE recorded as the contract | 2, 6, 14, 20, 21 | 5 |
| **B** | Driver-DEFINED behaviour trusted as if it were specified | 3, 26 | 2 |
| **C** | POSIX and GCC assumed to be universal | 7, 8, 10, 13, 15 | 5 |
| **D** | The configuration CI builds is not the one users build | 5, 9, 16, 17 | 4 |
| **E** | A gate that could not fail | 4, 11, 14 | 3 |
| **F** | An input domain nothing ever fed | 25, 26, 27 | 3 |
| **G** | Error and doc quality on a path only users reach | 12, 18, 22, 23 | 4 |

**A + B is 7 issues saying "this environment behaves this way, therefore every
environment does".** C + D is 9 more saying "the one environment we build is the
only one there is". Together that is **16 of 24 distinct issues — two thirds —
sharing a single meta-cause**: the project tested one environment and wrote that
environment's properties down as truth.

The suite was not weak. It was *blind in a direction*, and no amount of adding
tests in the direction it already looked would have helped.

---

## 2. The four rules

### Rule 1 — Test the CONTRACT, not the machine

The most expensive class (A) has one shape every time:

```cpp
static_assert(sizeof(WherePush) <= 256, "...");   // 256 is what THIS GPU reports
```

Vulkan guarantees 128. The assertion could never fail here, on any build, ever.
The same shape produced #20 (`maxComputeWorkGroupCount[x]`: guaranteed 65535, this
GPU reports 2³²−1), #21 (workgroup invocations: guaranteed 128, requested 256)
and #6 (subgroup range assumed to be RADV's).

> **A limit must be asserted against what the specification guarantees. When the
> local machine is more generous than the guarantee, that is exactly when the
> check matters and exactly when it cannot fire.**

Implemented:

- Push-constant blocks assert against `kGuaranteedPushConstantBytes` (128), with
  the three known-oversized ones pinned separately by name so a *new* oversized
  block is a compile error rather than a discovery.
- `choose_dispatch_grid()` takes the device limits **as parameters** rather than
  reading them from the Context, so `tests/cpp/test_dispatch_grid.cpp` exercises
  the guaranteed floor on hardware that reports far more.

### Rule 2 — A test that needs exotic hardware protects nobody

The fix for #20 shipped with seven tests. **Four of them skip on every machine
this project is developed on**, because reaching the limit requires a device
that imposes it.

That is not a criticism of those tests — it is the reason the rule exists. The
geometry they check is pure integer arithmetic, and pure arithmetic can be
tested anywhere.

> **When a rule depends on a device property, separate the rule from the device.
> Pass the property in. Then the rule is testable on every machine, and the
> hardware-dependent part shrinks to the one line that reads the property.**

`test_dispatch_grid.cpp` compiles **unconditionally** — no device, no Vulkan
build — so it runs in the three CPU-only C++ CI jobs and the Windows job as well
as the lavapipe one. Those are precisely the configurations where #20 was missed.

### Rule 3 — Where a specification says "implementation-defined", do it yourself

Class B is small and expensive. Three instances, one shape:

| | The built-in | What happened |
|---|---|---|
| #3 | `float16_t(x)` | SPIR-V leaves `OpFConvert`'s rounding mode implementation-defined. RADV rounds to nearest even, AMD Windows toward zero |
| #26 | `tanh(x)` | GLSL's definition is a ratio of exponentials; evaluated literally it is NaN above \|x\| = 88.72 |
| *(new)* | `0.5 * x * w` in gelu | GLSL permits reassociation of non-precise expressions; the grouping the driver chose overflowed to inf |

The gelu one was found by the test written for #26 and had never been reported.
Note what it means: **parenthesising an expression is not a fix**, because the
implementation is entitled to regroup it.

> **A determinism guarantee cannot be built on a built-in whose edge behaviour
> the specification declines to pin. Implement it, in a form with no
> implementation-defined step, and verify it against the CPU oracle.**

`f32_to_f16_bits` does the narrowing in the integer domain. `tanh_op` clamps the
input before the built-in can overflow. `gelu_op` returns the answer directly in
the region where the intermediate would exceed `FLT_MAX`.

### Rule 4 — Prove a gate can fail before trusting it

Class E is the one that costs the most confidence, because the suite reports
success. Three instances:

- **#14** — `PipelineCache` checked `shared_memory_bytes <= limit`, and the
  probes never set the field, so it evaluated `0 <= limit` and passed while the
  shader asked for 48 KiB on a 32 KiB device.
- **#11** — `check_layering.py` matched zero files on Windows and reported
  success.
- **#4** — the gemv probes declared a 44-byte push-constant range for a 108-byte
  block; RADV tolerated it for months.

And it recurs in the work fixing it. Writing this round:

- The first dispatch-grid index-space assertion was **wrong** — it expected the
  grid to cover 2³² invocations when the largest safe grid covers 4,294,901,760.
  The code was right and the test was not.
- The first `tanh` precision test **could not fail**: moving the clamp from 10.0
  to 5.0 left every sampled point either untouched or already within the gate.
  It needed 5.5 and 6.0, the last values further from 1.0 than the tolerance.
- `test_tanh_saturates_instead_of_overflowing` **is vacuous on RADV** and says so
  in its own docstring, because RADV's built-in already saturates.

> **Break the thing a new gate guards and watch it go red. If it cannot be made
> to fail on the available hardware, say so in the test itself.**

---

## 3. What a tolerance cannot say

Issue #26 is a NaN where a number belongs. No tolerance expresses that: NaN is
not *far from* 1.0, it is a different kind of answer.

The extreme-value suite therefore splits the contract in two:

- **Categorical**, asserted exactly and on every driver: a finite input must
  never produce NaN, and NaN/infinity classification must match the CPU.
- **Numeric**, asserted against the project's own gate from ARCHITECTURE.md §7.3
  (atol = rtol = 1e-5).

Getting the second one wrong is instructive. The first draft used `atol =
FLT_MIN`, far stricter than the documented policy, and failed on 16 of 17
operators. Being wrong found two divergences nothing had recorded:

- **The GPU flushes f32 subnormals to zero.** Vulkan permits this unless
  `shaderDenormPreserveFloat32` is requested, and vkml does not request it. So
  `relu(1e-45)` is 0 here and `1e-45` on the CPU. It changes an answer
  *categorically* in exactly one place: `sign()` compares a subnormal as zero and
  returns its argument instead of ±1.
- **The CPU underflows gelu's negative tail** to zero below about −6, while the
  GPU retains the true value of order 1e-21. torch agrees with the CPU.

Neither is a bug. Both are now pinned, so the next person to meet one finds a
test explaining it rather than a mystery.

---

## 4. Chosen inputs beat random ones at the edges

Every operator test drew from a normal distribution. `tanh` breaks above 88.72.
A random sweep lands there essentially never — which is precisely how #26
survived a suite of over a thousand tests.

`_extreme_values()` is a *chosen* list: ln(FLT_MAX) and its neighbours, the f32
limits, the smallest normals, the subnormal floor, signed zeroes, both
infinities. Twenty-six values found a bug that a million random ones would not.

---

## 5. Standing checklist for a new kernel

1. Does every limit it touches assert against the **guaranteed** minimum?
2. Can its rules be tested **without** the device — and if so, are they?
3. Does it depend on a built-in whose edge behaviour is implementation-defined?
4. Have you **broken** each new gate and watched it fail?
5. Does it get fed NaN, ±inf, subnormals and the f32 extremes?
6. Does it build and run in the **CPU-only** configuration CI actually uses?

---

## 6. Still open

- **#2 / #19** — three push-constant blocks over the 128-byte guarantee. Decided
  in `docs/adr/0009`, not implemented; blocks MNIST on a minimum-spec device.
- **#21** — every pipeline requests a 256-invocation workgroup against a
  guaranteed floor of 128. Unfixed: halving it moves occupancy and the GEMM tile
  geometry is tuned around the current width, so it needs a benchmark.
- **#27** — NaN semantics for `relu`, `amax`/`amin` and `sign` are a policy
  decision, not a defect to be patched.
- **#25** — `prod` and non-contiguous `max_pool2d` have no Vulkan kernel.
