# Verification Audit (P2, part 1)

**Status:** the measurement exists and its blocking gaps are closed.
**Date:** 2026-07-28.
**Objective (`PHASE2-MANIFESTO.md`, P2):** every operator tested for forward,
backward, edge cases, empty tensors, broadcasting, mixed precision, layouts,
large and random tensors.

Every milestone so far added tests for the thing it built. That answers *is the
new code tested* and never answers *what is not tested* — and 1,009 passing tests
made the second question feel answered. It was not asked.

---

## 1. The method

Reading test names cannot answer it: a test called `test_add` may exercise one
shape, and `conv2d` dispatches `im2col` and `matmul` under a name that mentions
neither. So execution is **recorded** instead, at the one point every evaluated
node passes through (`realize`, `src/dispatch/executor.cpp`), and reported
against the operator inventory read out of the enum itself.

    VKML_COVERAGE=1 python -m pytest tests/python -q
    python scripts/coverage_matrix.py tests/python/coverage.tsv

Each node contributes its operator, backend, dtype, rank, size class, and
whether any *source* was broadcast or non-contiguous. Recording costs nothing
measurable — 20.99 s against 21.93 s for the suite, inside run-to-run noise — and
is off unless the variable is set.

**What this does not measure**, stated because a coverage number that overstates
itself is worse than none: it records what was **executed**, not what was
**asserted**. Only three gates together mean anything, and the project now has
all three — `mutation_check.py` shows the suite *can* fail, the assertion audit
(`MILESTONE-B-REVIEW.md` §3.1) shows no test asserts nothing, and this shows what
the suite reaches.

`docs/COVERAGE.md` is the generated matrix, committed as a baseline. Its value is
the diff: a gap that appears between two revisions is a path some change stopped
exercising, which a green suite cannot show you.

---

## 2. The headline finding: one dtype computes

**`DType` declares five types. Arithmetic exists for one.**

The recording showed f16 in 7 of 123,105 node evaluations, all of them as the
*output of a cast* — no arithmetic operator had ever run on it. i32 and i64 were
the same. Probed directly rather than inferred:

| dtype | works | raises |
|---|---|---|
| f32 | everything | — |
| f16 | comparisons, cast | `add`, `sub`, `mul`, `neg`, `sum`, `where`, … |
| i32 | comparisons, cast | as above |
| i64 | comparisons, cast | as above |

```
DTypeError: cpu backend: op 'add' does not support dtype f16
```

**This is a capability gap wearing a coverage gap's clothes, and no test closes
it.** P2's "mixed precision" requirement cannot be satisfied by writing tests.

It is also the Milestone B finding repeating in a different enum. There, twenty
`OpKind` enumerators were declared with no implementation, and the resolution was
to implement nine and delete eleven. `DType::F16` is the same shape of promise:
the device reports `fp16 true` and `16-bit storage true`, `conftest.py` defines an
`fp16` tolerance, and nothing computes.

Two consequences already visible, both verified:

- The `Cast` backward rule **cannot be exercised at all**. Reaching it needs a
  graph that casts and then computes; casting f32→f32 creates no node, and
  casting anywhere else yields a tensor no operator accepts. It is the one
  backward rule still unfired, and it is blocked rather than untested
  (`test_cast_backward_is_unreachable_because_only_f32_computes`).
- `cast` is the one operator never given a non-contiguous input, for the same
  reason.

**This needs a decision, not more testing** — implement f16/integer arithmetic,
or narrow the enum to what exists. Same choice as Milestone B's, and the
manifesto's rule that an enum entry is a promise applies unchanged.

---

## 3. Gaps found and closed

| Gap | Before | After |
|---|---|---|
| Operators never executed | 0 of 63 | 0 |
| Backward rules never fired | **5** of 47 | **1** (blocked, §2) |
| Operators never given a strided input | **15** | **3** (2 impossible, 1 blocked) |
| Operators never run across workgroups | **9** | **0** |
| Operators on one backend only | 1 | 1 (`prod`, §4) |

77 tests added: 1,009 → 1,086.

**Backward rules.** `pow`, `clamp`, `scatter_add` and `col2im` had rules that had
never run. All four now have gradient tests against PyTorch. Writing them found
that `pow`'s gradient *with respect to the exponent* raises deliberately —
`ln(x)` is undefined for x ≤ 0, so the rule declines rather than returning a
number wrong on half its domain. That refusal is now pinned as a contract.

**Layouts and scale.** `tests/python/test_layout_and_scale.py` gives fifteen
operators transposed inputs and sizes above one workgroup, **on both backends**.

---

## 4. `prod` breaks the correctness chain

`ARCHITECTURE.md` §7 makes correctness a chain: CPU against PyTorch for
semantics, then Vulkan against CPU for kernel bugs. `prod` runs only the first
half, because there is no GPU kernel for it. Known since Milestone B; now
measured, and pinned by `test_prod_has_no_vulkan_kernel` so that implementing it
makes the test fail — which is the prompt to re-enable the parametrised runs.

---

## 5. Findings

Each cost real time, and none was predictable from reading code.

**The backends disagree about what they accept, visibly to users.**
`VulkanBackend::supports` requires a contiguous `src[0]` for `MaxPool2d` and
`MaxPool2dBackward`; the CPU kernel has no such requirement. An unsupported op
raises rather than falling back, so `max_pool2d(x.transpose(2, 3))` **computes on
the CPU and hard-fails on the GPU**. This is a concrete instance of the open
question carried since Milestone B — fall back by splitting the graph, or state
that Vulkan is all-or-nothing — and it now has a one-line reproduction
(`test_strided_max_pool2d_is_refused_on_vulkan`). A silent divergence between
backends is the worst of the available answers.

**Covering a path on one backend says nothing about the other, and the report
hid that.** The first version of the layout tests ran on the CPU only. A mutation
that broke `operand_offset` in `shaders/common.glsl` — the Vulkan strided
indexing — survived them completely. Split by backend, the recording showed
strided inputs reaching 51 operators on the CPU and 37 on Vulkan. The tests now
run on both, and the same mutation kills seven of them.

**A size class taken from the output is the wrong measure for a reduction.** The
tool first classified each dispatch by its output element count, so a reduction
over 1,517 elements producing 37 was recorded as a single-workgroup run — exactly
backwards, since the span is what the axis exists to detect. The tests were
right and the tool was wrong. It now classifies by the largest tensor touched,
input or output.

**Two of the flagged gaps were impossible rather than missing.**
`max_pool2d_backward` and `slice_backward` can never receive a strided input,
because the backward rules take `grad.contiguous()` before constructing them.
The right response was not to weaken the report but to pin the precondition: the
CPU kernel asserts only that its *output* is contiguous, so a strided input would
be read wrongly rather than rejected, which makes that copy load-bearing.

**A tool that reports gaps needs its own false-positive discipline.** The first
run flagged `input` and `const` as never executed — they are leaves, whose values
are supplied rather than computed, so the executor prunes them by construction.
It also reported `im2_col` unfired while `im2col` ran, because converting camel
case to snake case guesses wrong on names containing digits. Both were the report
crying wolf on meaningless cells, which is how a real gap hides. Fixed by asking
the extension for each operator's category, and by comparing names normalised
rather than converted.

---

## 6. Carried forward

| # | Item | Trigger |
|---|---|---|
| 26 | **Decide f16 and integer arithmetic: implement, or narrow `DType`** | §2. Needs a decision |
| 27 | Reconcile the backends' input requirements, or document the divergence | §5. Sharpens item 16 |
| 28 | Implement `prod` on Vulkan, or state it is CPU-only | §4 |
| 15 | Run the Python suite under a sanitizer in CI | Unchanged; still open |
| 16 | CPU fallback via graph splitting, or Vulkan all-or-nothing | Now has a reproduction (§5) |
| 29 | Make the coverage report a CI gate against a baseline of accepted gaps | Once §2 is decided; a ratchet is only useful when the accepted set is stable |

Empty tensors, dtype and rank coverage are reported as **data** rather than
judged, because applicability varies per operator — a triangular mask has no
rank-1 case, a matmul has no bool case — and a report that flags meaningless
cells is one nobody reads twice. Turning those into judged gaps needs a per
operator applicability model, which is item 29's real cost.

---

## 7. Gate

**P2 asks for every operator tested across eight properties. The measurement now
exists, and the gaps it found are closed except where closing them requires a
decision.**

Two of P2's eight properties — mixed precision, and the backward half of
`Cast` — cannot be reached until §2 is resolved. That is why this is *part 1*:
P2 is not complete, and the reason it is not complete is a capability gap rather
than missing tests.

All gates green: layering (56 files) · clang-format · debug `-Werror` · release ·
ASan build and suite · ctest · **1,086 Python tests, 5 skipped** · **24 of 24
mutations killed**.
