# Milestone B — Operator Completeness

**Status:** complete. **Date:** 2026-07-28.
**Objective (`PHASE2-MANIFESTO.md`, P1):** implement every operator the enum declares.

Milestone B set out to close a gap the Phase 2 audit found: twenty `OpKind`
enumerators declared with no implementation anywhere. That list is now empty.

---

## 1. Where the operator set stands

| | Phase 2 start | Now |
|---|---|---|
| Declared with no implementation | **20** | **0** |
| Vulkan implementations | 16 | **55** |
| CPU implementations | 47 | **57** |
| Compute operators declared | 67 | 57 |
| Python tests | 560 | **917** |

Vulkan covers **55 of 57** compute operators. The two exceptions are `Prod` and
`SliceBackward`, both CPU-only and neither on a training path.

The declared count *fell* from 67 to 57. That is the interesting number, and §2
explains it.

---

## 2. The ledger: how twenty operators were resolved

Nine got a kernel. Eleven were resolved by **deleting the enumerator** — the
operator exists at the API, built from operators that were already there.

**Implemented as kernels (9)**

| Operator | Why it needed one |
|---|---|
| `Cat` | Index remapping across two sources; no elementwise form |
| `Triu`, `Tril` | The predicate is *positional*, not value-based (one shared kernel) |
| `IndexSelect` | Gather along an axis by index vector |
| `ScatterAdd` | Repeated indices ⇒ accumulation; irreducible per `ARCHITECTURE.md` |
| `Im2Col` | Window extraction |
| `Col2Im` | Overlapping windows ⇒ accumulation; irreducible |
| `MaxPool2d`, `MaxPool2dBackward` | −∞ padding and argmax routing (one shared kernel) |

**Resolved by composition, enumerator removed (11)**

| Operator | Composes from |
|---|---|
| `MaskedFill` | `where` with a scalar |
| `LayerNorm`, `RmsNorm` | `mean`, `sub`, `square`, `rsqrt`, `mul` |
| `MseLoss` | `sub`, `square`, a reduction |
| `CrossEntropy` | `log_softmax`, a one-hot mask, `sum` |
| `Conv2d` | `im2col` + `matmul` + bias broadcast |
| `AvgPool2d` | `im2col` + `mean` |
| `BatchNorm` | `sub`, `rsqrt`, `mul`, `add` |
| `Dropout` | `rand`, `greater_equal`, `where` |
| `SgdStep`, `AdamStep` | Neither: a reservation for work that measurement showed was not needed |

**One operator was added:** `Rand`. `ARCHITECTURE.md` §6 lists it in the op
inventory but the enum never had it, so no random source existed anywhere —
weight initialisation drew from host NumPy. Dropout needed it.

### What the ledger says about the original inventory

**Eleven of twenty declared operators did not need to exist.** The enum was
written as a table of contents from the architecture document, before anything
was built, and it over-predicted: it named operations that turn out to be
compositions of other operations in the same list.

That is not a criticism of the original inventory — you cannot know which
operations compose until you have the ones they compose from. But it is a
finding worth carrying: **an enumerator declared ahead of its implementation is
a prediction, and this project's predictions ran about 55 % accurate.** The
manifesto's rule that an enum entry is a promise is what forced each one to be
resolved rather than left to drift.

`SgdStep`/`AdamStep` are the sharpest case. They were reserved for a fused
optimiser step, and measurement showed the update already ran on the device —
the cost was submissions, driven by explicit `realize()` calls, and a fused
kernel would not have addressed it. The reservation was speculative, and
removing it was the honest outcome (`fbc6d9d`).

---

## 3. Verification: does the suite actually detect defects?

357 tests were added. A green suite proves they ran, not that they *can fail*
(`MEASUREMENT-AUDIT.md` rule 10), so both were checked.

### 3.1 Static audit

Every test function was parsed for an assertion. **0 of 207 assert nothing.**
Three initially flagged use `np.testing.assert_*`, which the first scan missed —
a false positive in the scan, not a gap in the tests. No parametrisation list is
empty.

### 3.2 Mutation campaign

`scripts/mutation_check.py` applies one semantically meaningful mutation per
kernel — an off-by-one, a dropped guard, a reversed fold — rebuilds, and runs the
tests that should catch it. Every mutation compiles, so a kill is evidence about
the tests rather than about the compiler.

**12 of 12 mutations killed.**

| Mutation | Result |
|---|---|
| `tri`: diagonal off-by-one (`>=` → `>`) | KILLED |
| `cat`: reuse the output extent for the source index | KILLED |
| `index_select`: ignore the index vector | KILLED |
| `scatter_add`: reverse the fold order | KILLED |
| `im2col`: drop the padding bounds check | KILLED |
| `col2im`: drop the stride-boundary test | KILLED |
| `max_pool2d`: tie rule picks the last maximum | KILLED |
| `rand`: nine Philox rounds instead of ten | KILLED |
| `philox` (CPU): nine rounds | KILLED |
| `k_col2im`: drop the stride-boundary test | KILLED |
| `k_max_pool2d`: tie rule picks the last maximum | KILLED |
| `k_cat`: reuse the output extent | KILLED |

Each mutation was chosen to be one a plausible implementation could actually
contain, and several correspond to bugs the tests were written for specifically:
the tie rule, the stride-boundary test, and the source-extent trap.

**Scope of the claim.** This shows the suite detects *these twelve* defects. It
is not a coverage proof and does not survive a refactor of the mutated lines —
the script reports `PATTERN-MISSING` rather than passing quietly when that
happens, which is the honest failure mode but still requires maintenance.

---

## 4. Findings

Recorded because each cost real time and none was predictable from the
architecture documents.

**Padding semantics decide composability.** `avg_pool2d` composes from `im2col`;
`max_pool2d` does not. `im2col` pads with zero, which is exactly
`count_include_pad=True` for averaging and exactly wrong for a maximum — an
all-negative window would report 0. Verified against torch rather than reasoned
about. The same question, asked of two operators that look identical in shape,
has opposite answers.

**Determinism forced the same inversion three times.** `scatter_add`, `col2im`
and `max_pool2d_backward` all scatter, all face contention, and none can use an
atomic — the device has no global float `atomicAdd`, and atomic ordering would
not be reproducible anyway. All three invert the loop: one thread per
*destination*, pulling. That is now an established pattern rather than three
independent decisions, and it is why all three carry an `EXACT` tolerance
instead of a numerical bound.

**Tie-breaking is a contract.** `max_pool2d` routes a window's gradient to the
*first* maximum, not split among ties. A random sweep never produces a tie, so
this is only ever caught by a test written for it — and a `>=` where a `>`
belongs moves the gradient silently. The forward and adjoint share one `argmax`
function precisely so the rule cannot drift between them.

**Biased for normalising, unbiased for the running estimate.** torch's
`BatchNorm` uses different variance estimators for the two purposes. Getting it
wrong makes evaluation drift away from training as the running estimate
converges to the wrong value — invisible in a single-step comparison.

**Two implementations of one algorithm need a byte comparison.** The Philox
generator exists twice, in C++ and GLSL. Two different generators would each look
perfectly uniform while disagreeing on every value, so no distributional test can
detect a divergence. The test compares bytes, at sizes straddling the workgroup
boundary — a shader seeding per workgroup rather than per invocation would agree
below 256 elements and diverge above.

**A missing nanobind caster fails at import, not at compile.** Passing
`std::array` across the binding without `<nanobind/stl/array.h>` builds cleanly
and then fails with `ImportError: std::bad_cast` under a wall of leak warnings
that names neither the type nor the argument.

---

## 5. Carried forward

Recorded per P7; none blocks Milestone C.

| # | Item | Trigger |
|---|---|---|
| 15 | Run the Python suite under a sanitizer in CI | A use-after-free in C++ shape inference reached the Python suite as garbage output, because ASan covers only `tests/cpp` |
| 16 | CPU fallback via graph splitting, or state that Vulkan is all-or-nothing | An unported op raises rather than falling back; the design in `ARCHITECTURE.md` §3 Fork 3 was never built |
| 17 | Fuse `layer_norm`/`rms_norm` | A profile showing the extra passes matter |
| 18 | `scatter_add`'s O(n_out × index_len) scan | Large-vocabulary embedding backward |
| 19 | Batch optimiser `realize` calls into one submission | A profile showing submission overhead matters |

---

## 6. Gate

`PHASE2-MANIFESTO.md` P1 asks for every major subsystem required for a usable
framework. The operator layer is complete; `nn` modules, the data pipeline and
serialisation are not, and are Milestones C and D.

**Gate for Milestone B specifically — every declared operator implemented — is
met**, with all eight CI gates green: layering, format, debug `-Werror`, release,
ASan build and suite, ctest, 917 Python tests, validation layers clean.
