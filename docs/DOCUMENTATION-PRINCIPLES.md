# How this documentation is built

Ten principles, each of which exists because something went wrong without it.
They are not aspirations; every one names the defect that produced it, because a
principle without a scar is advice and gets ignored.

Run `python scripts/docs_health.py` to see where the documentation currently
stands against them.

---

## 1. Documentation is code

It builds, it has gates, it fails CI, and it carries a health report. A page is
not "done" when it is written — it is done when something checks it.

Nine gates run on every build. Each was written after a specific defect got
through: a page citing `pairwise_sum` in the wrong header, seventeen invented
example outputs, a version number that disagreed with `CMakeLists.txt`.

## 2. Every claim needs a source of truth

Not a source — *the* source. Signatures come from the imported module. Layer
dependencies come from the include scan. The PyTorch compatibility table comes
from the test suite, where each pair is an assertion that the two agree within a
declared tolerance.

If a claim has no source of truth, it is prose, and prose must say who is
claiming it and on what evidence.

## 3. Three categories, in order of preference

This is the sharpest rule here, and it arrived late.

1. **Generated from the source of truth.** Ideal. It cannot drift, because
   drifting would mean the build is broken.
2. **Observed from the real system.** Acceptable when generation is impossible.
   Running the thing and recording what happened is measurement.
3. **Reimplemented logic.** Assume it is wrong until proven otherwise.

An interactive GEMM kernel selector was rejected under this rule. Selection
depends on tensor shape, on `max_workgroup_invocations` read from the live
device, on three environment overrides and on a split-K planner. A JavaScript
version would be a second implementation with no gate able to compare the two —
the hazard `bench/gpu_bench.py` already documents when it moved its GPU-time rule
into the library because "a second copy drifts and starts silently reporting
multiply-counted numbers".

It would also have been wrong instructively: measured here, a 1×512 by 512×1
matmul selects `gemm_reg`, **not** `gemv`. The intuitive model is false, and a
hand-written selector encodes whichever model its author believed.

**"Can it be generated?" is not sufficient.** A thing can be generated from a
hand-written model and still be a lie. Ask whether it can be *driven by* or
*observed from* the source of truth.

## 4. Tooling over maintenance

Every recurring manual task is a bug in the tooling. If a fact has to be updated
by hand when the code changes, that is a defect waiting for the day somebody
forgets.

The support matrix, the compatibility table, the layer diagram, the search index
and the operator cross-references are all derived. Nobody updates them.

## 5. Fix the cause, not the page

Twenty-six of twenty-seven class pages had no prose linking to them. The patch
would have been to add links. The cause was one line: `inline()` cross-referenced
the 102 operators and not the 27 classes, so 65 mentions rendered as dead code
chips.

One renderer change fixed every existing page **and every page not yet written**.
Prefer changes with that shape.

## 6. Progressive disclosure

A newcomer, an advanced user and a contributor read the same page. Structure it
so each can stop where they need to: thirty seconds, five minutes, then the
implementation with file and line.

## 7. Teach, do not merely describe

A reference that lists what exists is a catalogue. Documentation should explain
why the design is what it is — and where the repository already records that
reasoning, link to it rather than paraphrase.

`prod` is CPU-only because a parallel reduction reorders the fold, and for a
product that is a different answer rather than a rounding difference. That
reasoning was in `vulkan_backend.cpp` all along, unreachable from a browser.

## 8. A gate guards a class of mistakes, not one instance

Twice a gate here was narrower than the class it was written for.

The unrendered-markup gate checked `*emphasis*` and meta descriptions, so a
generated heading shipped as ``A lazy `detach()` `` with the backticks intact and
the gate passed. Same class, different delimiter.

The design-system gate used a budget of ten font sizes against a nine-step
scale — so adding a fortieth size passed silently, which is exactly the step that
starts an accumulation. It is now strict.

Ask: does this check the whole class, or only the instance I happened to hit?

## 9. Every gate must be validated

A gate that has never failed is a gate that has never been shown to work. Break
the thing it guards, watch it go red, restore, watch it go green.

This has repeatedly found gates that did nothing. `check_docs_examples` reported
PASS over class-page examples it had never opened; the mutation campaign reported
SURVIVED for seventeen mutations it never executed, because it rebuilt one
extension and tested another.

Watch the verification itself, too. Twice a red test looked green because the
exit code came from `grep` at the end of a pipe rather than from the script.

## 10. Measure before changing, and disbelieve the first number

Most of the improvements here began with a measurement that contradicted the
intuition.

The type scale was going to fix a "flat hierarchy" — the probe had measured a
sidebar heading, and the real hierarchy was already above the peer median. The
link graph put `index` at 56 inbound until breadcrumbs were excluded, after
which the most-cited page was `api-linear-algebra-nn` at 14. Grouping the
sidebar's classes made it *longer*, not shorter, until they were collapsed.

Treat every intuition as a hypothesis, including the one that says the
measurement is right.

---

## What "good" looks like here

Not page count. The measures that matter:

- **the share of content that cannot rot** — derived rather than typed
- **claims with a machine-checked backing** — examples executed, links verified
- **gaps that carry a declared reason**, checked in both directions, so the site
  cannot claim something is unsupported after somebody implemented it
- **how little manual maintenance the documentation needs** while continuing to
  become more accurate

`scripts/docs_health.py` reports all of these.

---

## The instruments

| Tool | What it answers |
|---|---|
| `scripts/docs_health.py` | how much of this could quietly become false |
| `scripts/docs_graph.py` | orphans, dead ends, and which pages everything routes through |
| `scripts/measure_docs.py` | how twelve reference documentation sites make the same decisions |
| `scripts/shoot_docs.py` | what it actually looks like, four viewports, both themes |
| `scripts/check_docs_references.py` | every cited path, constant, heading id and declared gap |
| `scripts/check_docs_examples.py` | every `>>>` transcript, executed and compared |
| `scripts/check_design_system.py` | the type scale, weights and palette |
| `scripts/check_source_links.py` | every deep link into the source tree |

## Rejected ideas are kept

A documented rejection prevents the same investigation six months from now. Each
records why it was considered, why it was rejected, and what would justify
revisiting it — see the UI tracker for the browser playground, animated autograd,
the workgroup visualiser and the GEMM selector.
