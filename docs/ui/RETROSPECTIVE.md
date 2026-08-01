# Documentation v2 — a technical retrospective

Not a changelog. What worked, what did not, what was believed and turned out to
be false, and what the next version should do differently.

Run `python scripts/docs_health.py` for where things currently stand.

---

## The ideas that produced the largest long-term improvement

**Generating from the mechanism that enforces the rule.** The layer diagram is
built from the same include scan `check_layering.py` uses to fail CI. The picture
and the gate read the same includes with the same parser, so they cannot
disagree: the diagram is wrong only if the build is broken. Every other generated
artifact is a weaker version of this — derived from a source of truth, but not
from something that *already fails when the truth changes*. That is the shape to
reach for.

**Extracting from the tests rather than from prose.** The PyTorch compatibility
table is read from the test suite, where each pair is an assertion that vkML and
torch agree within a declared tolerance. Prose mentioned torch 14 times; the
tests pair 85, and those 41 published equivalences are *proven* rather than
claimed. Documentation extracted from tests is documentation that cannot be
optimistic.

**Bidirectional gates.** The capability gate fails both when a gap has no
declared reason *and* when a reason exists for a gap that no longer does. The
second direction is the one that pays off over years: it stops the site telling a
reader something is unsupported after somebody implemented it, in a table that
looks machine-generated and therefore trustworthy. Nothing else would catch that
— the tests pass, the page builds, the sentence is false.

**Fixing causes at the renderer.** Twenty-six of twenty-seven class pages had no
prose linking to them. The patch was to add links; the cause was that `inline()`
cross-referenced 102 operators and not 27 classes. One change fixed every
existing page and every page not yet written. Improvements with that shape are
worth disproportionate effort to find.

## Assumptions that turned out to be wrong

**"The type hierarchy is flat."** The measurement probe reported `h2` at 1.08×
body. It had measured a *sidebar* heading, because a documentation page's first
`h2` is usually in the navigation. The real figure is 1.60×, above the peer
median. Two things were nearly "fixed" that were already better than the
reference sites.

**"Grouping the sidebar will shorten it."** Grouping 27 classes by kind made it
*longer* — 52 visible links and 3.0 screens against 2.6 — because sixteen
expanded layer classes are as long as the operator list they replaced. Collapsing
all but the current group is what worked: 25 links, 1.75 screens.

**"`index` is the most-cited page."** It showed 56 inbound links. That is the
breadcrumb every page carries. Excluding navigation chrome, prose links fell 151
→ 84 and the genuinely most-cited page is `api-linear-algebra-nn` at 14.

**"An interactive kernel selector would teach well."** It would have encoded a
model of the C++ rather than the rule itself — and the intuitive model is wrong:
measured here, a 1×512 by 512×1 matmul selects `gemm_reg`, **not** `gemv`.

**"The mutation campaign was telling me the tests are weak."** Seventeen
mutations reported SURVIVED. They had never executed: the campaign rebuilt one
extension and pytest imported another. The suite was fine; the tool was lying in
the more damaging direction.

## The gates that found the most, and what that says

Ranked by defects caught:

1. `check_docs_examples` — 17 invented example outputs on its first run, then
   the class-page examples it had never opened.
2. `check_docs_references` — a page citing `pairwise_sum` in the wrong header;
   later, unrendered markup and headings with no ids.
3. `check_design_system` — caught its own author within minutes, twice.
4. `check_layering` — one real violation, autograd reaching into api.

The pattern: **the gates that caught the most were the ones checking the OUTPUT
a reader receives**, not the source that produced it. A check on the input can
be satisfied by a renderer that then does something else.

## What was found by auditing the gates themselves

Three gates were not doing their job, and none of it was visible from a green
run:

- **`check_docs_links` could never fail.** It printed the broken links it found
  and fell off the end of the file, exiting 0 always. It reported PASS in CI and
  in every local sweep for its entire life. Found by deliberately breaking a
  link and reading the exit code.
- **`check_docs_examples` reported PASS over content it had not opened** — it
  walked guide pages and operator prose in two hand-written loops and never
  looked at class-page examples.
- **The mutation campaign tested a binary it had not built** (above).

Two more were narrower than the class they were written for: the
unrendered-markup gate checked `*emphasis*` but not `` `code` ``, and the
design-system gate used a budget of ten sizes against a nine-step scale, so the
*first* new size passed silently.

**Every one was found by breaking something on purpose.** None by reading.

## What generation replaced

| Was maintained by hand | Now derived from |
|---|---|
| operator signatures | the imported module |
| the layer diagram | the include scan the gate runs |
| PyTorch compatibility | the test suite's own assertions |
| support matrix and gaps | extracted facts + declared reasons |
| search index (278 entries) | the build's own page maps |
| heading anchors | the heading text |
| prev/next order | `NAV_SECTIONS`, already declared for the sidebar |
| brand asset sizes | `make_assets.py` from one master |

## Principles that only emerged late

The **three categories** — generated, observed, reimplemented — arrived last,
from rejecting the GEMM selector. Before that the rule was "generate where
possible", which is not sharp enough: *a thing can be generated from a
hand-written model and still be a lie.*

**A gate must guard a class, not an instance** came from fixing the same gate
twice.

## What the tooling audit found

23 modules, ~6,200 lines. Coupling measured by which source of truth each reads:
16 read the imported module, 11 the C++ tree, 10 the test suite, 8 the built
site, 7 the prose.

**`check_docs_references.py` is the outlier**: 479 lines holding *seven*
unrelated checks — path references, native members, unrendered markup, heading
ids, asset sizes, sidebar length, capability reasons. Only the first matches its
name, and it is the only module reading all five sources of truth. It grew that
way because adding a function to an existing gate is easier than adding a gate.
Splitting it is v3 work, recorded rather than done at the end of a long session.

**A simplification deliberately NOT made.** The `<main class="content">` regex
appears in two checkers (`build.py` emits it, so it is not a third parser).
Extracting a shared module to remove one duplicated line would add a module, an
import and a dependency edge — more total complexity than it removes. Recorded
because "deduplicate everything" is a reflex worth resisting when the duplicate
is one line and the coupling is not.

**No dead generators, no stale artifacts.** But two pieces of dead code were
found and removed during v2: `pagenav()` (defined, styled, called nowhere — every
page a dead end for sequential reading) and the `.hero` CSS rules, orphaned by
the landing rewrite.

## What v3 should do

1. **Split `check_docs_references.py`** into gates named for what they check.
2. **A kernel-selection table recorded by observation** — `VKML_VULKAN_DEBUG=1`
   at TRACE prints the real choice. Needs a GPU, and the docs build deliberately
   needs none, so it becomes a recorded artifact with device metadata. Blocked on
   the baseline governance that the benchmark also needs (#71).
3. **Terminology consistency**, now checkable — the graph can compare vocabulary
   across pages.
4. **Close the twenty unlinked class pages**, which is a *writing* gap: those
   classes are never named in prose anywhere. Manufacturing mentions to satisfy a
   graph would be worse than the gap.
5. **Spacing on a scale** (UI-10). Deliberately split from the type scale because
   spacing is a long tail — 42 values, top ten covering 55% — so remapping moves
   layout visibly and needs its own verification pass.
6. **A gate that asserts every gate can fail.** The one class of defect found
   three times in v2 has no automated check, which is an obvious gap now that it
   has been stated three times.

## What I would design differently from the start

**Check the output, not the input.** Every gate that caught a lot checks what a
reader receives. Every gate that missed something was checking a source and
trusting the renderer.

**Write the gate before the content.** The invented example outputs, the
unrendered markup and the missing heading ids all existed *before* the gate that
found them. Each was written, reviewed, and shipped wrong.

**Distrust the first measurement, especially a flattering one.** The three
wrongest beliefs in this project each came from a measurement that was not
scoped correctly, and each would have led to work that made things worse.
