# Session handoff

Written when a session's working context ran out mid-stream. Delete it once the
next session has absorbed it — it is a note, not a document.

## Exact state

`main` ahead of `origin/main`. Working tree clean. ctest green, 1449 Python
tests pass, all doc gates and the 15 verification-dashboard gates green, site
rebuilds to 58 pages / 112 documented members with every link resolving.

## Two latent CI failures found and fixed, both from `f595d73`

Neither was visible from the repository's own reports, and both had survived
fifteen commits.

- **The clang-format gate had been red since `f595d73`.** Bisected: `f595d73~1`
  exits 0, `f595d73` exits 123, every commit after it 123. It was a
  `find | xargs` inside `ci.yml`, so `check_gate_coverage.py` could not see it
  and `verify_gates.py` had no gate to name. Fixed in `cba4dc1`, then made a
  script with a control in `49f1a3b`.
- **`import vkml` failed on a CPU-only build.** `configuration`,
  `record_decisions`, `decisions`, `decisions_published` and
  `stop_recording_decisions` were bound inside `#ifdef VKML_HAS_VULKAN` while
  `python/vkml/__init__.py` exported them unconditionally with a comment saying
  why it should. Three CI jobs build CPU-only. Fixed in `b78bc49`.

Lesson worth keeping: **both were found by running the gates locally rather than
by reading anything.** The repository's reports all said green.

## #99 is closed, and it corrected #100

`vkml.attribution` joins Decision to Measurement on `DispatchId` and partitions
a step's wall time. A CIFAR step now prints a per-kernel table with the
remainder shown — `EXTENSIBILITY-ROADMAP.md` §4a P0's exit criterion.

Two producer changes were needed first, because the fact model could not carry
the question: `ProfileEntry` became an **interval** (`start_ms`), and the
recorder can **retain** submissions rather than only the last.

Measured, and it changes the roadmap: **host and driver is 44.8% of a CIFAR
step, not the ~74% the batch-scaling inference implied.** Still the largest
single item, larger than `matmul` at 17.1%, so P1's sequencing holds with a
lower ceiling. Two things only direct attribution could say — 39 submissions per
step of which 11 carry no compute, and GPU idle inside submissions is 0.4%.

## Next concrete step

**#100, P1 — dispatch and submission overhead.** The candidates are re-ordered
in `EXTENSIBILITY-ROADMAP.md` §4a P1 by what P0 measured. The cheapest first
item is now the 11-in-39 submissions that carry no compute at all: uploads and
the download behind `.item()`, each paying full submission cost for a copy.

Then #118 (8 unmet P1 modules), #119-123, #124.

## Discoveries worth not re-deriving

- **Barrier-separated dispatches do not report disjoint intervals.** Brackets
  open at `TOP_OF_PIPE` and close at `ALL_COMMANDS`, so consecutive ones overlap
  by 0.2–4 µs — a fixed cost per boundary, proportionally largest where the
  dispatches are smallest. Quantified in `MEASUREMENT-AUDIT.md` §3b.
- `train()`'s `compute` bucket is host wall time, not GPU time. Reading "96.3%
  compute" as "96.3% arithmetic" produced a wrong conclusion in two documents,
  now corrected in both.
- `env_value()` in `src/util/env.cpp` is the project's only `getenv`.
- `vulkan_backend.cpp` has exactly one `set_label` call site.
- A control can rot exactly like a gate: adding `pages.yml` silently disarmed
  `check_gate_coverage`'s control, and `verify_gates` caught it.
- The site's `assets/derived/logo-256.png` returns 404 — `web/build.py` does not
  copy `assets/derived/` into `_site`. `og-card.png` and the favicon are likely
  the same. **Not yet fixed.**
- Two gates that would have caught earlier UI defects still do not exist: a CSS
  selector matching zero elements, and raw `<pre>` in `web/content/` bypassing
  the renderer.

## The one open decision

**#114.** Nothing defines "releasable". `PHASE2-MANIFESTO.md` is authoritative
and defines scope, but gives objective criteria only for P1. Three tasks (#108,
#110, #112) have "a scope decision" as their only closing condition and cannot be
resolved from the repository.
