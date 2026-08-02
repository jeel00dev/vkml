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

## P1's first slice is taken

The attribution showed the optimiser spending 24 of a step's 39 submissions on
eight parameters. `Optimizer.step()` is now three passes in the base class —
build every parameter's state, realise together; build every new value, realise
together; assign — and all four optimisers implement one `_plan` hook.

**39 → 25 submissions per step, 1.5–1.9× on the optimiser phase across all
seven configurations, parameters bit-identical over 60 steps.** Host and driver
fell from 42.0% to 35.7% of a step. `docs/adr/0006` §10.

The finding worth keeping: an intermediate arm removed **seven** submissions and
was **slower**. Submission count is a proxy, not the objective.

## Next concrete step

**ADR 0006 stage B — the Assign node.** It is now the whole of what remains in
the optimiser: the budget is `2 + N` and the `N` is `assign_`, still eager, one
submission per parameter. It also fixes `nn.BatchNorm2d`'s forward path, which
pays the same cost per layer.

Two ADR-sized changes sit in front of it, both recorded in `docs/adr/0006` §9
and neither started:

1. `detach()` must stop forcing evaluation.
2. An assigned tensor must not retain its history — otherwise each step's Assign
   holds the previous step's graph alive, transitively, for the whole run.

`docs/adr/0007` already separated "bound" from "computed", which was the third
blocker and is done.

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
- **Disproven:** an earlier handoff said `assets/derived/logo-256.png` returns
  404 because `web/build.py` does not copy `assets/derived/` into `_site`. It
  does — `build.py:1426` makes `_site/assets/` and copies every file into it,
  and all five referenced assets resolve in the built site. Checked, not
  inferred. `logo-256.png` is referenced by `index.html` and present.
- **`scripts/check_css_bindings.py` now exists** and covers one of the two gates
  that earlier handoffs asked for: a class the stylesheet does not define
  (renders unstyled) and a rule matching nothing (dead). It found seven dead
  selectors and one dead palette token on its first run. The other requested
  gate — raw `<pre>` in `web/content/` bypassing the renderer — still does not
  exist.

## The one open decision

**#114.** Nothing defines "releasable". `PHASE2-MANIFESTO.md` is authoritative
and defines scope, but gives objective criteria only for P1. Three tasks (#108,
#110, #112) have "a scope decision" as their only closing condition and cannot be
resolved from the repository.
