# Session handoff

Written when a session's working context ran out mid-stream. Delete it once the
next session has absorbed it — it is a note, not a document.

## Exact state

`origin/main` at `e6343ab`, working tree clean. ctest green, 1423 Python tests
pass (1418 at `VKML_MIN_SPEC=1`), all 19 CI gates green, site live at
https://jeel00dev.github.io/vkml/ with 108 operators at 100% prose.

## Where #99 actually is

**The roadmap's premise was wrong and is now corrected everywhere.** Timestamps
already existed and were already per-node; `vulkan_last_profile()` returns real
per-operation GPU time. What was missing was correlation, not plumbing.

Shipped: `DispatchId` end to end. `CommandRecorder::next_dispatch_id()` mints it,
`Decision.dispatch` and `ProfileEntry.dispatch` carry it, `decisions()` and
`vulkan_profile_records()` expose it. Verified live —
*chose gemm_naive instead of gemm_reg -> 0.1631 ms* — joined from two producers
neither of which knows the other. Three negative controls fail correctly
(omitted id, unrecorded id, off-by-one); a fourth (double increment) does not
fire with a single dispatch, which is recorded in
`OBSERVABILITY-ARCHITECTURE.md` §4b rather than hidden.

## Next concrete step

**#99 slice 2: aggregation.** A consumer that groups `vulkan_profile_records()`
by joined kernel and sums per step, then reports the **unaccounted remainder**
explicitly — total submit time minus the sum of attributed dispatches. The
remainder *is* the overhead #100 hunts, and printing it is this section's own
exit criterion in `EXTENSIBILITY-ROADMAP.md` §4a.

Build it as a **consumer**, not in either producer. It belongs in Python or a
script; neither the profiler nor the planner may learn the other's fact.

Then #100 (the measured 74%), #118 (8 unmet P1 modules), #119-123, #124.

## Discoveries worth not re-deriving

- `env_value()` in `src/util/env.cpp` is the project's only `getenv`. That is why
  configuration observability could be added at one site.
- `vulkan_backend.cpp:1099` is the only `set_label` call site.
- A control can rot exactly like a gate: adding `pages.yml` silently disarmed
  `check_gate_coverage`'s control, and `verify_gates` caught it.
- The site's `assets/derived/logo-256.png` returns 404 — `web/build.py` does not
  copy `assets/derived/` into `_site`. `og-card.png` and the favicon are likely
  the same. Not yet fixed.
- Two gates that would have caught most of this session's UI defects do not
  exist: a CSS selector matching zero elements, and raw `<pre>` in `web/content/`
  bypassing the renderer.

## The one open decision

**#114.** Nothing defines "releasable". `PHASE2-MANIFESTO.md` is authoritative
and defines scope, but gives objective criteria only for P1. Three tasks (#108,
#110, #112) have "a scope decision" as their only closing condition and cannot be
resolved from the repository.
