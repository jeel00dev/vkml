# Session handoff

Written when a session's working context ran out mid-stream. Delete it once the
next session has absorbed it — it is a note, not a document.

## Exact state

`main` ahead of `origin/main`. Working tree clean. ctest green, 1475 Python
tests pass, the CPU-only suite passes, all doc gates and the 16
verification-dashboard gates green, site rebuilds to 58 pages / 113 documented
members with every link resolving.

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

Measured, and it changes the roadmap: **host and driver was 44.8% of a CIFAR
step, not the ~74% the batch-scaling inference implied** — and is 24.0% now that
the three fixes below have been taken. `matmul` at 23.7% is now the largest
line in the table, which it has never been before. Two things only direct
attribution could say — 39 submissions per step of which 11 carried no compute,
and GPU idle inside submissions is 0.2%.

## Three slices of P1 are taken, each found by re-measuring the last

**The optimiser** (`docs/adr/0006` §10). Attribution showed it spending 24 of a
step's 39 submissions on eight parameters. `Optimizer.step()` is now three
passes in the base class — build every parameter's state, realise together;
build every new value, realise together; assign — with all four optimisers
implementing one `_plan` hook. 1.5–1.9× on the phase across seven
configurations, parameters bit-identical over 60 steps.

**Then `backward()`, which was worse** (`docs/adr/0006` §11). Re-running the
attribution rather than assuming the first fix had finished the job found the
same defect one layer down: the leaf-deposit loop realised one gradient at a
time, and two backward rules — `MaxPool2d` and `Slice` — called `realize()`
unconditionally where every other rule realises only in eager mode. **11 → 1
submission per backward pass**, gradients bit-identical.

**Then the per-parameter `assign_`, and NOT the way the roadmap predicted**
(`docs/adr/0006` §12). It was written down as blocked on two ADR-sized changes.
Re-checking both before starting: §10's ordering had already dissolved the
first — every optimiser now detaches *after* the batched realise, so `detach()`
forcing costs nothing — and the second was never on the path. `assign_` did not
need to become part of the graph; it needed to stop being **one submission per
call**, and only the backend knows what a submission is.
`Backend::copy_device_to_device` takes a span, `vkml::assign(dst, src)` is its
public form, and an optimiser step is now **a constant 3 submissions regardless
of the parameter count**.

```
                     at P0   optimiser   backward   batched assign
  submissions/step     39        25          15            8
  step wall        13.57 ms  12.09 ms   11.71 ms      10.07 ms
  host and driver   42.0%     35.7%      33.7%         24.0%
  GPU / wall         0.58      0.64       0.66          0.76
```

**No kernel changed and every result is bit-identical.** A quarter of the step's
wall time was scheduling.

Three findings worth keeping:

- An intermediate arm removed **seven** submissions and was **slower**.
  Submission count is a proxy, not the objective.
- **The whole validation suite runs eager**, so 1,456 tests could not tell the
  two versions of the backward rules apart. There is now
  `test_lazy_execution_gives_the_same_gradients_as_eager`, bit-for-bit, on both
  backends.
- **Three of the four claims the roadmap's P1 list started with were wrong**,
  and only re-measuring after each change could show it.

## Next concrete step

The optimiser is no longer the largest item and neither is backward. Eight
submissions per step: 2 uploads, 1 backward, 3 optimiser, 2 for `.item()`.
**`matmul` at 23.7% is now the largest line in the table.**

Two readings, and the choice between them is the next decision:

1. **Keep going on P1.** The remaining 4 non-compute submissions are uploads and
   the `.item()` download. Then command-buffer reuse.
2. **`M3_ROADMAP`'s GEMM work is no longer mis-sequenced.** The argument for
   deferring it was that arithmetic was a quarter of the step; it is now
   three quarters, and `matmul` alone is 23.7%.

`docs/adr/0006` stage B still has a purpose, but it is not the optimiser any
more: it is `nn.BatchNorm2d`, which calls `assign_` on the FORWARD pass of every
layer and cannot batch across layers, because each layer's assignment is
separated by the next layer's arithmetic. That is the case only a graph node
fixes, and it is what stage B's argument should be written around.

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
- **`scripts/check_css_bindings.py` now exists** and covers both gates earlier
  handoffs asked for. It found seven dead selectors, one dead palette token, and
  seven code blocks with no copy button on the get-started and concepts pages.
- **The second of those gates was asked for with the wrong framing**, and the
  correction is the useful part: it wanted a rule forbidding raw `<pre>` in
  `web/content/`, on the theory that such a block bypasses the renderer. It does
  not — `highlight_raw_blocks()` has coloured them since it was written. What it
  bypassed was the copy button, and the answer was a renderer pass covering
  every block rather than a rule authors have to remember.

## The one open decision

**#114.** Nothing defines "releasable". `PHASE2-MANIFESTO.md` is authoritative
and defines scope, but gives objective criteria only for P1. Three tasks (#108,
#110, #112) have "a scope decision" as their only closing condition and cannot be
resolved from the repository.
