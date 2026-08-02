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

## P1's exit criterion turned out to be about the batch size

It exits when the discrete and integrated GPUs stop tying on MNIST. They do —
**above batch 256**, and cleanly:

```
  batch    discrete 36 CU   integrated 6 CU
     64        2.18 s           1.99 s      tied (integrated faster)
    128        1.12 s           1.20 s      tied
    256        0.62 s           0.67 s      tied
    512        0.43 s           0.63 s      1.47x
   1024        0.25 s           0.43 s      1.72x
```

At batch 64 it cannot separate and no submission work will make it. That step
is **0.61 ms of arithmetic** against 0.89 ms of host; removing every one of its
four non-compute submissions is worth ~0.13 ms measured. Every size above
halved from this work; the tie at 64 did not move, and could not.

The criterion was well chosen — it stayed red through three real fixes — and it
is now spent, because it is satisfiable by changing the batch size. A successor
should name a **step at a fixed shape**: host and driver below 20% of a batch-64
MNIST step, say.

## #118's premise is partly disproven

"8 unmet P1 modules" was measured by name grep against an API that deliberately
uses **PyTorch's spellings**. Three of the eight existed five days before the
audit was written:

```
  Attention / MultiHeadAttention  ->  nn.MultiheadAttention      (c12622b)
  TransformerBlock                ->  nn.TransformerEncoderLayer (c12622b)
  FeedForward                     ->  inside the encoder layer, as torch keeps it
```

`PROJECT-SCOPE-ANALYSIS.md` had already confessed this exact mistake once, for
the losses, in the same table — and made it again in the row above. The cause is
permanent: the manifesto uses one vocabulary and the code another, on purpose.

`scripts/check_module_coverage.py` now **declares** the mapping and fails when a
declared target stops resolving. **28 of 34 present, 6 real gaps:** Conv1d,
Conv3d, PositionalEncoding, DataLoader prefetch (#22), DataLoader transforms
(#23), autograd checkpointing.

`PositionalEncoding` was the interesting one — it stood between the existing
attention/encoder modules and a transformer that assembles end to end — and it
is **now built**: `nn.PositionalEncoding`, sinusoidal, Vaswani §3.5.

Sinusoidal and not learned, deliberately: learned positions are
`Embedding(max_len, d_model)` indexed by position, which already composes.
Adding the one that cannot be built from the parts and leaving the one that can
is the rule the backward rules follow.

No torch class to compare against — `torch.nn` has no `PositionalEncoding` — so
the oracle is the **closed form in float64**, which is stronger than a torch
class rather than weaker: the table has an exact definition and no
implementation freedom. Eleven tests, five verified by breaking the thing they
guard.

**`nn.Conv1d` too**, composed from `Conv2d` with a height of 1 — identical
arithmetic, so a second GLSL kernel would be a second implementation of one
algorithm. Validated against *torch's* Conv1d, not against vkML's Conv2d:
comparing a composition to the thing it composes from proves the reshapes are
consistent and nothing about whether the result is a 1-D convolution.

**DataLoader transforms too** (#23), which the `data.py` docstring had deferred
with an explicit trigger — *"revisit when augmentation is wanted"* — that the
CIFAR example's own comment had already met. `DataLoader(transform=...)` takes
`f(rng, arrays) -> arrays`; `Compose`, `RandomHorizontalFlip` and `RandomCrop`
are the standard CIFAR pipeline, and `examples/cifar100 --augment` uses them.

The generator is **passed in, never reached for**, so a transform cannot quietly
become irreproducible — `nn.manual_seed` exists because layers called an
unseeded `default_rng()` and a divergence could not be re-observed.

**31 of 34 P1 modules present.** The three left are Conv3d (needs a genuinely
3-D `im2col`; does not compose), DataLoader prefetch (#22) and autograd
checkpointing.

### #22 is reshaped, not deferred again — and it is not a DataLoader problem

`data.py` deferred prefetch on a measurement: batch production was 0.2% of a
step. With transforms it became 21.2%, then 10.6% once the transforms were made
3.7× faster:

```
  no augmentation             batch  1.7%   transfer 5.8%   compute 92.6%
  --augment, first version    batch 21.2%   transfer 4.9%   compute 73.9%
  --augment, after the fix    batch 10.6%   transfer 6.1%   compute 83.3%
```

**The blocker is the GIL, and it is in the bindings, not the loader.** Measured
by spinning a counter thread and comparing its rate:

```
  GIL available, main thread sleeping         100%
  ...during the augmentation                 ~100%   numpy drops it, as hoped
  ...during realize() and its GPU wait      17-24%   nanobind does not
```

A producer thread would get about a fifth of the window it needs. Releasing the
GIL around the blocking calls would help every threaded use — and it is not a
one-line change: `Recorder` is not thread-safe, so it needs a stated threading
contract first. ADR-sized, belongs to the runtime, and 10.6% of an epoch is not
what justifies it.

### "Vectorise, never loop in Python" was wrong here, measured

Both shipped transforms were written vectorised on that rule and both were
**slower than the loops they replaced**, because the vectorised form does the
work for samples it then discards:

```
  crop   1.305 ms -> 0.342 ms   3.8x   64 contiguous slices beat a fancy-index gather
  flip   0.231 ms -> 0.070 ms   3.3x   reverse only the chosen rows, not all of them
```

Byte-identical output — the draws are unchanged, so the same seed gives the same
crops. The comments at each site now record the measurement rather than the
assumption.

Found while documenting these: **a class naming another class in its `see` list
rendered nothing** — the filter held operators only, so `MultiheadAttention`
pointed at `TransformerEncoderLayer` for its whole life and the link was
dropped on the way out. Fixing it took the site from 101 prose links to 142 and
un-orphaned fourteen class pages. `docs_graph.py --check` is what surfaced it,
by reporting the two new pages as orphans.

## Next concrete step — a decision, not an implementation

Eight submissions per CIFAR step: 2 uploads, 1 backward, 3 optimiser, 2 for
`.item()`. **`matmul` at 23.7% is now the largest line in the table.**

**Candidate 4 is measured and deferred.** The `.item()` half has a fix that was
tried and rejected — adding the root to `backward()`'s realise removes one
submission and, on `sum(a @ b)`, adds the whole forward (4 dispatches to 6,
caught by `test_backward_emits_no_degenerate_reductions`); the reasoning is in
`src/autograd/autograd.cpp` beside the code that would do it. The upload half is
worth **0.065 ms — 4% of a batch-64 step** (0.171 ms for two uploads against
0.104 ms for one of the same total bytes, minimum of 300 warm repeats) and needs
a new public API to collect it.

So the open question is whether P1 continues at all:

1. **Command-buffer reuse** (candidate 5) attacks the per-submission cost
   itself, which is ~0.5 ms of MNIST's 0.89 ms host time. Unmeasured.
2. **`M3_ROADMAP`'s GEMM work is no longer mis-sequenced.** The argument for
   deferring it was that arithmetic was a quarter of the step; on CIFAR it is
   now three quarters, and `matmul` alone is 23.7%.

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
