# Session handoff

Written when a session's working context ran out mid-stream. Delete it once the
next session has absorbed it — it is a note, not a document.

## Exact state

`main` is **17 commits ahead of `origin/main`** and has not been pushed. Working
tree clean.

Green: `ctest` on release · 1550 Python tests · 1544 at `VKML_MIN_SPEC=1` ·
the CPU-only suite · all 17 gates, 11 with an automated control · the site at
61 pages, 113 documented members, every link resolving.

---

## What this session did, in one line each

**Two latent CI failures, both from `f595d73`, neither visible from any report.**
The clang-format gate had been red for fifteen commits — it was a `find | xargs`
inside `ci.yml`, so `check_gate_coverage` could not see it and `verify_gates`
had no gate to name. And `import vkml` failed on a CPU-only build, because the
observability bindings were inside `#ifdef VKML_HAS_VULKAN` while the Python
layer exported them unconditionally with a comment saying why it should. Three
CI jobs build CPU-only. **Both were found by running the gates locally, not by
reading anything.**

**#99 closed: attribution.** `vkml.attribution` joins Decision to Measurement on
`DispatchId` and partitions a step's wall time. Two producer changes were needed
first, because the fact model could not carry the question — `ProfileEntry`
became an *interval* (`start_ms`), and the recorder can *retain* submissions
rather than only the last.

**Three slices of P1, each found by re-running that measurement rather than by
predicting the next one** (`docs/adr/0006` §10, §11, §12):

```
                     at P0   optimiser   backward   batched assign
  submissions/step     39        25          15            8
  step wall        13.57 ms  12.09 ms   11.71 ms      10.07 ms
  host and driver   42.0%     35.7%      33.7%         24.0%
  GPU / wall         0.58      0.64       0.66          0.76
```

No kernel changed and every result is bit-identical.

**Then three kernel changes, and all three were the same surprise**: they looked
memory-bound and were **addressing-bound**. The diagnosis needs a comparison
against an equal-traffic kernel — `relu` — which is cheap and was not being done.

`docs/adr/0010` — `reduce.comp` launched one workgroup per output, so a weight
gradient folding 64 samples into 73,728 values launched 73,728 workgroups to do
64 loads each. Throughput tracked the workgroup count, not the bytes: 1.6 GB/s
at four elements per output. A second lane-per-output structure, chosen on
coalescing and occupancy, gives **up to 83× on the kernel**.

`docs/adr/0011` — `im2col`/`col2im`, then `max_pool2d`. `relu` moving the same
bytes ran at 228–235 GB/s while `im2col` managed 29–44, because it performed
nine integer divisions per output element; a 3×3 `col2im` performed
*thirty-six*, and a 2×2 pool backward about twenty. Specialising the geometry
lets the compiler strength-reduce every one, unroll the loops, and delete the
`x % stride` test where stride is 1.

```
                     start  optimiser  backward  assign  reduce  unfold   pool
  step wall        13.57 ms  12.09 ms  11.71 ms 10.07 ms 8.87 ms 8.07 ms 8.04 ms
  GPU busy /20         —        —         —        —    128.4 ms 112.1 ms 105.2 ms
  host and driver    42.0%    35.7%     33.7%    24.0%   30.1%   30.4%   34.3%
```

**13.57 ms → 8.04 ms, a 1.69× end-to-end speedup**, every result bit-identical.
`matmul` at 30.9% is now the largest line **by a factor of five**.

Note the host share *rising* while the step shrinks — the same host cost against
a smaller step. A percentage has two moving ends; read the milliseconds.

**P1 module completeness: 28 → 31 of 34.** `PositionalEncoding`, `Conv1d` and
DataLoader transforms built; the list is now *generated* by
`scripts/check_module_coverage.py` rather than asserted.

---

## Findings worth not re-deriving

- **`PROJECT-SCOPE-ANALYSIS`'s P1 table was wrong three ways** — `Attention`,
  `TransformerBlock` and `FeedForward` existed under torch's spellings five days
  before the audit. The document had already confessed the same mistake one row
  down, for the losses. The cause is permanent: vkML uses torch's names on
  purpose and the manifesto uses its own. The mapping is declared and gated now.
- **A class naming another class in its `see` list rendered nothing.** The filter
  was `if t in PAGE_OF`, which holds operators only. `MultiheadAttention` pointed
  at `TransformerEncoderLayer` for its whole life and the link never existed.
  Fixing it took prose links 101 → 142 and un-orphaned fourteen class pages.
- **"Vectorise, never loop in Python" lost, measured.** Both DataLoader
  transforms were 3–4× *slower* vectorised than as loops, because the vectorised
  form does the work for samples it discards. Output byte-identical.
- **Fewer submissions is not the same as faster.** An intermediate optimiser arm
  removed seven submissions and was slower; the saving appeared only once both
  passes batched.
- **"It only moves memory" is not a reason to believe a kernel is memory-bound.**
  Three in a row were not. The check is one line: run an equal-traffic kernel
  with trivial addressing — `relu` — and compare. It cost nothing and reframed
  three optimisations.
- **A 12% regression that did not exist** was nearly accepted on a 25 µs
  dispatch whose *unchanged* control varied by 32%. The frozen-baseline A/B
  (rule 7) at sizes where the kernel dominates its bracket is what disproved it.
- **A roadmap item can often be tested before it is built.** Item 2's whole
  premise was answerable in one sweep because the alternatives were already
  behind environment switches. It cost twenty minutes and removed a High-scored
  item from the plan.
- **Barrier-separated dispatches do not report disjoint intervals.** Brackets
  open at `TOP_OF_PIPE` and close at `ALL_COMMANDS`, so consecutive ones overlap
  by 0.2–4 µs — a fixed cost per boundary. Quantified in `MEASUREMENT-AUDIT` §3b.
- **The whole validation suite runs eager.** 1,456 tests could not distinguish
  two versions of the backward rules that differ only in lazy mode.
  `test_lazy_execution_gives_the_same_gradients_as_eager` now covers it, on both
  backends.
- `train()`'s `compute` bucket is host wall time, not GPU time. Reading "96.3%
  compute" as "96.3% arithmetic" produced a wrong conclusion in two documents.
- **Disproven:** an earlier handoff said `assets/derived/` is not copied into
  `_site`. It is — `build.py:1426` — and all five referenced assets resolve.

## Three things tried and rejected, with the reason left at the code

- **Realising the root inside `backward()`** removes one submission per step and,
  on `sum(a @ b)`, adds the whole forward — 4 dispatches to 6, caught by
  `test_backward_emits_no_degenerate_reductions`. The reasoning is in
  `src/autograd/autograd.cpp` beside the code that would do it.
- **Batched uploads** are worth 0.065 ms — 4% of a batch-64 step — and need a new
  public API. Deferred with the number in `EXTENSIBILITY-ROADMAP` §4a P1.
- **DataLoader prefetch (#22) is not a DataLoader problem.** Measured: the
  bindings hold the GIL through the GPU wait (17–24% available to another
  thread), so no producer thread can overlap. Releasing it needs a threading
  contract — `Recorder` is not thread-safe — which is ADR-sized. Recorded in
  `python/vkml/data.py`.

---

## Next concrete step — a decision, not an implementation

**P1's exit criterion is met and spent.** The discrete and integrated GPUs
separate on MNIST above batch 256 (1.47× at 512, 1.72× at 1024) and cannot
separate at 64, where the step is 0.61 ms of arithmetic. It was a good criterion
— it stayed red through three real fixes — and it is satisfiable by changing the
batch size, so a successor should name a **step at a fixed shape**: host and
driver below 20% of a batch-64 MNIST step, say.

So the open question is what comes next, and the profile now answers most of it:

1. **`matmul`, at 30.9% and the largest line by a factor of five.** This is now
   `M3_ROADMAP`'s territory, and **that document's premise has been measured
   for the first time** — two tables are appended to it.

   Where the headroom is:
   - **Two of the CNN's four GEMMs are memory-bound** (intensity 7.3 and 19.1
     against the ~24 flop/byte crossover). Every tiling item aims at the
     compute roof and can do nothing for them.
   - **The compute-bound two run at 23–24% of peak where the same kernel
     reaches 34–37% on square shapes.** The headroom is ~1.5×, not the 3–4×
     "23% of peak" suggests — 37% is this kernel's demonstrated ceiling here.

   And **two things it is not**, both measured rather than assumed:
   - **Not tile geometry. Item 2 (runtime shape dispatch) is DISPROVEN for this
     workload** — scored High/Low/4-of-5, and testable today because the tile
     alternatives are already behind env switches. 32×32 wins on four of five
     shapes and ties the fifth; every alternative is 1.1–2.0× *worse*. Best
     possible per-shape selection: **1.00×**. In hindsight, every M in the
     workload is a multiple of 32.
   - **Not occupancy.** `gemm_reg` uses 41 VGPRs of 1024 per SIMD with nothing
     spilled — the 20-wave cap, already reached.

   That leaves the K loop and the shapes themselves (K of 288/576 vs 1024, plus
   a batch axis the benchmark lacks), which is items 6/10/13 and is unmeasured.
2. **Convolution's structure, and #101 is verified.** The weight-gradient path
   materialises a per-sample gradient — **18.9 MB for one layer of one step** —
   before reducing it. ADR 0010 made that reduction 20× faster, which was the
   cheap half; the expensive half is that the intermediate exists at all.
   Folding the batch into the GEMM's K axis removes it. Implicit GEMM (CUTLASS
   is cloned) would delete `im2col` rather than speed it up.
3. **Remaining P1 completeness.** Conv3d needs a genuinely 3-D `im2col` and does
   not compose. **Autograd checkpointing needs an autograd extension point that
   does not exist** — `apply_backward` is a closed switch over `OpKind`, so a
   user-defined backward has nowhere to go. That is an ADR before it is code.
4. **The R-series** (#119–#123): release verification, mutation coverage, a
   performance regression gate, and a public claim generated from measurement.

## The one open decision, unchanged

**#114.** Nothing defines "releasable". `PHASE2-MANIFESTO.md` is authoritative
and defines scope, but gives objective criteria only for P1. Three tasks (#108,
#110, #112) have "a scope decision" as their only closing condition and cannot be
resolved from the repository.
