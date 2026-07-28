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
| f16 | cast, and comparisons that *silently returned the wrong answer* | `add`, `sub`, `mul`, `neg`, `sum`, `where`, … |
| i32 | as above | as above |
| i64 | as above | as above |

(The "comparisons work" entry was itself wrong; see *The bug this uncovered*.)

```
DTypeError: cpu backend: op 'add' does not support dtype f16
```

**This is a capability gap wearing a coverage gap's clothes, and no test closes
it.** P2's "mixed precision" requirement cannot be satisfied by writing tests.

> **Correction, added when this was acted on.** The paragraph that stood here
> called `DType::F16` "the same shape of promise" as the twenty undocumented
> `OpKind` enumerators Milestone B resolved. That was wrong, and unfair to the
> design. f16 storage-only was deliberate and recorded: `dtype.h` scheduled
> arithmetic to a later milestone, `Half` is *intentionally* not an arithmetic
> type so nothing can accumulate in 16 bits by accident, `ARCHITECTURE.md` §7.3
> pre-derives the "fp16 storage, fp32 accum" tolerance, the IEEE conversions
> handle subnormals, and both `F16Buf` and `cast.comp` already existed. The
> *finding* below stands — nothing consumed f16, which is why `Cast`'s backward
> rule was unreachable — but "narrow the enum" was never a real option, because
> it would have discarded infrastructure built on purpose. The reason it had not
> happened is that the M-milestone plan deferred it while `PHASE2-MANIFESTO.md`
> P2 requires it; the manifesto is the later document and wins.

Two consequences already visible, both verified:

- The `Cast` backward rule **cannot be exercised at all**. Reaching it needs a
  graph that casts and then computes; casting f32→f32 creates no node, and
  casting anywhere else yields a tensor no operator accepts. It was the one
  backward rule still unfired, and it was blocked rather than untested.
- `cast` was the one operator never given a non-contiguous input, for the same
  reason.

Both are now closed — see *Resolved* below.

### Resolved: f16 computes, integers do not

The decision was to implement f16 for the compute set and leave the integer
types as what they already are. Both halves are now true rather than assumed.

**f16 is a compute dtype on both backends** — elementwise, comparisons,
reductions, softmax, `full`, `where`, `clamp` — following §7.3's contract
exactly: widen to float, compute, narrow once on the store. Every kernel body
was left alone on either side; the dtype decision lives in the handful of
helpers they already routed through (`widen` on the CPU, `load_f`/`store_f` in
the shaders), which is why the f16 and f32 paths cannot drift into different
numerics.

The two backends agree **bit for bit** on every f16 operation, asserted as
equality rather than within a tolerance — both widen, compute and narrow once,
so any extra rounding in a shader is a defect rather than a rounding difference.
That is the second link of the correctness chain, and it is the link that was
missing when this section was first written.

**Matmul followed, and its result contradicted the prediction.** The GEMM family
now computes f16 on the GPU, bit-exact against the CPU across every path the
dispatcher can choose — gemv, naive, tiled, register-blocked, and split-K. That
last one needed a decision: **split-K partials stay f32 whatever the operands
are**, because rounding a partial sum to 16 bits before the final fold breaks
fp32 accumulation across the split exactly as a 16-bit accumulator breaks it
within one. So the GEMM shaders carry two storage types, the operands' and the
destination's, and they differ in precisely that case.

### What f16 buys, measured

The reason to want f16 on a GPU is memory traffic, so it was measured rather
than asserted — Vulkan timestamp queries, warm pipelines, minimum of the
distribution, three independent runs (`MEASUREMENT-AUDIT.md` §7 rules 1, 2, 6).

| elementwise `a * b`, 2^24 elements | gpu min | traffic | achieved |
|---|---|---|---|
| f32 | 0.832 ms | 0.201 GB | 242 GB/s |
| f16 | 0.427 ms | 0.101 GB | 236 GB/s |

**1.95× faster at the same GB/s**, which is the signature of a bandwidth-bound
kernel: identical bytes per second, half the bytes, half the time. Both figures
sit at roughly 84 % of the device's theoretical peak.

The mechanism was confirmed by prediction rather than inferred. If the win is
traffic and nothing else, then f16 at 2^25 elements moves the same 0.201 GB as
f32 at 2^24 and should take the same time. Measured: **0.843 ms against
0.832 ms**, within 1.4 %. Same bytes, same time, whatever the dtype.

The corollary matters for anyone reading a smaller number: at 2^22 the win is
only 1.40×, because halving the traffic also moves the kernel down the
saturation curve. All six points lie on one curve when plotted against bytes
moved. `bench/gpu_bench.py` tracks the 2^24 pair.

### f16 GEMM is slower, and the reason corrects a claim made here

This document previously argued that f16 should pay *more* in a GEMM than in an
elementwise pass, "because a GEMM reads its operands repeatedly, so halving
operand traffic matters more there". **That reasoning was wrong**, and measuring
it is what showed so.

| 2048³ matmul | gpu min |
|---|---|
| f32, vectorised tile load | **6.74 ms** |
| f32, vectorised load disabled | 9.89 ms |
| f16 (vectorised load disabled) | 9.74 ms |

f16 is **1.45× slower than f32**, and none of that is f16's doing. `load4` reads
through `F32Vec4Buf`, so the vectorised tile load is f32-only and f16 falls back
to the scalar path — which was already there and already correct, which is why
it was chosen over writing an f16 vec4 loader on a first pass. Disabling
vectorisation for f32 too reproduces almost exactly the f16 time (9.89 against
9.74), at 512³ and 1024³ as well as here.

At **equal** vectorisation f16 is 2–6 % faster, and that small figure is the
real finding: a tiled GEMM is compute-bound, not bandwidth-bound. The repeated
operand reads the earlier argument appealed to hit *shared memory*, not global —
tiling already solved the bandwidth problem, which is what tiling is for. The
elementwise kernel got 1.95× precisely because it has no reuse to exploit.

**So f16 matmul is worth having for footprint, not for speed**, and it currently
costs speed. That is recorded rather than hidden: `bench/gpu_bench.py` tracks the
pair, and an f16 vectorised load (item 32) would restore parity rather than
deliver a win — its value is removing a regression, and its expected size is now
measured rather than hoped for.

**The integer types are storage and indices, not arithmetic**, and that is now
stated in `dtype.h` rather than discovered at a call site.

**Consequences measured, not assumed:** all 47 backward rules now fire, `Cast`
included — the last blocking gap. f16 went from 7 node evaluations, all of them
cast outputs, to 66 of real arithmetic.

### The bug this uncovered

Acting on the finding turned up a defect the coverage recording had pointed at
without naming: **the comparison kernels never checked their input dtype.**
`compare_f32` asserted only that its *output* was Bool, then read both operands
as f32 whatever they stored. So `a > b` on f16 read 2-byte halves as 4-byte
floats, and on i64 read two elements as one. Both returned a plausible mask and
neither raised.

```
f16 [1,2,3,4] > [4,3,2,1]  ->  [F, T, T, F]      correct: [F, F, T, T]
i32 with negatives         ->  [F, F, F, F]      correct: [F, T, T, T]
```

The i32 case is the instructive one: it agrees for *positive* values, because
IEEE-754 positive floats order the same way their bit patterns do as integers.
A test using positive inputs would have passed.

This is why the audit's dtype table said "works: comparisons" for all three —
it recorded that nothing raised, not that anything was right. Checking for
exceptions is not checking for correctness, and the recording surfaced the
combination without being able to judge it. Fixed: f16 compares correctly,
integers raise.

---

## 3. Gaps found and closed

| Gap | Before | After |
|---|---|---|
| Operators never executed | 0 of 63 | 0 |
| Backward rules never fired | **5** of 47 | **0** (the last needed f16, §2) |
| Operators never given a strided input | **15** | **3** (2 impossible, 1 blocked) |
| Operators never run across workgroups | **9** | **0** |
| Operators on one backend only | 1 | 1 (`prod`, §4) |

111 tests added: 1,009 → 1,120.

**Backward rules.** `pow`, `clamp`, `scatter_add` and `col2im` had rules that had
never run. All four now have gradient tests against PyTorch. Writing them found
that `pow`'s gradient *with respect to the exponent* raises deliberately —
`ln(x)` is undefined for x ≤ 0, so the rule declines rather than returning a
number wrong on half its domain. That refusal is now pinned as a contract.

**Layouts and scale.** `tests/python/test_layout_and_scale.py` gives fifteen
operators transposed inputs and sizes above one workgroup, **on both backends**.

---

## 4. `prod` is CPU-only, deliberately

`prod` runs only the first half of the correctness chain, because it has no GPU
kernel. Filed as "implement it, or state it is CPU-only"; the answer is the
second, and for a reason that only appeared on inspection.

**A product's fold order is not a rounding detail — it decides when the fold
overflows.** `reduce.comp` gives each lane a strided slice and combines the
lanes in a tree. For a sum that reordering is a rounding difference inside the
1e-5 gate. For a product it is a different answer:

```
alternating 1e20, 1e-20 (512 values)
  index order, as the CPU folds it     ->  1.0
  grouped, as lane-striding groups it  ->  inf
```

Matching the CPU would mean multiplying in index order, i.e. one lane working
sequentially — a kernel with no parallelism, slower than the CPU at the only
thing it would be correct for. And §7 wants the oracle to share our exact
semantics, so a GPU `prod` that legitimately disagreed would break the chain
rather than extend it.

Nothing is blocked: `prod` has **no backward rule and no caller** inside `nn`,
the losses or the optimisers — it is a public API operator with no internal
consumer, which is also why building a kernel for it would have been speculative
work. Recorded in both backends where someone would look to change it, and
pinned by `test_prod_folds_in_index_order` so the ordering has to be confronted
first. Revisit if a caller appears, and then as a segmented sequential kernel
rather than a tree.

---

## 4b. Backend parity, and the invariant nobody was testing

`ARCHITECTURE.md` §7 makes the chain CPU-against-PyTorch, then Vulkan-against-CPU.
The second link has a precondition that had never been written down: **CPU
support must be a superset of Vulkan support.** A GPU capability with no CPU
counterpart has no oracle.

Measured rather than argued. Sweeping every operator across
{f32, f16} × {contiguous, strided} — 80 combinations — found **17 that Vulkan
declined**, and the shape of them was the finding: almost all were f16 in the
movement and indexing family, left behind when f16 landed for the arithmetic
ops. Not an architectural fallback problem at all.

Closing them took the same load/store swap as before, plus one that could not
be: `scatter_add` accumulated with `dst[...] +=` *into the output buffer*, which
for f16 rounds after every contribution — the 16-bit accumulator §7.3 forbids,
and not what the shader does. It now sums into a float scratch and narrows once,
which is what makes the two bit-comparable. The CPU backend's performance is
explicitly last, so buying exactness with a temporary is the right trade.

**17 → 6.** What remains is `prod` (no Vulkan kernel) and `max_pool2d` with a
strided input (its shader indexes planes directly). f16 now has exactly the same
GPU coverage as f32.

### The mistake this found, and the test that now prevents it

Widening the Vulkan gates first made the GPU accept `triu`, `tril`,
`scatter_add`, `im2col`, `col2im` and `max_pool2d` in f16 **while the CPU still
raised** — six operators whose GPU results nothing could verify. The full suite
stayed green throughout, because every test drives the two backends separately
and none compared what they will *accept*.

`tests/python/test_backend_parity.py` now sweeps the surface and asserts the
invariant directly, and separately pins the exact set Vulkan declines so that
changing it is deliberate. Verified non-vacuous by reintroducing the defect: it
fails naming the operators and both remedies.

The general lesson is the one §1 asks for — what let the bug exist, survive
review, and escape the tests. Here all three had the same answer: capability
parity was an unstated assumption, so nothing could check it.

---

## 4c. The Python suite now runs under AddressSanitizer

`ctest --preset asan` sanitises `tests/cpp`, and most of the operator surface is
not tested there. That gap has a name: the `cat` use-after-free that reached the
Python suite as "rank 43020 exceeds kMaxDims=4" and that ASan never saw, because
ASan was not in the build the Python suite loads.

`scripts/asan_python.py` builds an instrumented extension and runs the same
suite against it; CI runs it as a separate job.

**Verified against that exact bug.** Reintroducing the dangling-iterator
construction in `cat`'s shape inference produces
`ERROR: AddressSanitizer: heap-buffer-overflow`, with the trace landing in
`vkml::cat` → `vector::_M_range_initialize_n`, and the run aborts. Restoring the
fix returns the suite to green. So this gate detects the class of defect it was
created for, rather than merely running.

Three things the mechanics forced, each verified rather than assumed:

- **The runtime must be preloaded.** Python is not instrumented, so an ASan
  extension imports with `undefined symbol: __asan_option_detect_stack_use_after_return`
  until `LD_PRELOAD` supplies it.
- **pytest's capture swallows the report.** With capture on, the process aborts
  with exit 134 and *stderr arrives empty* — the least useful possible failure.
  The runner passes `-s` for that reason alone.
- **Leaks are off.** CPython, numpy and torch all hold allocations at exit by
  design; LeakSanitizer would bury a real finding under them. What this gate is
  for is use-after-free, overflow and invalid free — the class that produced
  wrong output rather than a crash.

**Scope, stated because "the suite passed under ASan" would otherwise sound
broader than it is.** This covers the CPU backend and every host-side layer
above it, which is where the motivating bug was. Vulkan is absent: the runner
has no GPU, and a memory bug inside a compute shader is not something ASan can
see in any configuration. The suite reports 678 passed and 440 skipped for that
reason.

Cost: about 4x the wall time (82 s against 21 s) and a ~20x larger extension, so
it is a separate CI job rather than a flag on the existing one.

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
| ~~26~~ | ~~Decide f16 and integer arithmetic~~ | **Done** — f16 computes on the CPU; integers are storage and indices (§2) |
| ~~30~~ | ~~Implement f16 on the Vulkan backend~~ | **Done** — elementwise, comparisons, reductions and softmax; bit-exact against the CPU oracle (§2) |
| ~~31~~ | ~~f16 in the GEMM family~~ | **Done** — bit-exact on every path including split-K, whose partials stay f32 (§2) |
| ~~27~~ | ~~Reconcile the backends' input requirements~~ | **Done** — the divergence was mostly missing f16, now closed 17 → 6, and the parity invariant is tested (§4b) |
| 32 | **An f16 vectorised tile load for the GEMM shaders** | §2. f16 matmul is 1.45× slower than f32 purely because `load4` is f32-only. Restores parity; measured not to be a speedup, so it waits on someone actually running f16 GEMMs |
| 27 | Reconcile the backends' input requirements, or document the divergence | §5. Sharpens item 16 |
| ~~28~~ | ~~Implement `prod` on Vulkan, or state it is CPU-only~~ | **Done** — stated CPU-only, with the ordering argument and a test (§4) |
| ~~15~~ | ~~Run the Python suite under a sanitizer in CI~~ | **Done** — `scripts/asan_python.py` plus a CI job; verified against the bug that motivated it (§4c) |
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

Both properties that were unreachable — mixed precision, and the backward half
of `Cast` — are now reachable and covered, on **both** backends, with the two
verified bit-exact against each other, matmul included. Every f16 operator now
runs the full correctness chain.

All gates green: layering (56 files) · clang-format · debug `-Werror` · release ·
ASan build and suite · ctest · **1,171 Python tests, 5 skipped** · **29 of 29
mutations killed** · validation layers clean on the f16 GPU path, split-K
included.

Five of those mutations exist for the f16 precision contract, which a tolerance
comparison cannot check: accumulating a reduction or a dot product in 16 bits,
reading every comparison operand as f32, and — on the shader side — ignoring the
DTYPE specialisation constant or narrowing twice on store. All five are killed,
by tests that assert exact values rather than approximate ones.
