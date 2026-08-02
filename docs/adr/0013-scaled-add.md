# ADR 0013 — `scaled_add`, and the fused optimiser kernel that lost to it

**Status:** accepted, implemented and measured.
**Date:** 2026-08-02
**Covers:** `OpKind::ScaledAdd`, `shaders/scaled_add.comp`, `python/vkml/optim.py`'s SGD.
**Hardware:** AMD RX 5600M (RDNA1), RADV. Validation layers off throughout
(`MEASUREMENT-AUDIT.md` §6c).

---

## 1. The measurement this exists for

An MNIST MLP step spends **427 µs in the optimiser** updating 101,770 parameters. The traffic
is ~2 MB — read gradient, velocity and parameter, write velocity and parameter — so the
bandwidth floor at 288 GB/s is **7.1 µs**. It runs **60× off** that.

Split by whether the host or the device is doing it:

```
  optimiser, submit only    202 us   (host: building the graph and submitting)
  optimiser, drained        458 us
  => GPU                    256 us
  bandwidth floor             7.1 us
```

**Both halves are wrong, and for one reason**: an SGD-with-momentum step over four parameters
issued **24 dispatches**. Counted by op:

```
  full x8   mul x8   add x4   sub x4
```

Six per parameter, for arithmetic that is two lines:

```
  velocity = velocity*momentum + grad
  param    = param - velocity*lr
```

**A third of them exist only to materialise a number.** `scalar_like` wraps a Python float as a
rank-0 tensor (`src/api/ops.cpp`), so `velocity * 0.9` is a `full` node *and* a `mul` node.
Every scalar operation in vkML costs two dispatches and two allocations rather than one —
measured directly: `a * b` is 1 dispatch, `a * 2.0` is 2.

## 2. What the references do, and why

**ggml/llama.cpp fuses by hand-written pattern.** `ggml_can_fuse_ext` checks that a run of nodes
matches a named op sequence, that every intermediate has exactly one use, and that the shapes
agree; the Vulkan backend then dispatches one bespoke kernel. There are perhaps a dozen such
patterns — `{RMS_NORM, MUL}`, `{ROPE, VIEW, SET_ROWS}`, several MoE shapes. It suits LLM
inference, where a small fixed set of patterns covers nearly all the value, and it needs no
runtime compiler.

**tinygrad fuses generally, by compiling.** Its scheduler rewrites a UOp graph, decides where a
buffer has to be materialised (`bufferize`, `limit_bufs`) and generates a kernel for each
region. Fully general — and it works because tinygrad compiles kernels at runtime, which is its
whole thesis.

**vkML cannot take tinygrad's route.** Shaders are compiled to SPIR-V at build time and embedded
as headers; there is no runtime shader compiler, and adding one would bring a `glslc` runtime
dependency and put generated code in the path of a bit-reproducibility guarantee. So the
realistic options are ggml-shaped: name a pattern, write a kernel.

## 3. The alternatives, prototyped and benchmarked

All four arms drive the Recorder directly on the MNIST MLP's real parameter shapes. GPU time is
the `submit` window, not how long the host sat blocked — a blocking wait carries the ~35 µs
wake-up with it, and measured that way the difference between four dispatches and eight
vanished into the constant. Minimum of 200, two independent runs.

| | dispatches | GPU | |
|---|---|---|---|
| **A** composed, as vkML built it | 24 | 126.0 µs | — |
| **B** `scaled_add`: `a*alpha + b*beta` | 8 | 47.2 µs | **2.7×** |
| **C** `sgd_step`: the whole update, one kernel | 4 | 31.7 µs | 4.0× |
| **C2** the same, no barriers between parameters | 4 | 24.2 µs | 5.2× |

**C is faster than B and is rejected anyway.** Naming the cost, as §3 of the constitution
requires:

| | |
|---|---|
| **Benefit** | 4.0× against B's 2.7× |
| **Cost** | Two outputs, so it cannot be a graph node at all — it needs a new `Backend` entry point beside `compute`. The update's arithmetic then exists in three places (Python reference, GLSL, CPU oracle) instead of one. One kernel **per optimiser**: SGD, RMSProp, Adam and Adagrad each need their own. And the Python optimiser stops being expressed as tensor ops, which is how it is validated against torch step for step |
| **Worthwhile when** | One optimiser dominates and its maths has stopped changing |
| **Not worthwhile when** | There are four optimisers in the tree and the framework is still adding them |

`scaled_add` captures over half the win, is one output, is differentiable, needs no new backend
concept, and is not specific to optimisers at all: **every momentum optimiser is built from this
shape** — SGD's velocity and parameter updates, RMSProp's and Adam's moving averages. That is
P6, generic before specialised, and §3, which ranks maintainability above speed.

The `sgd_step` prototype was deleted. Its number is recorded here so the option stays open on
evidence rather than being rediscovered.

## 4. The design

`out = a*alpha + b*beta`, with both coefficients in `ScaledAddParams` and reaching the shader as
push constants.

**It is a fully general elementwise op**, not a fast path with holes. The first version took the
contiguous same-shape case only — dropping the per-operand metadata is what makes room for the
coefficients — and it threw on a bias gradient of shape `(1, n)` against a parameter's `(n,)`,
which is an ordinary broadcast and the first thing a real optimiser hits. **An operator that
fails on shapes its own API accepts is a defect, not an optimisation.**

The budget is met by storing **extents once and strides per operand**, and none for the output:
both inputs arrive broadcast to the output's shape so they provably share extents, and a
computed node's storage is always freshly allocated and contiguous. That is 84 bytes against the
128 Vulkan guarantees — the packing `docs/adr/0009` §2 describes for `where` and `softmax`, and
the reason `offset_from` is split out of `operand_offset` in `common.glsl`.

### Bit-identity

Verified by comparing bytes, on both backends, against both the composed form and an independent
f32 reference: `tests/python/test_invariants.py::test_scaled_add_is_bit_identical_to_the_composed_form`.
That is the acceptance criterion rule 8 prefers, because no instrument can distort it.

The kernel is two multiplies and an add, matching where `mul` would have rounded. **Whether
contraction would actually change the answer here is unverified**: an `fma()` variant was built
— the SPIR-V differed — and compared over 100,006 elements chosen to expose the intermediate
rounding, including `alpha = 1+2⁻²³`. Not one element differed. That is consistent with what
`check_precise_gemm.py` already records: on RADV the `NoContraction` decoration is invisible in
behaviour. The shape is kept because it is the one whose equivalence can be argued on paper.

## 5. What it cost, measured

```
                              before     after
  optimiser dispatches            24         8
  optimiser, host             202 us    120 us
  optimiser, drained          458 us    309 us     1.48x
  MNIST step (no upload)      873 us    756 us
  realistic step             1037 us    971 us     1.07x
```

The row that used to sit here as "optimiser, GPU 256 → 189 us" was `drained − host`. That is a
BLOCKING WAIT, not GPU time: it carries the host wake-up, and it is not comparable with a
timestamp. Measured by timestamps instead, the GPU is **149 µs across the two compute
submissions plus 37 µs of `assign` copies**.

The end-to-end figure is much smaller than the phase figure because the optimiser is only part
of a step.

> **Correction (same day).** This section originally continued: *"8 dispatches cost 189 µs here
> against 47 µs there"*, and offered that difference as a gap to be explained. **The two numbers
> are not the same quantity.** 189 µs was `drained − submit_only` — a blocking wall-clock wait
> over three submissions, unprofiled — while 47 µs was a GPU timestamp window over one
> submission, profiled. Rule 4 forbids comparing them, and the 189 additionally carries a ~43 µs
> host wake-up that is not GPU work at all.
>
> Measured properly, the optimiser step is **120 µs of host, 149 µs of GPU across the two compute
> submissions, and 295 µs drained** — plus a further 37 µs of GPU for the `assign` copies, which
> emit no `submit` window and were therefore missing from every total.
>
> `docs/OPTIMISER-COST-ATTRIBUTION.md` is the full item-by-item attribution, and it found the
> real cost: **two of the eight dispatches take a strided path because every weight gradient is a
> transposed view**, and one of them costs 78.8 µs where its contiguous twin costs 42.3.

## 6. Two defects this work exposed

Both are worth more than the speedup.

**Coefficients were applied after the node was realised.** `scaled_add` first called `binary()`
and then set the params — and `binary()` ends in `finish()`, which **realises the node in eager
mode**. Every eager result was computed with the defaults of 1 and 1. The lazy path hid it
completely, because there the params are set long before anything runs, and the GPU bit-identity
check passed for the same reason. The validation suite runs eager and caught it immediately.
The params are now set before `finish()`, with a comment saying why.

**The backward rule was dead code.** `coverage_matrix.py` reported `scaled_add` as a rule that
never fired: the optimiser calls it on detached tensors, so no gradient ever flows through it,
and the whole suite passed without ever running it. That is precisely why coverage tracks
"backward rule fired" separately from "tests green".
