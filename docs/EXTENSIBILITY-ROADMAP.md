# Extensibility Roadmap — from a tensor library to a model runtime

**Status:** proposal, nothing implemented
**Date:** 2026-08-01
**Covers:** the path from the current core (Stages 1–2 essentially complete) to model
interchange, LLM inference, and multi-device execution.

This roadmap answers one question: **what has to change so that adding the next sixty
operators costs a fraction of what adding the last one did.** Everything else follows from
that, because every remaining stage is, mechanically, a large number of new operators.

---

## 0. The measurement this roadmap is built on

`erfc` was added in `103e513`. It is the **cheapest operator this project can have** — a
unary elementwise function that slots into an existing shader's `switch`, needs no new
dispatch path, no new push-constant block, and no new memory pattern.

It touched **15 files and 145 lines.**

| Touchpoint | File |
|---|---|
| Op identity | `include/vkml/graph/op.h`, `src/graph/op.cpp` |
| Public C++ API | `include/vkml/api/ops.h`, `src/api/ops.cpp` |
| GLSL | `shaders/unary.comp` |
| Vulkan dispatch + `supports()` | `src/backend/vulkan/vulkan_backend.cpp` |
| CPU oracle | `src/backend/cpu/kernels_elementwise.cpp` |
| Gradient | `src/autograd/autograd.cpp` |
| Python surface | `bindings/module.cpp`, `python/vkml/__init__.py` |
| Numerics policy | `tests/python/tolerance.py` |
| Tests | `test_ops_vs_torch.py`, `test_vulkan_kernels.py`, `test_invariants.py` |
| Coverage gate | `docs/coverage-baseline.json` |

Now count what the remaining stages need. RoPE (4 variants), GQA, KV-cache reads and
writes, flash attention, causal masking, int8/int4 dequantising matmuls (one per format),
top-k, top-p, repetition penalty, plus the ONNX operator subset — that is **60 to 100 new
operators**, most of them harder than `erfc`.

At 15 files each, that is not primarily a *volume* problem. It is a **defect-rate** problem:
15 edit sites is 15 chances to make exactly the class of mistake this project keeps finding
— a rule applied in one place and forgotten in another. `load_state_dict` dropping `device`
while preserving `dtype` and `requires_grad` (issue #34) is that failure in miniature, and
it survived 1,410 tests.

> **The extensibility work is not a detour from the feature roadmap. It is the thing that
> decides whether the feature roadmap is affordable, and whether it arrives correct.**

---

## 1. Two strategies, and why the boring one wins

### Strategy A — narrow the primitive set (the tinygrad model)

`third_party/reference/tinygrad/tinygrad/uop/ops.py` defines a small closed set of UOps and
composes everything else from it. A new "operator" becomes a composition, not a kernel, so
the marginal cost approaches zero.

| | |
|---|---|
| **Benefit** | New ops cost almost nothing. One correct kernel serves many operators |
| **Cost** | A rewrite of the backend around a lowering pass. Fused kernels become the compiler's job, and vkML has none. Performance depends entirely on a fusion engine that does not exist yet |
| **Worthwhile when** | Starting fresh, or when the op count is about to explode past what hand-written kernels can serve |
| **Not worthwhile when** | 66 hand-tuned kernels already exist and a determinism contract pins their exact arithmetic — a lowering pass changes evaluation order by design |

### Strategy B — declarative op definitions with generated boilerplate (the PyTorch model)

PyTorch declares each operator once in `native_functions.yaml`; codegen emits the dispatcher
entries, the bindings, and the autograd nodes. The kernels stay hand-written.

| | |
|---|---|
| **Benefit** | The 15 touchpoints collapse to roughly 2: a declaration, and the kernels. The *rule* that an op needs a CPU oracle, a tolerance entry and a coverage row becomes mechanically enforced instead of remembered |
| **Cost** | A code generator to maintain, and generated code in the build. Debugging crosses a generation step |
| **Worthwhile when** | Many operators share a small number of shapes, which is exactly vkML's situation |
| **Not worthwhile when** | The op count is stable — generation is pure overhead if nothing new is coming |

### Recommendation

**Strategy B, applied incrementally, with A used opportunistically for the LLM operators.**

RoPE, GQA, causal masking and the samplers compose cleanly from operators that already
exist; they should be *composed* first and only lowered to fused kernels where a profile
demands it. That is A's benefit without A's rewrite.

The decisive argument against A is the determinism contract. It is a hard invariant, and a
lowering pass that re-associates arithmetic breaks it by construction. That trade needs its
own ADR, not a roadmap bullet.

---

## 2. The constraint nobody can design around: the CPU oracle

`ARCHITECTURE.md` §7 makes correctness a chain — CPU against PyTorch, then Vulkan against
the CPU — and `test_backend_parity.py` enforces that **CPU support is a superset of Vulkan
support**.

Every new Vulkan operator therefore needs a CPU implementation, and for the coming work that
is not free:

- **Flash attention** on the CPU is just attention, which is fine.
- **Quantised matmul** needs a CPU dequantiser per format — real work, times a dozen formats.
- **Paged KV cache** needs the CPU to model the same layout.

Meanwhile the CPU backend is deliberately naive and roughly 116× slower than torch (#31), so
these oracles are slow as well as numerous.

**This must be decided before Stage 6, not during it.** Three coherent options:

1. **Keep the superset rule.** Every quantised kernel gets a CPU dequantiser. Honest, slow,
   and expensive in author time.
2. **Narrow the rule to "verifiable" rather than "executable"** — a quantised op is checked
   against a *dequantise-then-f32-matmul* reference built from ops that already exist, rather
   than against a bit-identical CPU twin.
3. **Declare quantised inference a separate contract** with its own tolerance policy, since
   quantisation is a deliberate accuracy trade and bit-exactness against f32 is meaningless.

Option 2 preserves the chain's purpose (an oracle that shares vkML's semantics) at a fraction
of the cost, and is my recommendation — but it changes a stated invariant and needs an ADR.

---

## 3. Hardware constraints that shape every kernel choice

Measured on the development machine, not assumed:

| Capability | Navi10 (RX 5600M) | Consequence |
|---|---|---|
| `cooperative_matrix` | **no** | No tensor-core path. Flash attention and GEMM must use the scalar path |
| `global_float_atomic_add` | **no** | Cross-workgroup float reduction needs a two-pass or split-K structure, never an atomic |
| `shader_float16` | yes | f16 storage and arithmetic available |
| `subgroup_size_control` | yes, 32–64 | Subgroup-width kernels are viable but must not assume 64 |

`M3_ROADMAP.md` already commits to no cooperative-matrix work and no fp16 accumulation. Both
hold here.

---

## 4. Phases

Ordered by *what unblocks what*, not by the stage numbering. Each phase names the reference
already cloned under `third_party/reference/`, so the work starts from someone else's
solved problem.

---

## 4a. Performance — measured, and reordered because of it

An earlier draft of this document treated performance as one phase and deferred the rest to
`M3_ROADMAP.md`. That was wrong twice over: `M3_ROADMAP.md` is **sixteen items, all GEMM** —
it says nothing about convolution, elementwise bandwidth, the optimiser, or the dispatch
layer — and the end-to-end gap turns out not to be a kernel problem at all.

### The number this section exists for

Same architecture, same batch, same optimiser, measured 2026-08-01 on the development
machine (`build_cnn` against `build_torch_cnn` from `examples/cifar100/train.py`):

| | ms/step |
|---|---|
| PyTorch, **CPU**, 8 threads | **34.26** |
| vkML, **discrete GPU**, 36 CU | **35.77** |

**vkML on a 36-compute-unit discrete GPU is 4% slower than PyTorch on the CPU.** Since torch
on a GPU would be many times faster than torch on a CPU, the real distance to parity is not
the ~1× this table shows — it is that whole multiple, and it is the reason this section comes
before the feature phases.

### Where it goes: starvation, not slow kernels

Batch scaling separates fixed cost from arithmetic. If time per sample falls as the batch
grows, the device was idle waiting for work:

| batch | ms/step | ms/sample | vs batch 64 |
|---|---|---|---|
| 64 | 34.45 | 0.538 | 1.00× |
| 128 | **26.23** | 0.205 | 0.38× |
| 256 | 39.39 | 0.154 | 0.29× |
| 512 | 70.29 | 0.137 | **0.26×** |

Per-sample cost falls **3.9×**. The strongest single line is batch 128: it does twice the work
of batch 64 in *less wall time*, which only happens when fixed per-step cost dominates
arithmetic.

**At the batch size the examples use, roughly three quarters of a training step is overhead.**

> **Superseded by direct measurement, 2026-08-02.** This figure is an *inference* from batch
> scaling and covers the whole step, data loading and transfer included. P0 now measures the
> split directly: **44.8% host and driver, 54.8% GPU busy** for the compute region — see
> "P0 is met" below. The conclusion that overhead dominates arithmetic survives; the size
> does not. Quote the measured number.

Three independent observations agree:

- MNIST trains at 4.41 s/epoch on a 36-CU discrete card and 4.35 s/epoch on a 6-CU integrated
  one. Six times the compute, no difference — the workload never reaches the arithmetic.
- Issue #33 measures a submission at ~105 µs against ~9 µs for a dispatch.
- Issue #32 finds the optimiser is 62.7% of an MLP step across 12 submissions.

`M3_ROADMAP`'s sixteen GEMM items are real work aimed at the ~26% that is arithmetic. They
were correctly researched and are wrongly sequenced: tuning kernels that run for a quarter of
the step cannot fix a step that is three-quarters overhead.

### P0 — Attribution *(before any optimisation)*

**Nothing here can be prioritised properly, because vkML cannot currently say which KERNEL
costs what.** The evidence above is all indirect — batch scaling, device substitution,
submission counting — which is enough to locate the problem and not enough to close it.

**Corrected 2026-08-02, after reading the code.** An earlier draft of this section claimed
`vulkan_last_profile` returns submission-level tuples, that there is no per-dispatch breakdown,
and that timestamps must be written around each dispatch. All three are wrong, and the roadmap
should describe reality:

- `vk_command.h` defines `ProfileEntry{label, gpu_ms}` and `begin_timestamp(label)`; timestamps
  are already written and resolved **per node**.
- `vulkan_last_profile()` returns real per-operation GPU time — measured,
  `[('submit', 0.32288), ('matmul', 0.31884)]`.

What is actually missing is narrower, and none of it is timestamp plumbing:

- **The label is the operator, not the kernel.** `vulkan_backend.cpp:1099` is the only
  `set_label` call site and passes `op_name(node->op)`, so a `matmul` entry cannot distinguish
  `gemm_naive` from `gemm_reg`, nor show split-K's partitions.
- **No correlation.** Cost cannot be joined to the choice that produced it.
- **No aggregation** by kernel across a step.
- **No unaccounted remainder** — which is this section's own exit criterion, and *is* the
  overhead being hunted.

The fix is deliberately **not** to write the kernel name into the profiler's label. That would
give kernel selection two owners: the backend already publishes it as a Decision, and the
profiler would derive it again. Identity becomes a third fact instead — `DispatchId`, owned by
`CommandRecorder`, describing nothing, carried by both Decision and Measurement so that a
consumer joins them. See `docs/OBSERVABILITY-ARCHITECTURE.md` §4b.

**Reference.** llama.cpp's Vulkan backend keeps per-op timing behind a build flag; the same
query-pool-per-dispatch structure applies here. `VK_EXT_debug_utils` labels make the result
readable in RenderDoc and RGP as well.

**Exit criterion.** A CIFAR step prints a per-kernel table summing to the measured wall time,
with the unaccounted remainder shown explicitly rather than hidden — the remainder *is* the
overhead being hunted.

### P0 is met — and it corrects the number this section was built on

```
python examples/cifar100/train.py --attribute 20
```

20 steps, batch 64, after 20 warm-up steps, RX 5600M / RADV. Reproduced three times;
the percentages move by less than a point between runs while the wall time moves by 20%,
which is the clock behaviour §"Warm-up" in `MEASUREMENT-AUDIT.md` describes.

```
  kernel                count     gpu ms   % step
  matmul                  540     64.321    17.1%
  sum                     180     37.254     9.9%
  im2col                   60     32.034     8.5%
  max_pool2d_backward      60     21.222     5.6%
  add                     240     14.503     3.9%
  col2im                   40     13.603     3.6%
  (16 more)              2020     23.980     6.6%
  ------------------------------------------------
  GPU busy                       205.917    54.8%
  GPU idle in submits              1.334     0.4%
  host and driver *              168.419    44.8%
  ================================================
  step wall                      375.670   100.0%

  780 submissions, 560 of them with work to time
  * upper bound: a profiled wall clock includes the profiler's own readback
  GPU / wall = 0.55
```

**The overhead is real and it is not three quarters.** This section's headline —
*"roughly three quarters of a training step is overhead"* — came from batch scaling,
which measures how per-sample cost falls and infers a fixed component. Direct
attribution puts host and driver at **44.8%, and that is an upper bound** (rule 4:
the wall clock here is profiled, and the readback cannot be subtracted out by
comparing against an unprofiled run).

The two are not in contradiction so much as measuring different things. The batch-scaling
figure covers a whole training step including data loading and host-to-device transfer,
which `train()` reports separately; this covers forward, backward, optimiser and the
realisation that waits for them. Anything quoting 74% should now quote this instead,
because it is measured directly rather than inferred, and it names *which* bucket.

**P1's premise survives; its size changes.** Nearly half a step outside every submission
window is still the largest single item available, and larger than any kernel — `matmul`,
the biggest, is 17.1%. The reordering argument that put P1 before `M3_ROADMAP`'s sixteen
GEMM items holds. What changes is the expected ceiling: eliminating submission overhead
entirely buys up to ~45%, not ~75%.

**Two things the table says that batch scaling could not.**

- **780 submissions for 20 steps — 39 per step — and only 28 of them have anything to
  time.** The other 11 are copies: uploads and the download behind `.item()`. P1's first
  candidate is "one submission per step, not twelve"; the real count is 39, and a third of
  them carry no compute at all.
- **GPU idle inside submissions is 0.4%.** The barriers between dispatches are not the
  cost, so P1's third candidate — *fewer dispatches per operation* — is aimed at
  dispatch-side host cost, not at gaps on the device. It cannot be justified by GPU idle
  time, because there is almost none.

### P1 — Dispatch and submission overhead *(measured at 44.8% of a CIFAR step)*

The single largest win available, already partly diagnosed in #32 and #33, and larger than
any individual kernel — `matmul`, the biggest, is 17.1%.

Candidates, reordered by what P0 measured rather than by what was expected:

1. **One submission per step, not thirty-nine.** Measured: 780 submissions for 20 CIFAR
   steps. The earlier figure of twelve came from an MLP; a CNN is three times that. The
   optimiser is still the worst offender and is embarrassingly batchable — every
   parameter's update is independent.
2. **Eleven of those 39 carry no compute at all** — uploads, and the download behind
   `.item()`. They pay submission cost for a copy. This is the cheapest item on the list
   and it was invisible before P0, because a submission with no dispatch produces no
   profile.
3. **Command buffer reuse.** A training step re-records an identical sequence every iteration;
   the shapes do not change between steps.
4. **Fewer dispatches per operation** — the elementwise chain in an optimiser step is a
   sequence of tiny kernels, each paying full dispatch cost. **Note what this is not
   justified by:** GPU idle time inside submissions is 0.4%, so the barriers between
   dispatches are not the cost. The case for this is host-side dispatch overhead, and it
   has to be made on that basis.

**Reference.** vkML's own `BACKWARD-PERF-INVESTIGATION.md` and `PERFORMANCE-MODEL.md`; and for
the pattern, llama.cpp's `ggml-vulkan` builds one command buffer per graph rather than per
node, which is precisely the structural difference.

**Exit criterion.** The discrete and integrated GPUs stop tying on MNIST. That cannot be
produced by measurement noise, which a percentage improvement can.

### P2 — Convolution

CIFAR is a CNN and `M3_ROADMAP` does not mention convolution once. vkML exposes `im2col` and
`col2im` as operators, which implies convolution is lowered to explicit im2col followed by
GEMM — materialising a K×N expansion of the input in memory before any arithmetic happens.
**Verify this before designing around it.**

If confirmed, the options in ascending order of effort are implicit GEMM (fuse the im2col
addressing into the GEMM's operand load, so the expansion is never written), direct
convolution for small channel counts, and Winograd for 3×3.

**Reference.** CUTLASS — already cloned — is the reference architecture for implicit GEMM;
ncnn implements direct and Winograd paths for exactly this GPU class.

### P3 — Kernel tuning (`M3_ROADMAP`, unchanged)

The existing sixteen items, now correctly placed: they address the arithmetic that remains
after P1 removes the overhead around it. Items 1 (GEMV) and 3 (autotuning) stay the
highest-value entries.

### A free result worth taking first

At batch 512 the per-sample cost is 0.137 ms against 0.538 ms at batch 64. Nothing in the
library changes — the examples simply run at a batch size that starves the device less. This
is not an optimisation and should not be reported as one, but it does mean the current
example timings understate the hardware by roughly 4×, and any future comparison should state
its batch size.

---

## 4b. Feature phases

### Phase 0 — Operator declaration and generation *(the foundation)*

**Goal.** One declaration per operator; generate the op enum, the public API, the binding,
the re-export, and the test/tolerance/coverage scaffolding. Kernels stay hand-written.

**Approach.** A single `ops/*.yaml` (or TOML) declaring name, arity, dtype rules, shape rule,
gradient formula or `none`, and tolerance class. A generator emits the mechanical files;
`shaders/` and `kernels_*.cpp` remain authored. Start by generating the three *least* risky
touchpoints (binding, re-export, tolerance entry) and widen once the generator is trusted.

**References.** PyTorch's `native_functions.yaml` for the schema vocabulary; ONNX's operator
schema registry for how shape/type inference is expressed declaratively.

**Exit criteria.** Adding an `erfc`-class operator touches ≤ 4 files. The generator is
verified by regenerating the *existing* 66 operators and getting a byte-identical tree —
which also proves it models the current surface rather than a simplified one.

**Risk.** A generator that only handles elementwise ops is a trap: it makes the easy case
easier and leaves the hard cases outside the system. Gate acceptance on it expressing
`cat`, `scatter_add` and `max_pool2d`, not just `erfc`.

---

### Phase 1 — safetensors *(cheapest ecosystem win)*

**Goal.** `vkml.load_safetensors(path)` → a state dict loadable into a module.

**Approach.** The format is deliberately trivial: an 8-byte little-endian header length, a
JSON header mapping tensor names to `{dtype, shape, data_offsets}`, then a contiguous data
blob. No dependency, no protobuf, ~200 lines. vkML already has the checkpoint machinery
(`serialize.py`, `Checkpoint`, the decompression cap from #24) to hang it on.

**References.** `huggingface/safetensors` — the format specification in that repository.
vkML's own `serialize.py` for the surrounding conventions.

**Exit criteria.** A real HuggingFace checkpoint loads and its tensors match what the
`safetensors` Python package reports, byte for byte. Hostile inputs — truncated header,
offsets outside the blob, overlapping ranges — are rejected explicitly, reusing the posture
already established for checkpoint loading.

**Why now.** It is the smallest change that makes vkML able to consume other people's
weights, and it is a prerequisite for every LLM phase.

---

### Phase 2 — Submission overhead *(the largest measured performance win)*

**Goal.** Close #32/#33.

**Evidence.** Measured on 2026-08-01: MNIST MLP trains at **4.41 s/epoch on a 36-CU discrete
GPU and 4.35 s/epoch on a 6-CU integrated one**. A six-fold difference in compute capacity
produced no difference in wall time — the workload is bound by submissions, not arithmetic.

**Correction, 2026-08-02.** An earlier draft added *"CIFAR-100's CNN, by contrast, reports
96.3% compute, so the ceiling is specific to small models"*. That reads `train()`'s
`compute` bucket as GPU time, and it is not: the bucket is the host wall time of the
forward/backward/optimiser region, so 96.3% says only that batch loading and transfer are
3.7%. Attributing *inside* that region (§4a, "P0 is met") puts **44.8% of the CNN's step
outside every submission window**. The CNN does not behave differently in the way that
sentence claimed, and the ceiling is not specific to small models.

**Approach.** Already diagnosed in the issues: the optimiser is 62.7% of an MLP step across
12 submissions, and a submission costs ~105 µs against ~9 µs for a dispatch. Batch the
optimiser into one submission; more generally, make the plan layer submit once per step.

**References.** vkML's own `docs/BACKWARD-PERF-INVESTIGATION.md` and `PERFORMANCE-MODEL.md`.
No external reference needed — this is a vkML-specific structural issue and it is already
measured.

**Exit criteria.** Discrete and integrated GPUs stop tying on MNIST. That is a stronger and
cheaper signal than a percentage improvement, because it cannot be produced by noise.

---

### Phase 3 — LLM runtime primitives *(composed, not fused)*

**Goal.** RoPE, KV cache, GQA, causal masking — enough to run a decoder-only forward pass.

**Approach.** Compose from existing operators first. RoPE is a reshape, two element-wise
multiplies and a rotate; GQA is an index/repeat over the head axis; causal masking already
exists via `triu` + `masked_fill`. Only fuse where a profile demands it. This is where
Strategy A pays without a rewrite.

**References (verified present locally).**

| Need | Reference |
|---|---|
| RoPE variants | `llama.cpp/.../vulkan-shaders/rope_norm.comp`, `rope_neox.comp`, `rope_multi.comp`, `rope_vision.comp` |
| KV cache structure | `llama.cpp/src/llama-kv-cache*.cpp` |
| RoPE theory | RoFormer (Su et al., 2021) |
| GQA | Ainslie et al., 2023 |

**Exit criteria.** A small open-weights decoder produces logits matching a PyTorch reference
within the documented transcendental tolerance, greedily decoding the same token sequence.

---

### Phase 4 — Flash attention

**Goal.** Attention that does not materialise the N×N score matrix.

**Approach.** The scalar path, not cooperative matrix — Navi10 has none.
`llama.cpp` ships exactly this split: `flash_attn.comp` is the scalar implementation and
`flash_attn_cm1.comp` / `flash_attn_cm2.comp` are the coopmat variants, with
`flash_attn_split_k_reduce.comp` handling the cross-workgroup combine. That last file matters
specifically because vkML's GPU has **no global float atomicAdd**, so the reduction must be
structural.

**References.** FlashAttention (Dao et al., 2022) and FlashAttention-2 (2023) for the
algorithm; the llama.cpp shaders above for a Vulkan realisation on the same GPU class.

**Determinism risk — decide before implementing.** Online softmax accumulates in a different
order than the naive form. Under vkML's contract that is a numerical change requiring a
re-derived error bound, not a free optimisation. Treat it as an ADR, not a patch.

---

### Phase 5 — Quantisation

**Goal.** int8 and int4 weight-only inference.

**Approach.** Start with a **single** format end to end rather than a family. Q4_0 or Q8_0 is
the right first target: block-scaled, simple, and llama.cpp has both a dequantiser and a
dequantise-on-the-fly matmul for it. Add formats only once one is proven.

**References (verified present locally).** `llama.cpp/.../vulkan-shaders/dequant_q4_0.comp`,
`dequant_q8_0.comp` and the `dequant_q*_k.comp` family for k-quants;
`llama.cpp/gguf-py/gguf/gguf_reader.py` as a readable GGUF parser. GPTQ (Frantar et al.) and
AWQ (Lin et al.) for weight-only quantisation theory if calibration is wanted later.

**Blocking decision.** The CPU-oracle question from §2 must be answered first. This phase is
where the superset rule becomes genuinely expensive.

**Hardware note.** Check `VK_KHR_shader_integer_dot_product` before designing the int8 inner
loop; without it the dot product is manual and the arithmetic choice changes.

---

### Phase 6 — ONNX import

**Goal.** Load a model graph, not just weights.

**Approach, and a correction to the framing.** ONNX will not "instantly unlock thousands of
models". The default opset is ~190 operators, real exports use dynamic shapes and control
flow, and coverage is where every ONNX importer spends its time. The honest scope is: **declare
a supported operator subset, import within it, and fail loudly outside it** — which is exactly
the all-or-nothing posture vkML already takes for the Vulkan backend, and the same reason
`prod` raises `NotImplementedError` rather than silently falling back.

Sequence it *after* Phase 0 and Phase 3, because the subset is far cheaper to cover once the
operator set is broad and adding one is cheap.

**References.** `onnx/onnx` for the protobuf schema and the operator specifications;
**wonnx** (ONNX on WebGPU) as the closest architectural analogue — a GPU-compute ONNX runtime
without a vendor library underneath; **ncnn**'s ONNX converter for a mature, readable mapping
of ONNX ops onto hand-written GPU kernels. *These three are not currently cloned under
`third_party/reference/`; adding them is the first task of this phase.*

**Exit criteria.** A published ResNet-50 and a published BERT-base ONNX file both run and
match onnxruntime's CPU output within tolerance.

---

### Phase 7 — Multi-GPU

**Goal.** Data-parallel training across the two devices already in the machine.

**Approach.** Single-process, two devices — *not* multi-node. There is no NCCL for Vulkan, so
the gradient exchange is either host-staged or `VK_KHR_external_memory`. Host staging is the
correct first implementation: simple, portable, and measurable. Reference the PyTorch DDP
design for gradient bucketing and overlapping communication with the backward pass, and ring
allreduce for the exchange pattern.

**Determinism risk.** Summing gradients across devices changes reduction order. Same class of
decision as Phase 4.

**Sequencing.** After Phase 2. Distributing a workload that is submission-bound multiplies the
overhead rather than the throughput.

---

### Continuous — developer experience and release

Not a phase; these accumulate.

- **API reference documentation.** There is none today (`docs/` has no `api*`). Generated from
  the docstrings that already exist would be most of the way there.
- **Release mechanics.** No `CHANGELOG`, no git tags, no publish step. Needed before any
  version claim.
- **CI.** Deferred by decision until billing is restored. `Dockerfile.ci`, the local registry
  and the self-hosted runner already work and are committed; the GPU job is written but not
  landed.

---

## 5. What this roadmap deliberately does not do

- **No lowering pass or fusion compiler.** That is Strategy A, and it trades the determinism
  contract for extensibility. It may become right later; it is not right as a side effect.
- **No cooperative-matrix work.** Consistent with `M3_ROADMAP.md`, and the hardware has none.
- **No fp16 accumulation**, including inside flash attention, without a re-derived bound.
- **No new backend.** CUDA, Metal and WebGPU are all plausible and all dilute the one that
  works.
- **No training-time quantisation.** Weight-only inference first; QAT is a different project.

---

## 6. Suggested order, with rough shape

| Phase | Unblocks | Effort | Risk |
|---|---|---|---|
| 0 · Op declaration | everything after it | large | medium — generator scope creep |
| 1 · safetensors | all LLM work | small | low |
| 2 · Submission overhead | all small-model perf | medium | low — already diagnosed |
| 3 · LLM primitives | Phase 4, 5 | medium | low — composed from existing ops |
| 4 · Flash attention | long contexts | large | **high — determinism** |
| 5 · Quantisation | large models | large | **high — CPU oracle policy** |
| 6 · ONNX import | model interchange | large | medium — scope discipline |
| 7 · Multi-GPU | throughput | medium | **high — determinism** |

Phases 1 and 2 are independent of Phase 0 and can start immediately. Everything from Phase 3
onward should wait for it, because that is where the per-operator cost starts to compound.

---

## 7. Open questions this proposal does not settle

1. **Does the determinism contract survive flash attention and multi-GPU**, or does it become
   "deterministic within a configuration"? This is the single largest architectural question
   in the roadmap and it deserves an ADR before Phase 4.
2. **What replaces the CPU-superset rule for quantised kernels** (§2)?
3. **Is a generator worth it at 66 operators**, or only at 150? The Phase 0 exit criterion —
   regenerating the existing surface byte-identically — is designed to answer this cheaply,
   before committing to it.
4. **Which model is the target?** "LLM support" is unbounded; "Llama-3-8B in Q4_0 at
   interactive speed" is a specification. The second one can be finished.
