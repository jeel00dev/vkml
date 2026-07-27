# vkML Phase 2 — From Research to a Complete Machine Learning Library

**Status:** active. Supersedes the M-series research objective as the project's primary goal.
**Date:** 2026-07-27

The M-series established an engineering foundation by investigating GPU execution behaviour,
numerical correctness, resource utilisation and performance characteristics. Those results are
the project's knowledge base. They are **not discarded** — `THEORY.md`, `MEASUREMENT-AUDIT.md`,
`PERFORMANCE-MODEL.md`, `GAP_ANALYSIS.md` and the ADRs remain authoritative and continue to
guide implementation, prevent known mistakes, and validate future optimisation.

This is not a restart. Everything implemented — the Vulkan runtime, shader infrastructure,
kernels, tests, benchmarks and research — is the foundation the complete library is built on.

---

## Mission

> Build a complete, correct, usable, maintainable, production-quality machine learning library
> on top of the existing vkML foundation.

Performance and optimisation remain important. Correctness, completeness, maintainability,
architecture and usability come **first**. A fast but incomplete library has little value; a
complete and correct one can always be optimised later.

```
Existing foundation → Complete functional library → Verification → Real ML applications
   → Profiling → Optimisation → Refactoring → Production quality → Community ready
```

---

## Core principles

**1 — Never waste existing work.** Nothing is rewritten merely because a new phase started.
Components are reused, extended, improved, and refactored when necessary. The current
implementation is the starting point, not a prototype to discard.

**2 — Correctness before performance.** Every component must be correct, deterministic,
numerically validated and fully tested before significant optimisation effort.

**3 — Complete the library first.** Do not optimise isolated kernels while major functionality
is missing. Library completeness takes priority over benchmark numbers.

---

## Starting position, measured

Principle 1 requires knowing exactly what exists. Counted from the tree at `a2d6a25`
(2026-07-27), not estimated:

| Layer | State |
|---|---|
| `OpKind` declared | **74** (≈65 compute ops after view ops and leaves) |
| CPU implementations | **47** |
| **Vulkan implementations** | **16** — `VulkanBackend::supports` is documented in-source as "only the M1 kernel set" |
| Declared with no implementation anywhere | **20** |
| `nn` modules | 6 — `Module`, `Linear`, `ReLU`, `Tanh`, `Sigmoid`, `GELU`, `Sequential` |
| Optimisers | 2 — `SGD`, `Adam` |
| Losses | 0 implemented (`MseLoss`, `CrossEntropy` declared only) |
| Data pipeline | none |
| Serialisation | none |
| Tests | 560 Python + 1 C++ suite, all green |

### The finding that motivates this phase

**The Vulkan backend — the point of the project — covers 22 % of declared operators.** There is
no GPU kernel for `Add`, `Sub`, `Mul`, `Div`, `Sigmoid`, `Tanh`, `Gelu`, `Silu`, `Sqrt`, `Log`,
`Pow`, `Where`, `Clamp`, or any comparison.

> **Correction, 2026-07-27.** This section first stated that such a model "runs most of its
> arithmetic on the CPU and is silently correct-but-slow, because `supports()` falls back rather
> than failing." **That is wrong**, and it was taken from a comment in `supports()` rather than
> from the code. There is no fallback. `Executor::realize` throws `NotImplementedError` when the
> backend does not support a node, verified directly:
> `V.prod(t)` on `vulkan:0` raises *"backend 'vulkan:0' cannot evaluate op 'prod'"*.
>
> `ARCHITECTURE.md` §3 Fork 3 does specify CPU fallback via graph splitting, and `supports()` is
> the predicate it would need — but the splitting was never built, and the executor says so:
> *"multi-device splitting is the scheduler's job and arrives with the Vulkan backend."*
>
> The finding is unchanged and the consequence is worse than described: a model touching a
> single unported operator **could not run on Vulkan at all.**

Meanwhile M3 and ten research stages went into tuning one operator, GEMM, to 1,270 GFLOP/s.

That is precisely the pattern Principle 3 exists to stop, and it is the strongest available
argument for this phase. The coverage gap persisted behind a green test suite because nothing
exercised an unported operator *on the Vulkan device*: the parity tests run on CPU, and the
Vulkan tests only cover operators that already had kernels.

**Immediate consequence:** binary and unary elementwise Vulkan kernels are the highest-value
work in the project. They unblock every model, they are numerically free (no fold order changes,
so goldens stay pinned), and they are cheap relative to anything in `M3_ROADMAP.md`.

### The 20 operators declared with no implementation

`Cat` · `LayerNorm` · `RmsNorm` · `BatchNorm` · `Im2Col` · `Col2Im` · `Conv2d` · `MaxPool2d` ·
`AvgPool2d` · `MaxPool2dBackward` · `IndexSelect` · `ScatterAdd` · `MseLoss` · `CrossEntropy` ·
`SgdStep` · `AdamStep` · `MaskedFill` · `Dropout` · `Triu` · `Tril`

An enum entry is a promise. Each of these is either implemented in this phase or removed.

---

## Phase plan

**P1 — Functional completeness.** Tensor system · runtime (allocator, buffers, images, queues,
sync, pipeline cache, descriptors) · all core operators · `nn` modules (Linear, Conv1/2/3D,
Embedding, Attention, MHA, FeedForward, PositionalEncoding, TransformerBlock, residuals) ·
autograd (graph, backward, accumulation, checkpointing, custom gradients) · optimisers (SGD,
Momentum, Adam, AdamW, RMSProp) · losses (MSE, CrossEntropy, BCE, KL, Huber) · data utilities
(Dataset, DataLoader, batching, shuffling, prefetch, transforms) · model save/load/checkpoint.

**P2 — Verification.** Every operator tested for forward, backward, edge cases, empty tensors,
broadcasting, mixed precision, layouts, large and random tensors. PyTorch is the correctness
reference unless a documented reason says otherwise; mismatches are investigated and fixed,
never ignored.

**P3 — Real applications.** MNIST MLP → MNIST CNN → CIFAR-10 CNN → ResNet → UNet → Transformer
encoder/decoder → tiny GPT → BERT inference → Llama inference. These become integration tests
for the whole library.

**P4 — Benchmarking.** Latency, throughput, memory, scaling, startup, GPU utilisation, CPU
overhead — on realistic workloads rather than synthetic ones.

**P5 — Study existing libraries.** PyTorch, TensorFlow, JAX, tinygrad, llama.cpp, ggml, ONNX
Runtime, TVM, XLA, MNN, NCNN, Candle, Burn, Eigen, oneDNN, XNNPACK. Architecture, abstractions,
memory management, scheduling, kernel organisation, API design, dispatch, graph execution,
testing, build systems. Learn techniques; do not copy implementations.

**P6 — Optimisation.** Profiling-driven and iterative, never one-time. Kernels, bandwidth, cache
locality, synchronisation, command recording, pipeline creation, descriptor reuse, graph
execution, fusion, memory planning, async execution.

**P7 — Engineering excellence.** Easy to understand, extend, debug, maintain, review, document.
No hardcoded values, no duplicated logic, reusable abstractions, modern C++.

**P8 — Community readiness.** Installation, tutorials, examples, API and architecture docs,
contributor guidelines, testing and benchmark docs, release process, versioning, templates,
CI/CD. Maintainable by a community, not one developer.

**P9 — Cross-platform.** Linux, Windows, macOS via MoltenVK, Android — with CI proving
consistent behaviour.

**Later:** distributed and multi-GPU training, quantisation, graph compilers, additional
backends, model zoo, visualisation, profiling and debugging tools, inference servers. Each
should build on the established architecture rather than requiring redesign.

---

## Engineering rules (mandatory)

These apply to every contribution regardless of size, and take precedence over implementation
speed. A feature implemented quickly with poor architecture is **incomplete**.

**R1 — Improve before expanding.** Review the surrounding implementation first. If it violates
the architecture or standards, improve it, then add the feature. Never build on a poor
abstraction because it already exists. Every contribution leaves the codebase better than found.

**R2 — Architecture before code.** Understand the existing architecture, ownership,
dependencies, data flow, object lifetimes, interfaces and expected extensions before writing.
Code should fit the design, not force the design to adapt afterward.

**R3 — Coding standards are mandatory.** RAII · Rule of Zero where possible, Rule of Five when
owning · const correctness · explicit ownership · explicit interfaces · separation of concerns ·
single responsibility · composition over inheritance · standard library over custom · no
duplicated logic · minimal macros · no hidden global state · no magic numbers · no hardcoded
limits · compile-time constants where appropriate · reusable utilities over feature-specific
helpers.

**R4 — Design for future features.** Ask: if someone adds three similar features next year, does
this design help or hinder them?

**R5 — Research before optimisation.** Every optimisation cycle begins with study of existing
implementations: why they decided as they did, what succeeded, what failed, what trade-offs and
assumptions they accepted, and whether those assumptions hold for vkML. Reference projects are
studied continuously, not once.

**R6 — Never copy blindly.** Understand the mechanism, hardware and compiler assumptions, memory
model, workload characteristics and applicability before adopting anything. Every imported idea
carries a documented justification.

**R7 — Every optimisation requires measurement.**
`profile → identify bottleneck → research → predict → implement → benchmark → validate
correctness → review maintainability → accept or reject`.
Benchmarks cover execution time, throughput, latency, memory, GPU utilisation, CPU overhead,
compilation impact where relevant, numerical correctness and reproducibility. If performance
does not improve meaningfully, or maintainability suffers without sufficient benefit, revert.
Measurement follows `MEASUREMENT-AUDIT.md` §7 — those rules were paid for.

**R8 — Continuous refactoring.** Part of everyday development, not a phase. Implement a cleaner
abstraction while the code is fresh. Small continuous improvements over large rewrites.

**R9 — Documentation is part of the implementation.** A feature is incomplete until it is
understandable: design decisions, algorithms, assumptions, limitations, trade-offs, extension
points. Readers need to know *why*, not only *what*.

**R10 — Long-term quality over short-term speed.** Optimise for the next ten years, not the next
benchmark.

**R11 — Professional Git history.** Small focused commits, one logical change each, imperative
mood, short and meaningful. Squash noisy or experimental commits before merging. Every commit
buildable and testable where practical, and understandable without opening the diff.
Good: `Add tensor broadcasting support` · `Implement Adam optimizer` ·
`Refactor Vulkan descriptor cache` · `Fix gradient accumulation bug`.
Unacceptable: `fix` · `update` · `changes` · `working` · `final` · `test` · `misc` · `asdf`.

**R12 — No AI attribution.** No `Co-authored-by`, no generated-by trailers, no automated
attribution or unnecessary signatures. History reflects engineering work, not tooling.

**R13 — One logical change per pull request.** Single objective. Do not combine unrelated
features, refactors, formatting and optimisations.

**R14 — Never commit broken code.** No commit that fails to compile, breaks tests, introduces a
known regression, or leaves incomplete functionality without clear isolation. Build, test,
format and static-analyse first. Every commit should raise confidence.

**R15 — Preserve a linear, understandable history.** A reader years later should be able to see
how the architecture evolved, why decisions were made, and when features, optimisations and
refactors landed. History is long-term engineering documentation.

---

## The four questions

Every engineering decision answers these before performance is considered:

1. Is it correct?
2. Is it maintainable?
3. Is it reusable?
4. Does it genuinely improve the library?

---

## Vision

vkML is not a collection of fast Vulkan kernels. It is a complete, modern, open-source machine
learning framework: correct by design, thoroughly tested, validated against established
frameworks, performant through evidence-driven optimisation, architected for long-term
evolution, maintainable by a community, easy to understand and extend, and capable of supporting
research and production workloads without sacrificing software quality.

Day-to-day standards are in `.claude/skills/cpp_spec/SKILL.md`.
