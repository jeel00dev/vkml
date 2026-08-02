# Project scope analysis

**Purpose.** Task #114 asserted that nothing in the repository defines what "complete" means.
This document tests that assertion against every planning document in the tree.

**Result: the assertion is wrong.** The repository contains an explicit, dated, status-marked
scope definition, an explicit list of what is deferred, and an explicit authority order. Most of
what #114 asked to be decided has already been decided and written down.

Everything below is labelled **STATED** (a direct quotation), **EVIDENCE** (something measured
or read in the code), **INFERENCE** (my reasoning, which may be wrong), or **OPEN** (a question
the repository genuinely does not answer).

---

## 0. Documents read

`README.md` · `CLAUDE.md` · `CONTRIBUTING.md` · `docs/PHASE2-MANIFESTO.md` ·
`docs/EXTENSIBILITY-ROADMAP.md` · `docs/M3_ROADMAP.md` · `docs/ARCHITECTURE.md` ·
`docs/GAP_ANALYSIS.md` · `docs/adr/0001`–`0009` · `docs/MILESTONE-B-REVIEW.md` ·
`docs/MILESTONE-D-REVIEW.md` · `docs/THEORY.md` · `docs/MEASUREMENT-AUDIT.md` ·
`docs/PERFORMANCE-MODEL.md` · `docs/EXTENSIBILITY-ROADMAP.md` · `docs/SKILLS-ARCHITECTURE.md`

---

## 1. What the repository states vkML is trying to become

**STATED** — `docs/PHASE2-MANIFESTO.md`, title and Mission:

> vkML Phase 2 — From Research to a Complete Machine Learning Library
>
> **Status:** active. Supersedes the M-series research objective as the project's primary goal.
> **Date:** 2026-07-27
>
> Build a complete, correct, usable, maintainable, production-quality machine learning library
> on top of the existing vkML foundation.

**STATED** — same document, Vision:

> vkML is not a collection of fast Vulkan kernels. It is a complete, modern, open-source machine
> learning framework: correct by design, thoroughly tested, validated against established
> frameworks, performant through evidence-driven optimisation, architected for long-term
> evolution, maintainable by a community, easy to understand and extend, and capable of
> supporting research **and production** workloads without sacrificing software quality.

**STATED** — `README.md`, the origin:

> Most GPU deep learning runs on NVIDIA cards because it runs on CUDA. AMD has ROCm, but it only
> supports some cards — and mine is not one of them. That is the problem that started this
> project.

**STATED** — `README.md`, current status: *"**Alpha.** vkML trains real models and is tested
hard … it does not cover everything a mature framework does."*

So of the seven candidate identities #114 proposed, the repository answers directly: **a
production ML framework, vendor-neutral via Vulkan, supporting both research and production
workloads.** It is *not* stated to be a research-only framework, an educational project, a
PyTorch replacement, or a mere backend. No document contradicts this.

---

## 2. Capabilities described as core requirements

**STATED** — `PHASE2-MANIFESTO.md` "Phase plan", P1 *Functional completeness*, quoted in full
because it is the closest thing the repository has to a completion checklist:

> Tensor system · runtime (allocator, buffers, images, queues, sync, pipeline cache, descriptors)
> · all core operators · `nn` modules (Linear, Conv1/2/3D, Embedding, Attention, MHA,
> FeedForward, PositionalEncoding, TransformerBlock, residuals) · autograd (graph, backward,
> accumulation, checkpointing, custom gradients) · optimisers (SGD, Momentum, Adam, AdamW,
> RMSProp) · losses (MSE, CrossEntropy, BCE, KL, Huber) · data utilities (Dataset, DataLoader,
> batching, shuffling, **prefetch, transforms**) · model save/load/checkpoint.

P2 verification · P3 real applications · P4 benchmarking · P5 study existing libraries ·
P6 optimisation · P7 engineering excellence · P8 community readiness · P9 cross-platform
(Linux, Windows, macOS via MoltenVK, Android).

**STATED** — P3 names the target model progression explicitly:

> MNIST MLP → MNIST CNN → CIFAR-10 CNN → ResNet → UNet → Transformer encoder/decoder → tiny GPT
> → BERT inference → **Llama inference**

### EVIDENCE — P1 measured against the code

**Now generated, not written.** `scripts/check_module_coverage.py` maps every P1 item to the
API symbol that satisfies it and fails if a declared target stops resolving. Run it:

```
30 of 34 P1 modules present, 4 not implemented
not implemented: Conv3d, DataLoader prefetch, DataLoader transforms,
                 autograd checkpointing
```

| P1 item | State |
|---|---|
| Linear, Conv2d, Embedding, BatchNorm2d, LayerNorm, Dropout, pooling, Flatten, Sequential | present |
| Conv1d | present as **`nn.Conv1d`** — composed from Conv2d with a height of 1, identical arithmetic |
| **Conv3d** | **absent** — needs a genuinely 3-D `im2col`, which does not compose |
| Attention / MultiHeadAttention | present as **`nn.MultiheadAttention`** |
| TransformerBlock | present as **`nn.TransformerEncoderLayer`** |
| FeedForward | present, **inside the encoder layer** as `linear1`/activation/`linear2` — where torch keeps it too |
| PositionalEncoding | present as **`nn.PositionalEncoding`** — sinusoidal; learned positions are `Embedding(max_len, d_model)` and need no module |
| SGD, Momentum, Adam, AdamW, RMSProp | present |
| MSE, CrossEntropy, BCE, KL, Huber | present — as functions (`mse_loss`, `cross_entropy`, `binary_cross_entropy_with_logits`, `kl_div`, `huber_loss`), not classes |
| Dataset, DataLoader, batching, shuffling | present |
| **DataLoader prefetch, transforms** | **absent** — signature is `(dataset, batch_size, shuffle, drop_last, seed, device)` |
| model save/load/checkpoint | present (`V.Checkpoint`) |
| **autograd checkpointing** | **absent** — `V.Checkpoint` is model serialisation, a different thing |

#### The same mistake, twice, for a permanent reason

The first pass of this table recorded **all five losses as absent** — a class-name grep
against an API that exposes them as functions. The correction was made and the cause
written down.

**The row above it was wrong for the same reason and was not caught.** *"Attention,
MultiHeadAttention, FeedForward, PositionalEncoding, TransformerBlock — absent"* was three
parts wrong: `MultiheadAttention` and `TransformerEncoderLayer` had existed since
`c12622b`, **five days before this document was written**, and the feed-forward block lives
inside the latter.

The condition is permanent, which is why it produced the same error twice. **vkML
deliberately uses PyTorch's spellings** — `nn.py` says so at the encoder layer: *"Parameter
names match `torch.nn.TransformerEncoderLayer` so a state_dict loads unchanged"* — and the
manifesto uses its own. A grep for either name is wrong about half the list, and will stay
wrong.

So the mapping between the two vocabularies is now **declared, in
`scripts/check_module_coverage.py`**, and the gate fails if a declared target stops
resolving. The remaining risk is the opposite one — an item built and nobody adding it to
the map — which is a one-line edit rather than a re-derivation, and which shows up as a
count that has not moved.

**Cross-checked against `README.md`**, which independently listed *"Missing layers: Conv1d,
Conv3d, gradient checkpointing"* — agreeing on three of the gaps and silent on the two
DataLoader ones. It now points at the generator rather than restating the list.

---

## 3. Capabilities explicitly optional / deferred

**STATED** — `PHASE2-MANIFESTO.md`, closing the phase plan:

> **Later:** distributed and multi-GPU training, quantisation, graph compilers, additional
> backends, model zoo, visualisation, profiling and debugging tools, inference servers. Each
> should build on the established architecture rather than requiring redesign.

This is an explicit deferral list, and it settles several tracker items without anyone deciding
anything new. **Quantisation and multi-GPU are stated as deferred, not core.**

**STATED** — `README.md` under "What does not [work]": *"No distributed training, no
quantisation, no ONNX."* Consistent with the above, and framed as a current limitation rather
than a permanent exclusion.

**INFERENCE** (flagged, because no document says it): ONNX import is not named in the "Later"
list nor anywhere in the manifesto. The nearest stated item is "graph compilers", which is not
the same thing. **ONNX's status is genuinely undetermined** — see §10.

---

## 4–7. Prerequisites, aspirations, experiments, implementation details

| Class | Items | Basis |
|---|---|---|
| **Prerequisite** | P1 functional completeness; P2 verification | STATED as the first two phases, and P3 depends on them |
| **Prerequisite (performance)** | P0 attribution | STATED in `EXTENSIBILITY-ROADMAP` §4a: *"Nothing here can be prioritised properly, because vkML cannot currently say which kernel costs what"* |
| **Aspirational** | P8 community readiness; P9 Android; the "Later" list | STATED as goals with no exit criteria and no date |
| **Experiment** | `EXTENSIBILITY-ROADMAP` in its entirety | STATED: *"**Status:** proposal, nothing implemented"* |
| **Implementation detail, not a goal** | M3.2 autotuning, M3.3 tile hierarchy, M3.5 epilogue fusion; split-K; f16 tile loads | INFERENCE — these are means to P6 optimisation. No document calls any of them a goal, and `EXTENSIBILITY-ROADMAP` §4a explicitly demotes M3's sixteen items as *"correctly researched and wrongly sequenced"* |

---

## 8. Contradictions between documents

Recorded, not reconciled.

**C1 — Sequencing of optimisation.** `PHASE2-MANIFESTO` places P6 Optimisation after P1–P5.
`EXTENSIBILITY-ROADMAP` §4a places performance first, stating *"tuning kernels that run for a
quarter of the step cannot fix a step that is three-quarters overhead"* and backing it with
measurements. **Not resolved by authority alone**: the manifesto is active and the roadmap is a
proposal, so the manifesto's ordering stands until someone changes it — but the roadmap's
evidence is the stronger argument and is newer. This is a real, live disagreement.

**C2 — Status of quantisation and multi-GPU.** The manifesto defers both to "Later".
`EXTENSIBILITY-ROADMAP` gives them numbered phases (5 and 7) inside its plan. Since that roadmap
is a proposal, this is a proposed promotion rather than a contradiction — but a reader
consulting only the roadmap would conclude they are planned work.

**C3 — Transformer modules.** The manifesto lists Attention/MHA/FeedForward/
PositionalEncoding/TransformerBlock under **P1 functional completeness**. `README.md`'s
"Missing layers" list names only Conv1d, Conv3d and gradient checkpointing, omitting all five.
**One of the two documents is wrong about the current state.** The code says the manifesto's
list is unmet; the README understates the gap.

**C4 — Android.** P9 names Android as a cross-platform target. `README.md`'s platform section
does not mention Android at all. No document records it being dropped.

---

## 9. Has the definition changed over time?

**STATED, by date and status field:**

| Date | Document | Status it declares |
|---|---|---|
| (M-series) | `M3_ROADMAP.md`, `THEORY.md`, `MEASUREMENT-AUDIT.md`, ADRs | superseded *as the primary goal*, explicitly retained as knowledge |
| 2026-07-27 | `PHASE2-MANIFESTO.md` | **active**; "Supersedes the M-series research objective" |
| 2026-08-01 | `EXTENSIBILITY-ROADMAP.md` | **proposal, nothing implemented** |

**Yes, once, and it is documented.** The project moved from a research objective (M-series) to a
library-completion objective (Phase 2), and the manifesto says so in its own header rather than
leaving it to be inferred. The later roadmap has not superseded anything; it says so itself.

---

## 10. Which document is authoritative

**`docs/PHASE2-MANIFESTO.md`.** Not by my judgement — by its own status field, which is the only
document in the tree that claims supersession, and by the fact that no later document claims to
supersede *it*. `EXTENSIBILITY-ROADMAP.md` disclaims authority in its second line.

The manifesto itself delegates: *"`THEORY.md`, `MEASUREMENT-AUDIT.md`, `PERFORMANCE-MODEL.md`,
`GAP_ANALYSIS.md` and the ADRs remain authoritative and continue to guide implementation."* So
the authority order is stated, not assembled:

```
PHASE2-MANIFESTO  (goals, scope, phase order)
  └─ THEORY / MEASUREMENT-AUDIT / PERFORMANCE-MODEL / GAP_ANALYSIS / ADRs  (how, and constraints)
       └─ EXTENSIBILITY-ROADMAP, M3_ROADMAP  (proposals and superseded research)
```

---

## What is therefore NOT an open question

Recorded so nobody re-opens them:

- **What vkML is becoming** — a complete, production-quality, vendor-neutral ML framework
  supporting research and production workloads. STATED.
- **Whether LLM support is in scope** — yes. P3 names "tiny GPT → BERT inference → Llama
  inference" as target applications. STATED.
- **Whether quantisation and multi-GPU are core** — no, both are in the "Later" list. STATED.
- **Which document wins** — the manifesto. STATED by its own status field.
- **Whether the project is research-only or educational** — neither; the Vision explicitly names
  production workloads. STATED.

---

## OPEN — questions the repository genuinely does not answer

These are the residual decisions, and they are much narrower than #114 assumed.

**O-A. Is the manifesto still current?** It is dated 2026-07-27 and marked active. Everything in
this analysis rests on that. If it has been silently outgrown, the analysis inherits the error.
*Only the maintainer can confirm this.*

**O-B. Does "complete" mean P1–P9 all done, or P1–P2 plus a chosen subset?** The manifesto gives
nine phases and objective criteria for none of them except P1, whose checklist is enumerable.
P8 ("community readiness") and P9 (Android) have no exit test. **This is the real residue of
#114.**

**O-C. C1 — does the roadmap's measured re-sequencing override the manifesto's phase order?**
The evidence favours attribution-and-overhead first; the authority favours P1 completeness
first. Both are defensible and the project cannot do both first.

**O-D. What is ONNX's status?** Named in `README` as a limitation, absent from the manifesto's
core list *and* from its "Later" list. It is the one roadmap item with no stated home.

**O-E. Is Android still a target?** P9 says yes; no other document mentions it.

**O-F. Is there a release/versioning definition?** No document defines 1.0, a release process, or
what would justify leaving Alpha. P8 lists "release process, versioning" as work to be done, so
the repository states that this is *missing*, which is itself an answer of a kind.

---

## Why no seven-way decision framework is offered

#114 asked for one *if* the repository failed to define completion. It did not fail. Producing a
framework asking whether vkML is "a research framework or a PyTorch replacement or an
educational framework" would invent a decision the manifesto has already made, and would create
exactly the second model of reality that `ENGINEERING-PRINCIPLES.md` §5 and this project's whole
documentation discipline exist to prevent.

The genuine decisions are O-A to O-F above, and they are scoped, not existential.

---

## What would change these conclusions

- **A document I did not read.** I listed the ones I did in §0; if a vision or planning document
  exists outside `docs/`, `README.md` and `CLAUDE.md`, this analysis is incomplete.
- **The manifesto being stale.** Its status field is the load-bearing evidence for the entire
  authority order (§10). A single sentence from the maintainer overrides all of it.
- **A dated document newer than 2026-08-01** claiming supersession — that would change §9 and
  §10 immediately.
- **Any of the P1 measurements in §2 being wrong.** They were taken by importing the built
  module on 2026-08-02; a stale extension would falsify them, which is the exact defect recorded
  as tracker task #113.
