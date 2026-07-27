# vkML Engineering Skills — Architecture

**Status:** design proposal, revision 3. Pre-implementation; no skill has been written.
**Date:** 2026-07-27
**Supersedes:** revisions 1 and 2 (same path, 2026-07-27)
**Scope:** the permanent engineering handbook for vkML, expressed as Claude Skills.

This document is the deliverable for the architecture phase: what skills should exist, why,
what each owns, when each fires, how they compose, where they live, and in what order they
should be built. It deliberately contains **no skill contents**.

Revision 2 incorporated seven directives from review and a completeness pass; three of them
exposed genuine defects rather than adding scope. Revision 3 is the result of a second,
adversarial review conducted against the whole design (§11.5). It changed no skill and no
boundary — which is the strongest evidence available that the catalog is settled — but found
eleven usability and longevity defects, of which the most consequential is that **the handbook
was readable only by an AI agent**, and therefore could not serve the 100-contributor future it
was designed for. §12 records every change.

---

## 0. Summary

Build **28 skills — 27 across 5 tiers plus 1 meta — with 3 deferred, and a lean always-loaded
constitution** (`CLAUDE.md`), stored in `vkml/.claude/skills/` and committed to the repository.

Six decisions carry the design:

| # | Decision | Why |
|---|---|---|
| 1 | **Skills are indexed by *moment of invocation*, not by topic** | A skill only helps if it loads at the right time. "Header design" and "API design" are different topics but one moment; "thread safety" is one topic but three moments. Topic-indexed skills mis-fire |
| 2 | **The constitution routes; it does not restate** | `CLAUDE.md` carries philosophy, priorities, the mandatory workflow, and a trigger→skill routing table. No engineering rule is duplicated there. The routing table is what makes on-demand loading safe (§3) |
| 3 | **Comprehensive ≠ monolithic: one trigger, many reference files** | `cpp-core` must cover 40+ topics while remaining a single unambiguous trigger. Progressive disclosure — a short router `SKILL.md` plus `references/*.md` loaded only when the specific question arises |
| 4 | **One rule, one home** | Every normative statement has exactly one owning skill, recorded in the Canonical Ownership Register (§6). Duplication is a defect, and its fix is a citation |
| 5 | **Skills encode *procedure*; `docs/` remains the source of truth for *facts*** | 5,156 lines of rigorous documentation already exist. Skills cite it, never restate it, or the two drift and the copy wins |
| 6 | **The handbook is a repository artifact, versioned with the code** | A 10-year project with contributors needs standards that arrive with the checkout and change under review, not standards that live in one developer's home directory |

Two skills exist that were in no requested list and that I consider load-bearing:
**`evidence-and-measurement`** (§5.4) — what may be asserted and how a number is made
trustworthy, merged into one canonical root because every one of the nine documented failures
was simultaneously a measurement error *and* a claim error; and **`skill-authoring`** (§5.5) —
governance of the handbook itself, without which a 28-skill system has no defined way to grow.

Three supporting artifacts are **not** skills, and it matters that they are not (§11.5):

| Artifact | Purpose | Why not a skill |
|---|---|---|
| `docs/ENGINEERING.md` | One-page human entry point: links into `.claude/skills/`, no content of its own | `.claude/skills/` is an agent-facing path a human contributor will never find. The skills are plain Markdown and perfectly readable; only discoverability is missing |
| `scripts/preflight.sh` | Runs format, tidy, layering, build and tests in one command | Mechanising a check makes it reliable *and* removes it from a checklist a human has to remember. What can be a script should never be a rule |
| `LICENSE` | Missing from the repository entirely | Found in the review; blocks any outside contribution |

---

## 1. What the project actually looks like today

The design is grounded in an audit of the repository, not on the topic list alone. Everything
below was read from the tree on 2026-07-27.

| Property | Value | Consequence for the handbook |
|---|---|---|
| Source | ~18,500 lines C++20 / GLSL / Python | Past the point where conventions can live in one head |
| `TODO` / `FIXME` / `HACK` markers | **0** | The "no intentional debt" rule is already being lived. Skills must preserve it, not introduce it |
| Documentation | 5,156 lines across 13 documents + 4 ADRs | The handbook's factual base already exists. Skills route to it |
| Milestones complete | M0–M3 (CPU oracle, Vulkan bring-up, elementwise/reductions, GEMM incl. split-K, GEMV, shape dispatch) | The high-frequency task is now *optimise and validate*, not *bootstrap* |
| Research stages | M3-R1…R5, M4-R1…R5 | A research cadence exists and is undocumented as a *process* |
| Tests | 105 Python parity tests + 8 C++ suites; CI asserts ≥250 collected | Testing discipline exists; its *rules* are implicit |
| Mechanical enforcement | `.clang-format`, `.clang-tidy`, `scripts/check_layering.py`, 4-preset CMake, 4-job CI | Style needs no skill — it is already a build failure |
| **Git commits** | **zero — the entire tree is untracked** | The single most urgent gap. 23,600 lines with no history, no bisectability, no recovery point |

Three findings shaped the catalog:

**1. The project's hardest-won knowledge is procedural, not factual.** `MEASUREMENT-AUDIT.md`
§7 is ten rules distilled from repeated failure. `THEORY.md` grades every law
*Proven / Strong / Hypothesis / Conjecture / Open*, states each law's scope, and keeps a
prediction scorecard recording five wrong predictions alongside ten right ones. `tolerance.py`
refuses an operation with no citable error bound. None of this is enforced by tooling, and all
of it is exactly what a skill is for.

**2. Style, formatting and layering are already mechanised.** A skill restating naming
conventions would duplicate `.clang-tidy` and could contradict it — which is why §11 recommends
*not* creating a coding-style skill despite it appearing in the requested list.

**3. Zero commits is a live risk, not a process nicety.** It keeps `git-workflow` in Phase 1.

---

## 2. Design principles

These are the rules the catalog was derived from, and the part of this document most worth
reviewing — the skill list follows from them.

### 2.1 Index by moment, not by topic

A skill is selected by matching its `description` against the task at hand. Its value is a
function of *when it fires*, not of how well its subject is organised.

- Topics that always co-occur become **one** skill. Adding an operator always requires its
  backward rule, so `autograd` is a reference file inside `op-authoring`, not a skill.
- One topic arising in unrelated moments is **referenced from several** skills, never
  duplicated (§6).
- Skills whose moment has not arrived are **deferred**, not written (§5.6).

### 2.2 Comprehensive through progressive disclosure

The brief asks for an "extremely comprehensive" C++ skill and correctly forbids monolithic
skills. These are reconciled by structure, not compromise:

```
cpp-core/
  SKILL.md              ~200 lines — the decision rules that apply to any C++ change,
                        plus a router table naming which reference answers which question
  references/
    ownership.md        RAII, rule of zero/five, smart pointers, lifetime, ADR-0001's model
    values-and-moves.md value semantics, move, copy elision, sink parameters
    templates.md        concepts, CRTP, static vs dynamic polymorphism, instantiation cost
    errors.md           exceptions, noexcept, exception-safety levels, error.h's policy
    constness.md        const/constexpr/consteval, immutability, thread-safety implications
    ub-and-strictness.md aliasing, alignment, signed overflow, sanitiser-visible classes
    performance-types.md layout, cache, false sharing, allocators, span/optional/variant/expected
```

`SKILL.md` is read whenever the skill fires; a reference only when its question is live. Depth
without cost, and each reference stays independently maintainable — which is what "single
responsibility" is protecting.

### 2.3 Skills cite documents; they do not restate them

`docs/` is authoritative for facts: measured hardware constants, tolerance policy, the law
book, ADR decisions. A skill that copies a number will eventually disagree with the document,
and the reader cannot tell which is stale.

Rule: **a skill may state a procedure, a threshold that governs the procedure, and a pointer.
It may not restate a measured fact.**

### 2.4 One rule, one home

The failure mode of a large skill set is not a gap; it is two skills answering one question
differently. Every cross-cutting concern therefore has exactly one owning skill, recorded in
the register at §6. Each catalog entry carries an explicit **Not this skill** boundary, and §10
gives the pairwise overlap resolutions.

When a skill needs a rule it does not own, it cites the owner. Restating is a defect with a
mechanical fix.

### 2.5 Skills must survive the author's absence

Written for a contributor with no context: no "as we discussed", no unexplained jargon without
a pointer, worked examples from the actual tree, and a stated review trigger — the event that
should cause the skill to be revisited.

### 2.6 The catalog has a size budget

28 active skills is near the practical ceiling for reliable trigger matching. **Cap: 32.**
Beyond that, the answer to "we need to cover X" is a reference file inside an existing skill,
or a consolidation — never a 33rd skill. `skill-authoring` (§5.5) owns enforcement.

---

## 3. The constitution

`vkml/CLAUDE.md`, always loaded, **~120–150 lines**, containing exactly six sections:

| § | Content | Not this |
|---|---|---|
| 1 | What vkML is; the one-paragraph philosophy | Architecture detail — that is `docs/ARCHITECTURE.md` |
| 2 | Engineering priorities, **ordered**, including the numerics veto | The numerical policy itself |
| 3 | The mandatory workflow every change follows (§3.1) | Any step's internals |
| 4 | **The routing table**: trigger → skill (§3.2) | The skills' content |
| 5 | Where the authoritative documents live, one line each | Their content |
| 6 | High-level principles: no intentional debt, no unmeasured claims, no restated facts | Rules that implement them |

No measurement rules, no C++ rules, no tolerances, no testing rules, no documentation rules.

### 3.1 Why routing replaces duplication

Revision 1 justified a constitution by arguing that some invariants are unsafe to leave behind
an on-demand trigger — *never loosen a tolerance to make a test pass* has already caused damage
by the time a skill would have fired. That argument was sound but the conclusion was wrong: the
fix is not to copy the rule into the constitution, it is to **guarantee the skill fires**.

That is the routing table's job, and it is load-bearing rather than decorative. Step 1 of the
mandatory workflow is *consult the routing table before starting*, which converts "the skill
might not trigger" into "the skill is looked up deliberately."

**Residual risk, stated honestly:** a task whose phrasing matches no row can still miss a rule.
Three mitigations, in order of strength: the routing table's rows are phrased as *symptoms*
("a parity test is failing", "results changed") rather than as skill names; `implementation-checklist`
(§5.5) runs before any work is declared complete and re-checks the same ground from the
opposite direction; and `code-review` is a third pass. A rule has to slip past all three to
escape.

### 3.2 The mandatory workflow

Section 3 of the constitution states this sequence and nothing more; each step names its
owning skill.

```
1. Understand      what is actually being asked; consult the routing table
2. Fit             does the current architecture support this?          → architecture-evolution
                     obviously yes → say so in one line and continue
                     resists       → escalate: extend cleanly, or refactor first
3. Design          for anything expensive to reverse                    → decision-records
4. Implement       following the standards of the owning skills
5. Verify          scripts/preflight.sh, then whatever the change needs → testing, numerics,
                                                                          evidence-and-measurement
6. Review          the completion gate, scaled to the change            → implementation-checklist
7. Record          commit, docs, ADR or stage record as applicable      → git-workflow
```

Step 2 is the architectural gate, and it sits before implementation because that is the only
point at which "refactor first" is still cheap. **It is deliberately cheap by default.** Most
changes — a new elementwise op, a new test, a bug fix — obviously fit, and the honest answer is
one line. A gate that demands a full analysis every time is ceremony, ceremony gets routed
around, and then the gate is absent on the one change that needed it. `architecture-evolution`
owns the escalation triggers that distinguish the two cases (§5.2).

Steps 5 and 6 are scaled the same way, for the same reason (§5.5).

---

## 4. Layering of the handbook

The skills form a dependency stack mirroring — deliberately — the way `THEORY.md` layers its
laws. Higher tiers may cite lower tiers; the reverse is a design error.

```
    M  META           governance of the handbook itself       (skill-authoring — off to the side,
        ┆                                                      references every tier, cited by none)
        ┆
   T5  PROCESS        how work is proposed, evidenced, verified, reviewed, recorded, shipped
        |
   T4  CORRECTNESS    what makes a result believable: numerics, tests, evidence, theory
    &   PERFORMANCE
        |
   T3  DOMAIN         vkML's subject matter: tensors, ops, kernels, Vulkan, memory, bindings
        |
   T2  ARCHITECTURE   where code goes, how it may depend, how it evolves, how change is recorded
        |
   T1  CRAFT          how a line of C++ is written
        |
   T0  CONSTITUTION   philosophy, priorities, workflow, routing        (CLAUDE.md, always loaded)
```

The meta tier is drawn detached deliberately. `skill-authoring` governs the container and must
be able to reference anything; if it sat inside T5 it could be cited by its peers, and the
layering would be circular.

The layering has the diagnostic value it has in `THEORY.md`: a T3 skill that needs to restate a
T1 rule signals that the T1 rule is wrong or the boundary is misdrawn.

---

## 5. The skill catalog

28 active skills. Each entry gives its **responsibility** (one sentence — the single thing it
owns), **why it exists**, **trigger** (the moment, phrased as the `description` will be),
**not this skill** (the boundary), and **references** where progressive disclosure is warranted.

---

### 5.1 Tier 1 — Craft (3)

#### T1-1 · `cpp-core` — *Phase 1*
- **Responsibility.** How to write a line of C++ in this codebase: ownership, value semantics,
  templates, error handling, const-correctness, UB avoidance, and performance-aware types.
- **Why.** The most frequently needed skill in the set and the one that most determines whether
  the code would be accepted upstream in a project of LLVM's or Qt's standard. Promoted to
  Phase 1 on review: its purpose is not to repair bad code but to **prevent architectural drift
  as the codebase grows from 20k toward 100k+ lines**. Consistency is cheapest to establish
  before the growth, not after — and unlike formatting, none of this is mechanically enforceable.
- **Trigger.** Writing, modifying or reviewing any C++ — declaring a type, choosing a parameter
  or return form, adding a member, introducing a template, deciding `noexcept`.
- **Not this skill.** Which file the code goes in (`header-and-file-design`); whether it is
  public (`api-design`); which layer may call it (`architecture-and-layering`); formatting
  (`.clang-format`, mechanically enforced).
- **References.** 7 files per §2.2.

#### T1-2 · `header-and-file-design` — *Phase 4*
- **Responsibility.** Translation-unit structure: header vs source, include hygiene, forward
  declarations, circular-dependency avoidance, PImpl, header-only trade-offs, compile-time cost.
- **Why.** A distinct moment from writing the code — asked when a *file* is created or an
  include is added, and answered by different criteria (build time, ABI surface, coupling).
  vkML already has `include/vkml/` vs `src/` as a hard public/private boundary and an
  internal-header convention (`src/backend/vulkan/vk_device.h`); the rule is implicit.
- **Trigger.** Creating a file, adding an `#include`, splitting or merging a translation unit,
  investigating build time, considering PImpl.
- **Not this skill.** Language constructs inside the file (`cpp-core`); whether a header may
  depend on another layer (`architecture-and-layering`).
- **Reviewed for merger into `cpp-core`; kept separate — see §11.3.**

#### T1-3 · `api-design` — *Phase 4*
- **Responsibility.** The contract of anything visible outside its module: public vs internal,
  naming semantics, parameter and error contracts, PyTorch-surface parity, stability promises,
  deprecation.
- **Why.** vkML has three surfaces with different stability obligations — `include/vkml/api/`
  (C++ public), the nanobind module (Python public), and every `backend/api/` interface (an
  internal contract with two implementations and more to come). The rules differ per surface
  and are unwritten. ADR-0001's guardrail 1 — *`Node` must not appear in the public Tensor API*
  — is exactly this skill's subject.
- **Trigger.** Adding or changing anything in `include/vkml/`, the Python surface, or a
  `backend/api/` interface; naming a user-visible symbol; deciding whether something is public.
- **Not this skill.** The binding mechanism (`python-bindings`); versioning and deprecation
  *windows* (`release-and-compat`, deferred); layer permissions (`architecture-and-layering`);
  evolving an interface that has live callers (`architecture-evolution`).

---

### 5.2 Tier 2 — Architecture (4)

#### T2-4 · `architecture-and-layering` — *Phase 3*
- **Responsibility.** The *static* architecture: where new code belongs, what it may depend on,
  how subsystem seams are drawn — including coupling/cohesion, extension points, backend seams,
  and the project-wide threading and capability policies.
- **Why.** The layer rule from `ARCHITECTURE.md` §4.1 is enforced by `check_layering.py` for
  *includes only*. It cannot see a design that satisfies the include graph while violating its
  intent, and it offers no guidance on where a new subsystem should sit. The script catches
  violations; this skill prevents them. It also carries today's obligation toward future
  backends: keeping the `backend/api` seam clean enough that a third implementation is an
  addition rather than a redesign (§5.6, `backend-porting`).
- **Trigger.** Adding a subsystem, directory or backend; deciding which layer owns a
  responsibility; resolving a dependency the layering check rejects; designing an extension point.
- **Not this skill.** Deciding whether the current architecture can absorb a specific feature
  (`architecture-evolution`); recording the decision (`decision-records`); file placement within
  a settled layer (`header-and-file-design`).

#### T2-5 · `architecture-evolution` — *Phase 3* · **new in revision 2**
- **Responsibility.** The gate before implementation: can the existing abstractions support this
  feature? — recognising architectural smells, deciding extend-vs-refactor-first, and evolving
  interfaces that have live callers without accumulating debt.
- **Why.** Requested at review, and the request exposed a real defect: revision 1's
  `refactoring-and-evolution` conflated the *decision* ("should this design change?") with the
  *execution* ("how do I change it safely"). Those are different moments with different inputs —
  the decision happens before any code is written, when refactoring is still cheap; the execution
  happens after. Merging them means the decision is made by whoever is already mid-implementation,
  which is exactly when "extend the existing thing" always wins. The flow this skill owns:

  ```
  need a feature → review the current design → can existing abstractions carry it?
        yes → extend, and leave the abstraction no worse
        no  → refactor the architecture first, then implement
  ```

  The rule it enforces: **the project never builds on a poor abstraction merely because the
  abstraction already exists.** Every feature must improve or preserve architectural quality.
  It also owns the counter-rule, which matters just as much: **when *not* to refactor** — a
  smell inside a stable, well-tested, single-caller component with no pending feature is not a
  reason to spend the risk budget. Without that half, the skill licenses unbounded churn.
  It owns a third thing, added in revision 3 and just as load-bearing: **the escalation
  triggers** — the short list of signals that turn the cheap default answer into a real
  analysis. A feature needing a special case in an existing abstraction to be accepted; a
  parameter added only to select behaviour for one caller; a change forcing an edit in a layer
  that should not have known about it; the same conditional appearing in a third place. Absent
  a trigger, step 2 of the workflow is one line and the work proceeds.
- **Trigger.** Starting any feature that does not fit cleanly; a design that resists a change;
  an abstraction that needs a special case to accept a caller; deciding whether to refactor;
  changing an interface with existing callers; assessing coupling, modularity or extensibility.
- **Not this skill.** The static rules and target architecture (`architecture-and-layering`);
  the safe-change mechanics once the decision is made (`refactoring-mechanics`); recording the
  decision (`decision-records`).
- **References.** `smells.md` (the catalog, each with the vkML-specific signal that reveals it),
  `extend-or-refactor.md` (the decision procedure and its evidence requirements),
  `interface-evolution.md` (changing a contract with live callers).

#### T2-6 · `decision-records` — *Phase 4*
- **Responsibility.** When a decision requires an ADR, the format it takes, and how decisions
  are superseded rather than silently reversed.
- **Why.** Four ADRs exist and share a distinctive, unusually good structure: *Context →
  Measurements → Options considered (each with a verdict) → Decision → Guardrails adopted now →
  Rejected micro-optimisations, recorded so they are not rediscovered*. ADR-0001 is the model —
  it benchmarks the alternatives before choosing. That last section is rare and valuable; it is
  how a decade-long project avoids relitigating settled questions. The format is currently
  transmitted by imitation.
- **Trigger.** Making a decision expensive to reverse, affecting a public contract, changing an
  invariant, or reached against an intuitive alternative; revisiting or superseding an ADR.
- **Not this skill.** Stage records for experiments (`optimization`); prose (`documentation`);
  milestone gates (`milestone-planning`).
- **Templates.** `templates/adr.md`.

#### T2-7 · `refactoring-mechanics` — *Phase 5* · **renamed and narrowed in revision 2**
- **Responsibility.** Executing a change to existing code safely: behaviour-preservation proof,
  small-step protocol, staged migration of a contract with live callers, moving code across
  layers, and debt recording when a cleanup must be deferred.
- **Why.** Once `architecture-evolution` has decided *that* something changes, this is *how* —
  and vkML has an unusually strong tool for it. A refactor that changes no fold order must
  produce byte-identical goldens, which converts "did I preserve behaviour?" from a judgement
  into a check. That is a stronger guarantee than most projects can offer and it deserves a
  documented procedure.
- **Trigger.** Performing a behaviour-preserving change; splitting or moving code; migrating
  callers off an interface; recording deferred cleanup.
- **Not this skill.** Whether to refactor at all (`architecture-evolution`); whether the change
  is numerically neutral (`numerics-and-determinism`); demanding a refactor in someone else's
  change (`code-review`).

---

### 5.3 Tier 3 — Domain (6)

#### T3-8 · `tensor-and-graph-semantics` — *Phase 3*
- **Responsibility.** Invariants of the central data structures: rank/extent/byte-stride
  algebra, broadcasting, views and aliasing, dtype promotion, node immutability, graph
  ownership, lazy-vs-eager realisation.
- **Why.** Every op, kernel and backend must honour these, and a violation surfaces far from its
  cause. They are spread across `shape.h`'s comments, ADR-0001, ADR-0003 and `ARCHITECTURE.md`
  §4.2, with real subtleties in each: row-major *against* ggml's convention; `is_contiguous`
  ignoring extent-1 axes; `reshaped()` returning `nullopt` rather than silently copying, so the
  copy stays visible in the graph.
- **Trigger.** Any work touching `Shape`, `Storage`, `Node`, `Graph`, views, broadcasting, dtype
  rules or realisation semantics.
- **Not this skill.** Adding an operator over these structures (`op-authoring`); memory placement
  (`memory-and-allocation`); C++ ownership mechanics in the abstract (`cpp-core`).

#### T3-9 · `op-authoring` — *Phase 3*
- **Responsibility.** The complete recipe for adding or changing an operator: CPU oracle first,
  shape/dtype rules, the backward rule as graph nodes, dispatch registration, tolerance entry,
  test matrix — in order, with gates.
- **Why.** The highest-frequency multi-step task, currently reconstructed from first principles
  each time. It spans eight files across six layers and the ordering is load-bearing: the CPU
  oracle must exist before the Vulkan kernel, or a mismatch cannot be attributed
  (`ARCHITECTURE.md` §7.1). Backward rules are folded in rather than split out because an
  operator without its gradient is an incomplete operator — always the same moment.
- **Trigger.** Adding an operator; changing an operator's semantics, shape rules or gradient;
  extending one to a new dtype or backend.
- **Not this skill.** GLSL implementation (`kernel-authoring`); the tolerance *policy*
  (`numerics-and-determinism`); harness mechanics (`testing`).
- **References.** `backward-rules.md`, `dispatch-registration.md`, `op-checklist.md`.

#### T3-10 · `kernel-authoring` — *Phase 2*
- **Responsibility.** Device-side code: GLSL conventions, specialisation constants over defines,
  push-constant layout shared with C++, subgroup and LDS usage, deterministic reduction
  structure, tile-geometry vocabulary.
- **Why.** The most performance- and correctness-critical code in the project, with rules that
  exist nowhere else: no global float atomics (hardware-forced), pairwise fold with
  `kPairwiseBlock = 32` mirrored in `tolerance.py`, spec constants preferred over `#define` to
  keep one SPIR-V blob, `scalarBlockLayout` so one header describes both sides of the push
  constant. `THEORY.md` law A1 — the 64-float private-array budget — is a hard authoring
  constraint that no API exposes and that cost several stages to discover.
- **Trigger.** Writing or modifying any `.comp` shader; changing tile geometry; adding a
  specialisation constant; changing a push-constant layout.
- **Not this skill.** Host-side pipeline and dispatch machinery (`vulkan-backend`); whether a
  fold-order change is permitted (`numerics-and-determinism`); tuning it (`optimization`).
- **References.** `tile-geometry.md`, `determinism-in-kernels.md`, `push-constants.md`.

#### T3-11 · `vulkan-backend` — *Phase 4*
- **Responsibility.** Host-side Vulkan: device and capability negotiation, resource lifetime,
  synchronisation, command recording, pipeline creation and caching, validation-layer
  discipline, pipeline-statistics capture.
- **Why.** Vulkan's failure modes are silent, and the correct-but-unusual choices here must be
  defended in writing so a future contributor does not "fix" them: no descriptor sets at all
  (BDA in push constants), one global barrier rather than per-buffer barriers, timeline
  semaphores instead of fence pools. Capability negotiation is a live hazard —
  `shader_core_count` is 0 on non-AMD hardware, and `MEASUREMENT-AUDIT.md` §6.3 requires every
  consumer to *decline* an occupancy decision rather than guess.
- **Trigger.** Any change under `src/backend/vulkan/`; adding a Vulkan feature dependency;
  debugging a validation error, device loss or synchronisation hazard.
- **Not this skill.** Shader source (`kernel-authoring`); allocator policy
  (`memory-and-allocation`); interpreting pipeline statistics as evidence
  (`evidence-and-measurement`).

#### T3-12 · `memory-and-allocation` — *Phase 4*
- **Responsibility.** The memory budget as a design constraint: the two-level model (device
  suballocator, graph offset planner), lifetime classes, staging, alignment, leak accounting,
  peak-VRAM prediction.
- **Why.** 5.75 GiB total with only 256 MiB host-visible and no resizable BAR is the constraint
  that shaped six of the architecture's decisions. Memory questions cut *across* the layer stack
  — the planner is backend-agnostic, the suballocator is Vulkan, staging is a host concern — so
  no single layer skill can own it.
- **Trigger.** Allocating device memory; changing the planner, allocator or staging path;
  investigating an OOM, a leak or peak-usage growth; adding a persistent buffer.
- **Not this skill.** Host-side C++ ownership (`cpp-core`); Vulkan API mechanics
  (`vulkan-backend`); whether a memory change altered results (`numerics-and-determinism`).

#### T3-13 · `python-bindings` — *Phase 5*
- **Responsibility.** The C++/Python boundary: object lifetime across it, GIL discipline,
  exception mapping, buffer-protocol and dlpack interop, type stubs, keeping the Python surface
  PyTorch-shaped.
- **Why.** Where two ownership models meet, and the reason `error.h` throws rather than aborts —
  a `GGML_ASSERT`-style abort would take down a user's interpreter. ADR-0001 turns on this:
  `shared_ptr` was retained *because* Python holds tensors for unpredictable lifetimes. Lifetime
  errors here are use-after-free, not exceptions.
- **Trigger.** Changing `bindings/module.cpp` or `python/vkml/`; exposing a C++ type to Python;
  changing exception mapping; adding array interop.
- **Not this skill.** What the API should *be* (`api-design`); Python test structure (`testing`).

---

### 5.4 Tier 4 — Correctness and performance (6)

#### T4-14 · `evidence-and-measurement` — *Phase 1* · **merged in revision 2**
- **Responsibility.** The canonical source for what may be asserted and how a number is made
  trustworthy: claim classes, the confidence vocabulary, benchmarking rules, instrument
  validity, the noise floor, A/B protocol, prediction-before-experiment discipline, and
  baseline governance.
- **Why.** The highest-value skill in the set by the project's own history, and merged from
  revision 1's separate `measurement` and `evidence-standards` because **every one of the nine
  documented failures was simultaneously a measurement error and a claim error** — a number was
  taken wrongly and then asserted confidently. Splitting them lets a contributor load one and
  not the other at exactly the moment both are needed. `GAP_ANALYSIS.md` §1.1 shows the pattern
  in its severe form: Stage 8 did not discover that a 4×4 register block is bad on Navi 10; it
  discovered that 4×4 *plus a 6-deep carry stack* is bad, and only the second statement is true.
  `MEASUREMENT-AUDIT.md` opens by observing that confident wrong numbers survive review. Its ten
  rules are non-obvious and individually expensive to rediscover: GPU timestamps are ~20× more
  reproducible than wall clock; report the minimum, never the mean; never sum per-dispatch
  timestamps across independent dispatches; wall clock is inadmissible unless the operation
  dominates the window. The same document's opening move — when an instrument fails, audit the
  instrument and write the rule — is also owned here.
- **Trigger.** Taking any measurement; comparing two configurations; interpreting a benchmark;
  a surprising result; reporting any result; claiming something is done, fixed, faster or
  correct; writing a conclusion in any document; updating a baseline.
- **Not this skill.** What to do with the result (`optimization`); promoting it to a law
  (`performance-theory`); correctness comparison of *values* (`testing`).
- **References.** `claim-classes.md` (measured / derived / cited / inferred / assumed, and what
  each requires), `benchmarking.md` (the ten rules and their derivations), `instruments.md`
  (per-instrument distortions, including host-side profiling as the graph layer grows —
  ADR-0001 measured graph overhead at ~20 % of step time for a 5,000-node graph), `baselines.md`.
- **This skill's rules are cited, never restated, by:** `optimization`, `performance-theory`,
  `testing`, `kernel-authoring`, `build-and-ci`, `implementation-checklist`, and every skill
  that reports a result.

#### T4-15 · `numerics-and-determinism` — *Phase 2*
- **Responsibility.** The numerical contract: tolerance as a property of the operation with a
  citable source, the four error kinds, bit-reproducibility, fold-order changes, golden
  re-pinning.
- **Why.** vkML's defining guarantee, and the one place it diverges from unanimous production
  practice (`GAP_ANALYSIS.md` I7 — no studied project offers bit-reproducibility). It is also
  the guarantee most easily lost by accident: any change to reduction order silently breaks it.
  `tolerance.py`'s docstring records two occasions where a *test* was wrong rather than the
  code, and the rule that emerged — no tolerance without a citable bound — must govern every
  future change, not just Python tests.
- **Trigger.** Anything that could alter floating-point results: reduction order, accumulator
  precision, fold structure, split-K, fusion, a new tolerance, a failing parity test.
- **Not this skill.** Where the test lives (`testing`); whether the change was faster
  (`evidence-and-measurement`); kernel-level implementation of a fold (`kernel-authoring`).
- **References.** `tolerance-policy.md`, `fold-order-changes.md` (re-derivation and golden
  re-pinning), `determinism-guarantee.md`.
- **Holds a veto over every performance skill (§7.1).**

#### T4-16 · `testing` — *Phase 2*
- **Responsibility.** What test to write, at which tier, against which oracle, and where it
  lives — the three-way oracle chain, property tests, regression goldens, resource tests,
  vacuity checking.
- **Why.** The nine-tier plan in `ARCHITECTURE.md` §7.4 is a plan, not a procedure; it does not
  answer "I changed X, what must I test?" The oracle chain (vulkan ↔ cpu ↔ torch) is the central
  correctness mechanism and its *ordering* is what makes a failure attributable.
  `MEASUREMENT-AUDIT.md` rule 10 — check every correctness gate for vacuity before trusting a
  pass — is enforced in CI for collection count only; it needs to be a habit, since a test that
  asserts nothing is worse than no test.
- **Trigger.** Adding or changing a test; deciding what coverage a change needs; a test failing;
  fixing a bug (which must produce a regression test).
- **Not this skill.** Tolerance values (`numerics-and-determinism`); benchmark validity
  (`evidence-and-measurement`); CI job configuration (`build-and-ci`).
- **References.** `oracle-chain.md`, `test-tiers.md`, `vacuity.md`.

#### T4-17 · `optimization` — *Phase 2*
- **Responsibility.** The optimisation loop: profile first, form a falsifiable hypothesis with a
  quantified prediction, state falsification criteria *before* measuring, gate on resources
  before timing, verify correctness, accept or roll back.
- **Why.** vkML has already evolved a stronger loop than most projects — visible in
  `M3-01-TILE-GEOMETRY.md`, which states its prediction, expected compiler statistics and
  falsification criteria in §3 and only reaches benchmarks in §8. The project-specific rule that
  must not be lost: **gate on `PipelineStats` before timing** (`GAP_ANALYSIS.md` §5.3) — Stage
  8's regression was fully visible in scratch/spill statistics before a single benchmark ran, a
  capability no studied production project has.
- **Trigger.** Any work whose purpose is speed; a performance regression; choosing among
  optimisation candidates; writing a stage record.
- **Not this skill.** Measurement methodology or claim discipline — **cited from
  `evidence-and-measurement`, never restated**; whether the change is numerically permitted
  (`numerics-and-determinism`, which has a veto); kernel implementation (`kernel-authoring`).
- **References.** `pruning-before-timing.md`, `rollback.md`; `templates/stage-record.md`.

#### T4-18 · `performance-theory` — *Phase 2*
- **Responsibility.** The **epistemics of performance knowledge**: what the project is entitled
  to believe about how this machine behaves, how a belief is promoted or demoted, how far it may
  be generalised, and how failed predictions are kept on the record. `THEORY.md` is its artifact,
  not its subject — the distinction matters, because framed as document maintenance this skill
  would belong in `documentation` and would lose the one property that makes it work.
- **Why.** `THEORY.md` is a research instrument, not a summary — laws are graded, every law
  carries an explicit scope (universal / architecture-specific / driver-specific /
  vkML-specific), and the scorecard records ten correct and five wrong predictions with the note
  that *a theory that only ever confirms itself is not being tested hard enough*. Keeping that
  honest needs a procedure; the natural drift is toward quietly upgrading confidence and
  forgetting failures. Kept separate from `optimization` because the two have opposite biases —
  optimization wants a result, theory wants a falsification — and merging lets the first quietly
  consume the second.
- **Trigger.** A measurement that confirms or contradicts a stated law; proposing a new law;
  changing a law's confidence or scope; asking what the theory predicts or forbids.
- **Not this skill.** Taking the measurement or the confidence vocabulary itself
  (`evidence-and-measurement`); acting on the prediction (`optimization`).

#### T4-19 · `debugging-and-diagnostics` — *Phase 3*
- **Responsibility.** Finding a defect, and the facilities that make it findable: the diagnosis
  protocol, eager mode, validation layers, sanitisers, bisection, plus logging, assertions,
  error-message quality and debug env-var conventions.
- **Why.** The two halves are one moment — the diagnostic is usually added *because* something
  is being hunted. The project-specific first question is unusual and earned: **before assuming
  an implementation bug, rule out a measurement error and a wrong test.** Both have a documented
  history here. The instruments are also non-obvious: `VKML_EAGER=1` forces per-op realisation
  so a failure points at the offending op rather than the realise boundary; the ASan preset pins
  clang; validation layers must never be on while benchmarking.
- **Trigger.** A test failing, a crash, a wrong number, a hang, a validation error; adding a
  log, assertion or diagnostic switch; improving an error message.
- **Not this skill.** Writing the regression test that follows the fix (`testing`); Vulkan
  synchronisation semantics (`vulkan-backend`).

---

### 5.5 Tier 5 — Process (8) · and the meta tier (1)

#### T5-20 · `implementation-checklist` — *Phase 1* · **new in revision 2**
- **Responsibility.** The systematic verification an implementation must pass before it is
  declared complete — the canonical checklist, **risk-scaled**, with the evidence each item
  requires.
- **Risk scaling (revision 3).** A uniform 35-item checklist run on a typo fix teaches people
  to skip it, and a checklist that is skipped is worse than none because it creates false
  assurance. The checklist is therefore structured in three parts:
  **(a) Mechanical** — `scripts/preflight.sh`: format, tidy, layering, build, tests. Never a
  human judgement, never an item on a list, and it either passes or it does not.
  **(b) Universal** — four judgements that apply to every change, however small: is every claim
  in the summary supported by evidence actually produced; is the commit one logical change; is
  anything left unfinished stated rather than implied; does the change leave the surrounding
  code no worse.
  **(c) Conditional sections** — activated by what the change touched. Touched a public header
  → the API section. Touched arithmetic → the numerics section. Added a kernel → the
  performance and determinism sections. Each section names its owning skill instead of
  restating its rules, so the checklist stays a *router with evidence requirements* rather than
  a second copy of the handbook.
  The sections cover the categories the review specified: architecture (single responsibility,
  abstraction, coupling, cohesion) · C++ (ownership, lifetime, RAII, rule of zero/five,
  exception safety, move semantics, header organisation, includes, thread safety, const
  correctness) · performance (bottlenecks, unnecessary allocations, cache friendliness, whether
  profiling was required) · testing (unit, integration, regression, edge cases, PyTorch parity,
  vacuity) · documentation (comments, public API, design rationale, extension points) ·
  maintainability (readability, naming, reusability, extensibility) · git (commit boundaries,
  message quality, build and test verification).
- **Why.** Requested at review, and it fills the gap that makes everything else stick. Without
  a completion gate the other 27 skills are advisory: they fire while work is in progress and
  nothing re-checks the result. This is also the highest-frequency moment in agent-driven
  development — *am I actually done?* — and the point at which the brief's final goal ("every
  future implementation naturally follows professional practice") is either met or not.
  Splitting it from `code-review` is deliberate: this skill owns **what "good" means**, and
  `code-review` owns **how to adjudicate someone's work against it**. One checklist, two
  entry points, no duplication.
- **Trigger.** Declaring any implementation complete; self-review before committing; asking
  whether work is finished; final verification before handing off.
- **Not this skill.** The standards themselves (each owning skill); adjudicating another's
  change (`code-review`); commit mechanics (`git-workflow`).
- **References.** One file per conditional section, so only the relevant ones are loaded.

#### T5-21 · `code-review` — *Phase 3*
- **Responsibility.** Adjudicating a change against the handbook: the severity model (what
  blocks a merge vs what is advisory), how to demand a surrounding-code fix, how to review an
  external contribution, and review etiquette.
- **Why.** The point at which standards are enforced on work you did not write, and the
  mechanism by which they survive contributors who have not read them. Needs an explicit
  severity model, because a review treating a layering violation and a naming preference as
  equally blocking teaches contributors to ignore both. Narrowed in revision 2: the checklist
  content moved to `implementation-checklist`, leaving this skill the adjudication — which is
  the part that genuinely differs when the author is someone else.
- **Trigger.** Reviewing a diff, PR or branch; responding to review feedback; triaging an
  external contribution.
- **Not this skill.** What to check (`implementation-checklist`); merge mechanics
  (`git-workflow`); demanding a refactor's justification (`architecture-evolution`).

#### T5-22 · `git-workflow` — *Phase 1*
- **Responsibility.** History as an engineering artifact: commit granularity and message form,
  branch and milestone naming, bisectability, what must never be committed, and the relationship
  between commits and gates.
- **Why.** **Urgent.** The repository has zero commits: 18,500 lines of source and 5,156 lines
  of documentation exist only as untracked files. No recovery point, no bisection, no
  attribution, and no way to answer "when did this number change?" — which the measurement
  discipline depends on. Beyond the emergency, a project whose central claim is
  bit-reproducibility needs commits that pin code, goldens and baselines together.
- **Trigger.** Committing, branching, merging, writing a commit message, recovering work, or
  investigating when a behaviour changed.
- **Not this skill.** What CI runs on the commit (`build-and-ci`); review (`code-review`);
  release tagging (`release-and-compat`, deferred).

#### M-23 · `skill-authoring` — *Phase 1 exit* · **new in revision 2; meta-tier in revision 3**
- **Tier.** **Meta** — outside the T1–T5 dependency stack, because it governs the *container*
  rather than any engineering question. It may reference every tier including the constitution,
  and nothing references it. Marking it inside T5 in revision 2 implied it could be cited by
  peers, which would have made the layering circular.
- **Responsibility.** Governance of the handbook itself: when a new skill is justified, when one
  should split or merge, the required structure of a `SKILL.md`, how the Canonical Ownership
  Register is maintained, the catalog size cap, **the trigger-quality standard, and the
  handbook's maintenance cadence**.
- **Trigger quality (revision 3).** The architecture's stated main risk is trigger reliability
  at 28 skills (§11.4), and revision 2 mitigated it only with catch passes. That is defence
  after the fact. The standard attacks it at the source, and is a hard requirement on every
  skill: a `description` must (a) name **symptoms**, not topics — "results changed between
  runs", not "determinism"; (b) state **do not use when**, naming the neighbouring skill that
  owns the adjacent case, since mis-fires between neighbours are the dominant failure; and
  (c) be checked against the descriptions of every skill it borders in §10 before it ships.
- **Maintenance cadence (revision 3).** Skills rot, and over a decade that is a certainty rather
  than a risk. Three obligations: every `SKILL.md` carries a **last-validated marker** (the
  commit its examples and citations were checked against), so staleness is visible rather than
  silent; a skill is re-validated when its owning subsystem changes materially, which
  `code-review` treats as part of the change; and the Ownership Register (§6) is re-checked
  whenever a skill is added, split or retired. Nothing here is scheduled — schedules are
  ignored — it is all triggered by events that already happen.
- **Why.** Found in the completeness pass (§11.1). A 28-skill system with a decade of expected
  growth and multiple maintainers has no defined way to evolve, which is precisely the failure
  the handbook exists to prevent — applied to the handbook. Concretely it must answer: a new
  concern arrives, does it become a skill, a reference file, or a section? Two skills are giving
  contradictory guidance, which one loses? A skill has grown past its trigger, where does it
  split? The cap in §2.6 is enforced here.
  Scheduled as the **exit artifact of Phase 1** rather than its first item, deliberately: it
  should codify the structure that the first five skills actually converged on, not a guess made
  before any exists.
- **Trigger.** Adding, splitting, merging, retiring or restructuring a skill; two skills
  conflicting; asking where a new concern belongs.
- **Not this skill.** The content of any skill.
- **Templates.** `templates/skill.md`.

#### T5-24 · `build-and-ci` — *Phase 4*
- **Responsibility.** The build and its gates: CMake structure and presets, the shader
  compile/embed step, dependency admission policy, `third_party/` rules, **licence
  compatibility and attribution**, CI job design, and `scripts/preflight.sh`.
- **Why.** Build and CI are touched in the same moment by the same person, and CI's design is
  already opinionated in ways worth preserving: cheap checks first so an obvious mistake fails
  in seconds; four configurations including an ASan preset pinned to clang; an explicit
  anti-vacuity assertion that ≥250 tests were collected; benchmarks that run but deliberately do
  not gate, because CI runners are too noisy. The dependency policy also needs stating —
  `third_party/reference/` holds six read-only study clones that are *never* linked, a
  distinction easy for a contributor to get wrong. Licensing was found unassigned in the
  revision-3 review and lands here because it is decided at the moment a dependency is admitted:
  `third_party/doctest/` is vendored under its own licence, the six reference clones are not
  redistributed, and **the repository itself still has no `LICENSE` file** — which blocks any
  outside contribution and should be fixed before the handbook's first external reader.
- **Trigger.** Changing CMake or presets; adding a dependency or third-party tree; changing the
  shader build; modifying CI.
- **Not this skill.** What the tests assert (`testing`); benchmark validity
  (`evidence-and-measurement`).

#### T5-25 · `documentation` — *Phase 4*
- **Responsibility.** Every written artifact that is not an ADR or a stage record: which document
  type a piece of knowledge belongs in, in-code comment philosophy, API documentation, examples
  and tutorials, codebase-navigation material, document lifecycle.
- **Why.** vkML's documentation is its main asset, and its distinguishing habit is that comments
  explain *why*, with the alternative and its cost — `shape.h` spends 20 lines justifying
  row-major order against ggml's convention, and `error.h` explains why throwing beats aborting
  for a Python-driven library. That habit must be made explicit to survive. This skill also owns
  the taxonomy: the project now has architecture docs, ADRs, stage records, theory, audits,
  roadmaps and baselines, and "where does this go?" is recurring. Examples and tutorials are
  folded in — same moment, same quality bar, and they must be built and tested like any code.
- **Trigger.** Writing or updating any document; adding a non-obvious comment; writing an
  example or tutorial; deciding where a finding belongs.
- **Not this skill.** ADRs (`decision-records`); stage records (`optimization`); law entries
  (`performance-theory`).

#### T5-26 · `open-source-study` — *Phase 4*
- **Responsibility.** How to study a mature external project before borrowing from it: read-only
  acquisition with pinned revisions, reading order, extracting invariants vs disagreements, the
  applicability filter, provenance recording, and the prohibition on blind copying.
- **Why.** Executed at a high standard twice already — `ARCHITECTURE.md` Appendix A records every
  borrowed idea with its source, why it exists there and why it applies here; `GAP_ANALYSIS.md`
  Part II asks the sharper question of what six projects *agree* on, on the grounds that a choice
  made independently by four teams is evidence about the problem rather than about the teams,
  and separately records where they *disagree* as a marker of open questions. That methodology
  is reusable and undocumented as a method. It carries the counter-rule too: applicability is
  judged against vkML's constraints, and `GAP_ANALYSIS.md` §7 shows the discipline in action —
  fp16 accumulation is how llama.cpp buys its throughput and is rejected outright because it
  contradicts the numerical policy.
- **Trigger.** Studying an external project; considering adopting a technique; answering "how
  does X do this?"; before proposing an architectural change with external precedent.
- **Not this skill.** Implementing the borrowed idea (domain skills); recording the decision
  (`decision-records`).

#### T5-27 · `milestone-planning` — *Phase 5*
- **Responsibility.** Structuring work: milestone and stage decomposition, exit gates with
  falsifiable criteria, timeboxes, non-goals, scope control.
- **Why.** The project runs on gates — "nothing proceeds until its gate is green" — and they are
  good ones, stated as measurable conditions rather than intentions (M3's is *≥3 TFLOPS fp32 at
  1024³; below 2 TFLOPS, stop and profile rather than proceed*). Two habits are worth
  institutionalising: recording explicit non-goals so they cannot creep in (`ARCHITECTURE.md`
  §9), and ordering work by risk — *cheap and numerically free before expensive and
  load-bearing*, *search before redesign* (`M3_ROADMAP.md`).
- **Trigger.** Planning a milestone or stage; defining a gate; deciding what comes next;
  assessing whether a gate has been met; a scope-creep judgement.
- **Not this skill.** Architectural content of the plan (`architecture-and-layering`); the
  experiment inside a stage (`optimization`).

#### T5-28 · `serialization-and-compat` — *Phase 5*
- **Responsibility.** Anything persisted across process boundaries: model checkpoints, the
  tuned-parameter database, pipeline caches, benchmark baselines — format, versioning,
  forward/backward compatibility, and safety when the input is untrusted.
- **Why.** Nearer than it appears. M3.2 requires a persistent tuned-parameter store and a
  disk-serialised `VkPipelineCache`; `bench/baselines/rx5600m.json` already exists and is
  compared against. Each is a compatibility contract that will outlive the code that wrote it.
  It also carries a safety obligation the brief did not raise: checkpoint loading is the classic
  remote-code-execution surface in ML frameworks (PyTorch's `pickle` default), and vkML should
  fix its format policy before it has users rather than after.
- **Trigger.** Designing or changing a persisted format; loading external data; versioning a
  file; adding a disk cache.
- **Not this skill.** In-memory representation (`tensor-and-graph-semantics`); API versioning
  (`release-and-compat`, deferred); what a baseline *means* (`evidence-and-measurement`).

---

### 5.6 Deferred skills — write when the moment first arrives (3)

Listed for architectural completeness; **not** in the implementation plan. A skill written
before its moment exists is guesswork that will need rewriting.

| ID | Skill | Responsibility | Write when | Whose obligation until then |
|---|---|---|---|---|
| D-1 | `backend-porting` | Adding a backend: the conformance suite it must pass, capability negotiation, `supports_op` fallback, what may and may not be backend-specific | A third backend is contemplated | `architecture-and-layering` keeps the `backend/api` seam clean enough that this is an addition, not a redesign |
| D-2 | `release-and-compat` | Versioning, ABI/API stability windows, changelog, deprecation, packaging | The first tagged release is contemplated | `api-design` records stability intent per surface |
| D-3 | `contribution-workflow` | Onboarding, `CONTRIBUTING.md`, issue/PR templates, triage, governance, maintainer duties | The first external contributor appears | `code-review` covers reviewing external work; `documentation` covers navigation material |

---

## 6. The Canonical Ownership Register

The mechanism that satisfies "one canonical source, no contradictory guidance." Every
cross-cutting concern has exactly **one** owning skill. Every other skill cites it. A skill
found restating a rule it does not own is a defect, and the fix is a citation.

| Concern | Canonical owner | Cited by |
|---|---|---|
| Claim classes, confidence vocabulary, reporting rules | `evidence-and-measurement` | **all** |
| Benchmarking rules, instruments, noise floor, A/B protocol | `evidence-and-measurement` | `optimization`, `performance-theory`, `testing`, `kernel-authoring`, `build-and-ci`, `implementation-checklist` |
| Baseline governance (when a baseline may be updated) | `evidence-and-measurement` | `optimization`, `build-and-ci`, `serialization-and-compat` |
| Tolerance policy and error bounds | `numerics-and-determinism` | `testing`, `op-authoring`, `kernel-authoring`, `optimization` |
| Determinism, bit-identity, golden re-pinning | `numerics-and-determinism` | `refactoring-mechanics`, `optimization`, `kernel-authoring`, `testing` |
| Law grading, scope, prediction scorecard | `performance-theory` | `optimization` |
| Stage/research record format | `optimization` | `performance-theory`, `documentation` |
| Test tiers and the oracle chain | `testing` | `op-authoring`, `kernel-authoring`, `debugging-and-diagnostics` |
| Layer rule and the dependency DAG | `architecture-and-layering` | `header-and-file-design`, all domain skills |
| Extend-vs-refactor decision; architectural smells | `architecture-evolution` | `op-authoring`, `code-review`, `milestone-planning` |
| Behaviour-preservation proof; migration mechanics | `refactoring-mechanics` | `architecture-evolution` |
| Ownership and lifetime model | `cpp-core` | `tensor-and-graph-semantics`, `python-bindings`, `memory-and-allocation` |
| Public/internal boundary and stability intent | `api-design` | `python-bindings`, `header-and-file-design`, `architecture-evolution` |
| The verification checklist | `implementation-checklist` | `code-review`, `git-workflow` |
| Review severity model | `code-review` | `implementation-checklist` |
| ADR format and supersession | `decision-records` | `architecture-evolution`, `architecture-and-layering` |
| Commit boundaries and message form | `git-workflow` | `implementation-checklist` |
| Document taxonomy and comment philosophy | `documentation` | **all** |
| Licence compatibility and attribution | `build-and-ci` | `documentation` |
| Catalog governance, size cap, trigger standard, maintenance cadence | `skill-authoring` | — |

**Changing an interface that already has callers** is the one concern owned by three skills, and
it is recorded explicitly because a three-way split is where contradictions breed:

| Question | Owner |
|---|---|
| *May* this contract change, and what does it promise? | `api-design` |
| *Must* it change — is the current shape the problem? | `architecture-evolution` |
| *How* do callers migrate without breaking behaviour? | `refactoring-mechanics` |

Read in order, they are a sequence rather than a competition. Any skill answering more than its
own question is a defect.

---

## 7. How the skills compose

### 7.1 Precedence

When two loaded skills conflict, the more specific wins, with the constitution above all:

```
CLAUDE.md  >  T3 domain  >  T4 correctness/performance  >  T2 architecture  >  T1 craft
```

Two overrides invert the usual ordering and are stated explicitly because they are the
project's defining constraints:

> **`numerics-and-determinism` outranks every performance skill.** No optimisation, however
> well measured, is accepted if it changes results in a way that skill forbids. `M3_ROADMAP.md`
> already encodes this — split-K is gated on a re-derived error bound *before* implementation.

> **`architecture-evolution` outranks delivery pressure.** If existing abstractions cannot carry
> a feature, the refactor happens first. "Ship it now, clean it later" is not an available
> option, per the no-intentional-debt principle.

An unresolvable conflict is a catalog defect, not a judgement call: it means a register entry
(§6) or a boundary (§10) is wrong and should be fixed. `skill-authoring` owns that repair.

### 7.2 The dependency graph

An arrow means *may cite*. No cycles; a cycle would mean two skills own one decision.

```
                        ┌────────────────────────┐
                        │      CLAUDE.md         │  always loaded — routes, does not restate
                        └───────────┬────────────┘
                                    │
        ┌───────────────────────────┼────────────────────────────┐
        │                           │                            │
┌───────▼──────────────┐  ┌─────────▼──────────────┐  ┌──────────▼───────────┐
│ T5 PROCESS           │  │ T4 CORRECTNESS / PERF  │  │ T3 DOMAIN            │
│                      │  │                        │  │                      │
│ implementation-      │  │  evidence-and-         │  │ tensor-and-graph     │
│   checklist ─────────┼─►│    measurement ◄───────┼──┤   ▲                  │
│ code-review          │  │      ▲   ▲   ▲         │  │   │                  │
│ git-workflow         │  │      │   │   │         │  │ op-authoring         │
│ build-and-ci         │  │ optimization           │  │   │   ▲              │
│                      │  │      │   ▲             │  │   ▼   │              │
│ documentation        │  │      │ performance-    │  │ kernel-authoring     │
│ open-source-study ───┼─►│      │   theory        │  │   │                  │
│ milestone-planning   │  │      ▼                 │  │   ▼                  │
│ serialization        │  │ numerics-and-          │  │ vulkan-backend       │
│                      │  │   determinism ─────────┼─►│ memory-and-alloc     │
│                      │  │      │                 │  │ python-bindings      │
│                      │  │ testing                │  │                      │
│                      │  │ debugging-and-diag     │  │                      │
└──────────────────────┘  └────────────────────────┘  └──────────┬───────────┘
                                                                 │
        ┌──────────────────────────────┐            ┌────────────▼─────────┐
        │ T2 ARCHITECTURE              │◄───────────┤ T1 CRAFT             │
        │ architecture-and-layering    │            │ cpp-core             │
        │ architecture-evolution       │            │ header-and-file      │
        │ decision-records             │            │ api-design           │
        │ refactoring-mechanics        │            │                      │
        └──────────────────────────────┘            └──────────────────────┘

   ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
     META  skill-authoring   ┄┄► may reference every tier above; cited by none.
   └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘      Governs the container, never an engineering question.
```

### 7.3 Named chains

The recurring multi-skill workflows — how the set composes in practice. Each arrow is a handoff
with a gate.

**Chain A — add an operator** *(highest frequency)*
```
architecture-evolution (does the current op/dispatch design carry this, or must it change first?)
  → tensor-and-graph-semantics → op-authoring → cpp-core → numerics-and-determinism
  → testing → [kernel-authoring → vulkan-backend → testing]
  → implementation-checklist → documentation → git-workflow
```
Gate before the bracketed GPU stage: the CPU oracle must pass against PyTorch first, or a
Vulkan mismatch cannot be attributed.

**Chain B — optimise a kernel** *(the current phase's main loop)*
```
performance-theory (what do the laws predict or forbid?)
  → optimization (hypothesis + quantified prediction + falsification criteria, stated first)
  → numerics-and-determinism (fold-neutral? if not, re-derive the bound BEFORE coding — veto)
  → kernel-authoring
  → evidence-and-measurement (resources before timing; instrument validity)
  → testing (goldens)
  → performance-theory (promote, demote, or record the failed prediction)
  → optimization (stage record) → implementation-checklist → git-workflow
```

**Chain C — introduce a feature the architecture does not cleanly support**
```
architecture-evolution (smell identified; extend or refactor?)
  → open-source-study (does prior art solve this, and does it apply here?)
  → architecture-and-layering (target design) → decision-records (ADR with measured alternatives)
  → refactoring-mechanics (migrate, proving behaviour preservation)
  → [feature implementation] → implementation-checklist
```

**Chain D — diagnose a failure**
```
debugging-and-diagnostics
  → evidence-and-measurement (is the instrument wrong?)  ─┐
  → testing (is the test wrong?)                         ─┤ both ruled out BEFORE
                                                          │ assuming a code defect
  → [domain skill] → testing (the regression test is part of the fix) → git-workflow
```

**Chain E — declare work complete** *(runs at the end of every other chain)*
```
implementation-checklist → [any skill it routes to for an unmet item]
  → evidence-and-measurement (is every claim in the summary supported?)
  → git-workflow
```

---

## 8. Folder organisation

```
vkml/
├── CLAUDE.md                       # constitution — always loaded, ~120–150 lines, routes only
├── LICENSE                         # missing today; blocks outside contribution (§11.5 F12)
├── docs/
│   └── ENGINEERING.md              # human entry point — links into .claude/skills/, no content
├── scripts/
│   ├── check_layering.py           # existing
│   └── preflight.sh                # format + tidy + layering + build + tests, one command
└── .claude/
    └── skills/
        ├── README.md               # index: catalog, tiers, chains, ownership register
        │
        ├── cpp-core/               SKILL.md + references/{ownership,values-and-moves,
        │                             templates,errors,constness,ub-and-strictness,
        │                             performance-types}.md
        ├── header-and-file-design/ SKILL.md
        ├── api-design/             SKILL.md
        │
        ├── architecture-and-layering/ SKILL.md
        ├── architecture-evolution/ SKILL.md + references/{smells,extend-or-refactor,
        │                             interface-evolution}.md
        ├── decision-records/       SKILL.md + templates/adr.md
        ├── refactoring-mechanics/  SKILL.md
        │
        ├── tensor-and-graph-semantics/ SKILL.md
        ├── op-authoring/           SKILL.md + references/{backward-rules,
        │                             dispatch-registration,op-checklist}.md
        ├── kernel-authoring/       SKILL.md + references/{tile-geometry,
        │                             determinism-in-kernels,push-constants}.md
        ├── vulkan-backend/         SKILL.md
        ├── memory-and-allocation/  SKILL.md
        ├── python-bindings/        SKILL.md
        │
        ├── evidence-and-measurement/ SKILL.md + references/{claim-classes,benchmarking,
        │                             instruments,baselines}.md
        ├── numerics-and-determinism/ SKILL.md + references/{tolerance-policy,
        │                             fold-order-changes,determinism-guarantee}.md
        ├── testing/                SKILL.md + references/{oracle-chain,test-tiers,vacuity}.md
        ├── optimization/           SKILL.md + references/{pruning-before-timing,rollback}.md
        │                             + templates/stage-record.md
        ├── performance-theory/     SKILL.md
        ├── debugging-and-diagnostics/ SKILL.md
        │
        ├── implementation-checklist/ SKILL.md + references/{architecture,cpp,performance,
        │                             testing,documentation,maintainability,git}.md
        ├── code-review/            SKILL.md
        ├── git-workflow/           SKILL.md
        ├── skill-authoring/        SKILL.md + templates/skill.md
        ├── build-and-ci/           SKILL.md
        ├── documentation/          SKILL.md
        ├── open-source-study/      SKILL.md
        ├── milestone-planning/     SKILL.md
        └── serialization-and-compat/ SKILL.md
```

**Why project-scoped rather than personal (`~/.claude/skills/`):**

1. The handbook arrives with the checkout — a contributor gets the standards without setup.
2. It versions with the code. When the layer rule changes, the commit that changes
   `check_layering.py` also changes the skill — reviewable together, bisectable together.
3. It changes under review. Standards should be amended by the same process as code.
4. Even the nominally generic skills carry vkML-specific examples and citations, so splitting
   by generality would split them arbitrarily and leave the handbook in two places.

**Naming.** Kebab-case, no `vkml-` prefix — the directory already scopes them, and shorter names
read better in a trigger list. Names describe the *moment*, not the topic (§2.1).

---

## 9. Recommended implementation order

Ordered by what is being lost now and by what prevents drift, not by tier.

### Phase 1 — Foundations: prevent loss, prevent drift, define "done" · 5 skills + constitution + meta

| Order | Skill | Rationale |
|---|---|---|
| 1 | `CLAUDE.md` | Nothing else is safe until priorities, the mandatory workflow and the routing table exist; every skill cites it |
| 2 | **`git-workflow`** | **Zero commits.** 23,600 lines with no recovery point. Highest expected loss in the project by a wide margin |
| 3 | `evidence-and-measurement` | Governs every subsequent deliverable, including the other skills. The current phase is optimisation and the ten rules live in a document nobody is required to read |
| 4 | **`cpp-core`** | *Promoted per review.* Its job is preventing drift as the codebase grows toward 100k+ lines, not repairing debt — and consistency is cheapest to establish before the growth. Largest single writing effort in the plan |
| 5 | `implementation-checklist` | The completion gate. Without it the other skills are advisory; with it, "every future implementation follows the standards" becomes checkable |
| 6 | `skill-authoring` | **Phase 1 exit artifact**, written last so it codifies the structure the first five converged on rather than guessing it |

**Exit gate:** the tree is committed with a coherent history; a measurement can be defended by
citing a rule; a completed implementation can be verified against a checklist; the template for
writing the remaining 22 skills exists and is derived from practice.

*Build note for `cpp-core`:* incremental. `SKILL.md` plus `ownership.md` and `errors.md` first
— the two with existing in-tree precedent in ADR-0001 and `error.h` — and the remaining five
references as their questions arise. This keeps Phase 1 from being gated on the largest artifact.

### Phase 2 — The current work loop · 5 skills
`numerics-and-determinism` · `optimization` · `performance-theory` · `testing` ·
`kernel-authoring`

**Exit gate:** one full Chain B iteration executed end to end using only the skills — no
out-of-band instruction needed.

### Phase 3 — Everyday engineering and architectural evolution · 6 skills
`architecture-and-layering` · `architecture-evolution` · `op-authoring` ·
`tensor-and-graph-semantics` · `code-review` · `debugging-and-diagnostics`

**Exit gate:** Chain A and Chain C each executed end to end once.

### Phase 4 — Structure and durability · 7 skills
`decision-records` · `documentation` · `header-and-file-design` · `api-design` ·
`vulkan-backend` · `memory-and-allocation` · `build-and-ci`

### Phase 5 — Scale-out · 5 skills
`refactoring-mechanics` · `open-source-study` · `milestone-planning` · `python-bindings` ·
`serialization-and-compat`

Then the three deferred skills (§5.6) as their moments arrive.

### Sizing

| Phase | Skills | Rough effort | Cumulative coverage of daily work |
|---|---|---|---|
| 1 | 5 + constitution + meta | ~2 days | The rules that prevent loss and drift, plus the definition of done |
| 2 | 5 | ~2 days | ~75 % of current work |
| 3 | 6 | ~2–3 days | ~92 % |
| 4 | 7 | ~2–3 days | ~98 % |
| 5 | 5 | as needed | remainder |

---

## 10. Overlap analysis

Each pair below is a genuine overlap resolved by a boundary rule stated in **both** skills, so a
reader arriving at either is redirected correctly. Pairs marked ⚠ are the ones most likely to
drift and should be re-checked whenever either skill changes.

| Pair | Overlap | Boundary rule |
|---|---|---|
| ⚠ `architecture-and-layering` ↔ `architecture-evolution` | Both concern architectural quality | Layering owns the **static** rules and target state — where things go, what may depend on what. Evolution owns the **change-time decision** — can this design absorb this feature, or must it change first. Test: is a feature request on the table? Yes → evolution |
| ⚠ `architecture-evolution` ↔ `refactoring-mechanics` | Both concern changing existing code | Evolution decides **whether and what**; mechanics executes **how**, and proves behaviour was preserved. The handoff is explicit and one-directional |
| ⚠ `implementation-checklist` ↔ `code-review` | Same underlying checklist | Checklist owns **what "good" means** and the evidence each item requires; review owns **adjudication** — severity, what blocks a merge, how to demand a change, external contributions. Review cites the checklist; it never restates it |
| ⚠ `evidence-and-measurement` ↔ `optimization` | Both handle numbers | Evidence owns **instrument validity and what may be claimed**; optimization owns **the loop that consumes the number**. "Can I trust this?" → evidence. "What do I do about it?" → optimization |
| ⚠ `evidence-and-measurement` ↔ `performance-theory` | Both grade confidence | Evidence owns the **vocabulary and the prediction-before-experiment rule**; theory owns **the law book** — promotion, demotion, scope, the scorecard |
| `evidence-and-measurement` ↔ `testing` | Both compare two things | Evidence compares **timings and resources**; testing compares **values**. Different validity criteria — a byte comparison cannot be perturbed by measuring, a timing can |
| `numerics-and-determinism` ↔ `optimization` | Whether a speedup is permitted | Numerics holds a **veto** and must be consulted before implementation, not after measuring (§7.1) |
| `numerics-and-determinism` ↔ `testing` | Tolerances in tests | Numerics owns **what the tolerance is and why**; testing owns **where the check lives and what else must be checked**. A tolerance value never originates in a test file |
| `cpp-core` ↔ `header-and-file-design` | Include hygiene, forward declarations | cpp-core owns **language constructs**; header-design owns **translation-unit structure**. "Should this be a `unique_ptr`?" → cpp-core. "Should this header include that one?" → header-design |
| `cpp-core` ↔ `api-design` | const-correctness, naming, parameter forms | api-design owns **what is exposed and its contract**; cpp-core owns **how it is written**. If removing it would break a caller outside the module → api-design |
| `cpp-core` ↔ `architecture-and-layering` | Interfaces, composition vs inheritance, SOLID | cpp-core owns **within a component**; architecture owns **between components**. Test: does the answer change which directory the code lives in? |
| `kernel-authoring` ↔ `vulkan-backend` | The dispatch boundary | kernel owns **device-side** (GLSL, tiles, LDS, subgroups); vulkan-backend owns **host-side** (pipelines, barriers, command buffers). The push-constant struct is shared and owned by kernel-authoring, since its layout is a shader constraint |
| `vulkan-backend` ↔ `memory-and-allocation` | Buffers | vulkan-backend owns **`VkDeviceMemory` and API mechanics**; memory owns **policy**: budget, lifetime class, planner offsets, leak accounting |
| `memory-and-allocation` ↔ `cpp-core` | Allocators, ownership | cpp-core owns **host-side C++ lifetime**; memory owns **device memory as a scarce budgeted resource** |
| `op-authoring` ↔ `tensor-and-graph-semantics` | Shape and dtype rules | tensor-semantics owns **the invariants**; op-authoring owns **the recipe that honours them** |
| `op-authoring` ↔ `kernel-authoring` | Implementing an op on GPU | op-authoring owns **the cross-layer sequence and its gates**; kernel owns **the shader**. op-authoring calls kernel-authoring as a step |
| `architecture-and-layering` ↔ `decision-records` | Architectural change | Architecture owns **the rules and target state**; ADR owns **how a change to them is recorded and superseded** |
| `documentation` ↔ `decision-records` ↔ `optimization` | Written artifacts | Documentation owns **prose, comments, API docs, examples, and the taxonomy that routes between all three**; ADR owns **decisions**; optimization owns **stage records** |
| `api-design` ↔ `python-bindings` | The Python surface | api-design owns **the shape of the surface**; bindings owns **the mechanism that projects it**. "Should this method exist?" → api-design. "How does its lifetime cross the boundary?" → bindings |
| `build-and-ci` ↔ `testing` | CI test jobs | testing owns **what is asserted**; build-and-ci owns **when it runs, in what configuration, and what blocks a merge** |
| `skill-authoring` ↔ everything | Structure of the handbook | skill-authoring owns **the container**; every other skill owns **its contents**. It never rules on an engineering question |

---

## 11. Completeness review

A deliberate critical pass over the whole architecture, as the review asked. Findings are
recorded including the ones I decided **not** to act on, so the reasoning is auditable.

### 11.1 Gaps found and closed

| # | Gap | Resolution |
|---|---|---|
| 1 | Revision 1's `refactoring-and-evolution` conflated the **decision** to change a design with the **execution** of the change. Merged, the decision is always made by someone already mid-implementation — the moment when "just extend it" always wins | Split into `architecture-evolution` (T2-5) and `refactoring-mechanics` (T2-7). Directive 3 exposed this; it is a real defect fix, not added scope |
| 2 | Revision 1's `code-review` owned both **what to check** and **how to adjudicate**, so an author self-checking had no trigger and would load a skill written for reviewing others | Split into `implementation-checklist` (T5-20) and `code-review` (T5-21). Directive 4 exposed this |
| 3 | Revision 1 split `measurement` from `evidence-standards`, creating exactly the duplication risk the review warns about — and every documented failure was *both* kinds of error at once | Merged into `evidence-and-measurement` (T4-14), declared the canonical root in the register (§6) |
| 4 | **No skill governed the handbook itself.** 28 skills, a decade of growth, multiple maintainers, and no defined way to add, split, merge or retire one — the exact failure the handbook exists to prevent, applied to the handbook | New `skill-authoring` (M-23), plus the size cap in §2.6 |
| 5 | **"Future backends" is an explicit 10-year goal with no owning skill.** Adding a third backend is a major recurring task and its cost is decided by seam quality today | New deferred `backend-porting` (D-1), with today's obligation assigned to `architecture-and-layering` |
| 6 | **Baseline governance was unassigned.** `bench/baselines/rx5600m.json` is compared against, but nothing said when a baseline may be updated — the mechanism by which a regression is silently blessed | Assigned to `evidence-and-measurement`, `references/baselines.md` |
| 7 | **Host-side profiling was implicitly out of scope** of a GPU-centric measurement skill, though ADR-0001 measured graph-build overhead at ~20 % of step time for a 5,000-node graph and M7/M8 make that regime routine | Explicitly in scope, `references/instruments.md` |
| 8 | The constitution's justification in revision 1 (*some invariants are unsafe behind an on-demand trigger*) argued for duplication. The correct fix is to guarantee the trigger fires | Rewritten as §3: routing table + mandatory workflow + three independent catch passes, with residual risk stated |

### 11.2 Sizing check

- **Too large?** `cpp-core` is the only candidate and is handled by progressive disclosure plus
  incremental construction (Phase 1 build note). `evidence-and-measurement` is newly merged and
  is the second-largest — split trigger recorded: if `references/benchmarking.md` and
  `references/claim-classes.md` develop no cross-references over a year of use, the merge was
  wrong and should be reversed.
- **Too small?** `refactoring-mechanics` after the split, and `code-review` after the split.
  Both remain substantial: mechanics carries the bit-identity proof procedure, staged migration
  and cross-layer moves; review carries the severity model and external-contribution handling.
  Neither is thin enough to fold back without recreating the defect that separated them.
- **Catalog size.** 28 active, cap 32 (§2.6). Four slots of headroom, three of them
  pre-allocated to the deferred skills. Growth beyond that means consolidation, and
  `skill-authoring` enforces it.

### 11.3 Considered and rejected

| Proposal | Verdict | Reason |
|---|---|---|
| Merge `header-and-file-design` into `cpp-core` | **Rejected** | Tempting now that `cpp-core` is Phase 1, but file-structure questions genuinely arise with no language question attached — "this build is slow", "this include creates a cycle" — and folding them in makes the largest Phase 1 artifact materially larger. Watch item: if it stays under ~120 lines after a year, revisit |
| Split concurrency/threading into its own skill | **Rejected for now** | vkML is single-threaded today. Policy lives in `architecture-and-layering`, mechanics in `cpp-core`. **Split trigger recorded:** when a second execution thread ships (DataLoader workers or multi-stream), this becomes its own skill |
| Merge `performance-theory` into `optimization` | **Rejected** (carried from revision 1) | Opposite biases — optimization wants a result, theory wants a falsification. Merging lets the first quietly consume the second, which is the drift `THEORY.md` §10 guards against |
| Create a coding-style skill | **Rejected** | `.clang-format` and `.clang-tidy` make it a build failure. A skill would duplicate and eventually contradict the tooling. The non-mechanisable residue sits in `api-design` (naming semantics) and `documentation` (comment philosophy) |
| Create an incident/postmortem skill | **Rejected** | `MEASUREMENT-AUDIT.md` *is* a postmortem, and the practice — when an instrument fails, audit it and write the rule — is owned by `evidence-and-measurement`. The document type is in `documentation`'s taxonomy. No third home needed |
| Fold `serialization-and-compat` into `api-design` | **Rejected** | A file format is a contract, but format design (versioning headers, magic numbers, forward-compat, safe parsing of untrusted input) shares almost no procedure with signature design |

### 11.4 Scale test

Would a professional organisation run a long-lived project on this structure?

The parts that scale: tier layering gives a place for anything new; the ownership register makes
contradictions detectable rather than latent; deferred skills prevent speculative work;
`skill-authoring` gives the system a defined evolution path; and repository scoping means
standards arrive with the checkout and change under review.

The part that needs watching: **trigger reliability at 28 skills.** Three mitigations are in
place — symptom-phrased routing rows in the constitution, `implementation-checklist` as a
second pass from the opposite direction, and `code-review` as a third. The honest statement is
that this is the architecture's main risk, it is bounded by the size cap, and it is measurable:
if skills routinely fail to fire during Phase 2, the catalog is too fine-grained and should be
consolidated before Phase 3 rather than after.

### 11.5 Second adversarial review (revision 3)

Conducted against the finished revision-2 design, reading it as an outside reviewer with no
investment in it, looking specifically for reasons to change rather than reasons to keep.

**The catalog survived unchanged** — 28 skills, no boundary moved, no merge or split reversed.
That is the most useful result the review could have produced, and it is stated first because
the eleven findings below could otherwise read as a design in trouble. Every finding is about
**usability or longevity**, not structure: the design was right and largely unusable at the
edges.

The method was to run real upcoming tasks through the architecture and watch where it failed.

| # | Finding | Severity | Resolution |
|---|---|---|---|
| F1 | **The architectural gate is too expensive to survive.** Running `logsumexp` — a routine op addition — through Chain A fires 11 skills, the first being `architecture-evolution`, for a feature the architecture obviously supports. A gate that demands full analysis every time gets routed around, and is then absent on the change that needed it | **High** | Step 2 of the mandatory workflow is now explicitly cheap by default (§3.2), and `architecture-evolution` owns a short list of **escalation triggers** that distinguish routine from genuine (§5.2) |
| F2 | **A 35-item uniform checklist will not be run.** Applied to a typo fix it is theatre; applied inconsistently it produces false assurance, which is worse than no checklist | **High** | `implementation-checklist` restructured into mechanical / universal / conditional parts, activated by what the change touched (§5.5) |
| F3 | **The handbook is agent-facing only.** `.claude/skills/` is a path a human contributor will never look in. The stated goal is 100+ contributors, most of whom will read Markdown, not run an agent | **High** | `docs/ENGINEERING.md` — a one-page human index that links into the skills and contains no content of its own, so nothing is duplicated. The skills are already plain Markdown; only discoverability was missing |
| F4 | **What can be mechanised was left as human judgement.** Format, tidy, layering, build and tests are all scriptable, yet appeared as checklist items | **High** | `scripts/preflight.sh`; the checklist covers only what a script cannot. This shortens F2's list and raises reliability at the same time |
| F5 | **Latent contradiction: "no intentional technical debt" vs `refactoring-mechanics`' debt-recording procedure.** Read literally, one forbids what the other provides for | **Medium** | Reconciled in §13: the principle forbids *unrecorded, undecided* debt. A deliberate deferral with a rationale, an owner and a trigger is a decision. Silence is the defect, not deferral |
| F6 | **Trigger reliability was mitigated only after the fact** — three catch passes, no standard for writing a description that fires correctly in the first place | **Medium** | `skill-authoring` now owns a hard trigger-quality standard: symptom phrasing, a mandatory "do not use when" naming the neighbouring skill, and a cross-check against every bordering skill in §10 (§5.5) |
| F7 | **No maintenance cadence.** Over ten years skills rot silently; nothing made staleness visible or tied re-validation to an event that actually occurs | **Medium** | Last-validated markers on every `SKILL.md`, re-validation triggered by material change to the owning subsystem, register re-checked on every catalog change (§5.5). Event-triggered, never scheduled |
| F8 | **Interface evolution is a three-way concern** — `api-design`, `architecture-evolution`, `refactoring-mechanics` — and was only implied in prose, not in the register. Three-way splits are where contradictions breed | **Medium** | Explicit *may / must / how* table added to §6 |
| F9 | **`skill-authoring` was placed in T5**, implying peers could cite it and making the layering circular | **Low** | Moved to a detached **meta tier** (§4) |
| F10 | **`performance-theory` read as document maintenance**, which would justify folding it into `documentation` and would destroy the property that makes it work | **Low** | Responsibility restated as the *epistemics of performance knowledge*; `THEORY.md` is its artifact, not its subject (§5.4) |
| F11 | **No `LICENSE` file, and licence policy was unassigned** — a blocker for any outside contribution and an odd omission in a repository that vendors `doctest` and clones six reference projects | **Medium** | Assigned to `build-and-ci`, since it is decided when a dependency is admitted; the missing artifact is flagged in §8 and §14 |

**Weakest skill in the catalog.** `milestone-planning` — used perhaps ten times a year, and its
content is close to `decision-records`'. It is kept because gate definition is genuinely
procedural and the project already runs on gates, but it is the first merge candidate if it
proves thin in practice, and `skill-authoring` should treat it as such.

**Questions the review asked and answered "no change needed":** can any skill be simplified
(only `implementation-checklist`, done); are any unnecessary (none, though see above); are any
missing (none — F3, F4 and F11 wanted *artifacts*, not skills, which is the right answer and a
good sign the catalog is complete); do the tiers hold (yes, once F9 was fixed); will 100
contributors work (yes, once F3 was fixed — that was the gap).

---

## 12. Changes from revision 1

| # | Change | Source |
|---|---|---|
| 1 | `cpp-core` moved from Phase 3 to **Phase 1** | Directive 1. Rationale accepted and adopted: the skill exists to prevent drift as the codebase grows, not to repair debt |
| 2 | `CLAUDE.md` respecified as **routing-only**, ~120–150 lines, six sections, no duplicated engineering rules; revision 1's justification for inlining invariants replaced with the routing-table argument and an honest residual-risk statement | Directive 2 |
| 3 | New **`architecture-evolution`**; `refactoring-and-evolution` renamed **`refactoring-mechanics`** and narrowed to execution | Directive 3 |
| 4 | New **`implementation-checklist`**; `code-review` narrowed to adjudication | Directive 4 |
| 5 | `measurement` + `evidence-standards` merged into **`evidence-and-measurement`**; new **Canonical Ownership Register** (§6) makes single-source ownership explicit for every cross-cutting concern | Directive 5 |
| 6 | Completeness review (§11): 8 gaps closed, 6 proposals rejected with reasons, sizing and scale assessed; new `skill-authoring`, new deferred `backend-porting`; catalog size cap introduced | Directive 6 |
| 7 | Mandatory workflow (§3.2) added to the constitution, encoding the architectural gate before implementation; two precedence overrides stated (§7.1); chains updated to route through the new gates | Directives 3, 7 |
| — | Count: 26 active + 2 deferred → **28 active + 3 deferred** | |

### Revision 3 — second adversarial review

No skill added, removed, merged, split or re-scoped. Eleven usability and longevity defects
fixed (§11.5), of which four were high severity: the architectural gate made cheap by default
(F1), the checklist made risk-scaled (F2), a human entry point added (F3), and mechanisable
checks moved into a script (F4). Three supporting artifacts introduced — `docs/ENGINEERING.md`,
`scripts/preflight.sh`, `LICENSE` — none of them skills, which is itself evidence the catalog
is complete. `skill-authoring` gained the trigger-quality standard and the maintenance cadence,
and moved to a detached meta tier.

---

## 13. Explicit non-goals

- **No skill restates a measured fact.** Hardware constants, tolerances and laws live in
  `docs/`; skills cite them (§2.3).
- **No skill duplicates mechanical enforcement.** If `.clang-tidy`, `.clang-format`,
  `check_layering.py` or CI already fails the build, the skill says so and points at the tool.
- **No skill restates a rule it does not own.** Enforced by the register (§6).
- **No aspirational skills.** The three deferred skills are written when their moment exists.
- **No skill teaches C++ from scratch.** These are project standards for a competent engineer,
  in the register of LLVM's or Chromium's style documents.
- **No process ceremony without a failure it prevents.** Every rule should be traceable to a
  real failure mode, ideally one this project has already had. A gate that fires at full weight
  on every change is ceremony, and ceremony gets routed around (§11.5 F1).
- **No rule that a script could enforce.** If it can be checked mechanically it belongs in
  `scripts/preflight.sh` or CI, not in a skill and not on a checklist (§11.5 F4).
- **"No intentional technical debt" forbids *unrecorded* debt, not deferral.** A deliberate
  deferral carrying a rationale, an owner and a trigger for revisiting is a decision, and
  `refactoring-mechanics` provides the form for it. What the principle forbids is the silent
  kind: a compromise made and not written down. Stated here because the two rules read as
  contradictory otherwise (§11.5 F5).
- **No second source of truth for the roadmap.** `milestone-planning` owns the *shape* of a
  milestone; `M3_ROADMAP.md` and its successors own the content.

---

## 14. Open questions

Two remain. The commit-ordering question was settled and executed: history begins at the current
state with an honest import (`a2d1541`), no fabricated milestone commits.

1. **Which licence.** The repository has none, which blocks any outside contribution and is the
   only finding in §11.5 that this document cannot resolve on its own. Apache-2.0 is the
   conventional choice for a framework expecting contributors (patent grant, permissive); MIT is
   simpler; the reference projects on disk are MIT (llama.cpp, tinygrad, CLBlast) and
   BSD-3-Clause (CUTLASS). This is a decision, not an architectural question, and it should be
   made before the handbook has an external reader.

2. **`op-authoring` is Phase 3, but no new operator is imminent** — M3.x is kernel optimisation.
   If M4 (training on GPU, which adds ops) is next it should stay; if continued M3 optimisation
   is next, it could swap with a Phase 4 skill. A roadmap decision rather than an architectural
   one.
