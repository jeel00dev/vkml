---
name: cpp_spec
description: The vkML engineering specification. Use for every implementation, modification, optimisation, review, or refactor in this repository — C++, GLSL, CMake, Python bindings, or tests. Defines the development philosophy, the mandatory implementation workflow, project invariants, modern C++20 standards, architecture and layering rules, refactoring policy, performance and measurement discipline, testing requirements, documentation duties, the pre-completion review checklist, and the Git workflow.
---

# vkML Engineering Specification

vkML is a Vulkan-first machine-learning framework in modern C++, built to be correct, fast,
and maintained for a decade. This document is its engineering standard. Follow it on every
change; where it conflicts with habit, this wins.

**Current phase — read this before choosing what to work on.** The M-series research phase is
over. vkML is now building a *complete* library on top of that foundation
(`docs/PHASE2-MANIFESTO.md`). The measured starting position: 74 operators declared, 47
implemented on CPU, **16 on Vulkan**, 20 declared with no implementation at all. Completeness
therefore outranks optimisation until the library can train real models — see §1 and §7.

**Authoritative documents.** This spec states *how to work*. `docs/` states *what is true*:
`PHASE2-MANIFESTO.md` (mission, phase plan, mandatory rules), `ARCHITECTURE.md` (design),
`THEORY.md` (measured performance laws), `MEASUREMENT-AUDIT.md` (instrument validity),
`PERFORMANCE-MODEL.md`, `GAP_ANALYSIS.md`, `M3_ROADMAP.md`, `adr/*.md` (decisions). Cite them;
never copy numbers out of them.

---

## 1. Philosophy

**Priorities, in order.** When two conflict, the earlier wins.

1. **Correctness** — a wrong answer fast is worthless. Determinism and numerical guarantees are
   part of correctness here, not a separate concern.
2. **Completeness** — a library that cannot train a model has no users, however fast its GEMM
   is. Do not optimise an isolated kernel while major functionality is missing.
3. **Architecture** — decide where something belongs before writing it. Structure is expensive
   to change later; code is cheap.
4. **Maintainability** — the next reader has no context. Clever is a liability; obvious is an
   asset.
5. **Performance** — pursued relentlessly, but only with evidence (§7), and only once 1–4 hold.

**Three rules override everything, including performance:**

- **No change may break bit-reproducibility or the numerical contract** (§3.3) without an
  explicit, documented re-derivation. Not for any speedup.
- **No feature is built on an abstraction that cannot carry it.** Fix the abstraction first
  (§6). "Ship now, clean later" is not available.
- **No operator is optimised while operators are missing.** Tuning one kernel further while
  three-quarters of the operator set has no GPU implementation is the specific failure Phase 2
  exists to correct.

**Working principles.**

- Prefer the simplest design that meets the requirement — but simplicity means *few concepts*,
  not *few lines*.
- Make design decisions explicit. An unexplained choice will be undone by someone who assumes it
  was arbitrary.
- Low coupling, high cohesion: a module should be describable in one sentence, and changing it
  should not force edits elsewhere.
- Build abstractions that a second caller will want. Do not build them speculatively for a
  caller who does not exist.
- No intentional technical debt. Deferral is fine when it is *recorded* — rationale, owner,
  and the trigger for revisiting. Silence is the defect, not deferral.

---

## 2. Mandatory implementation workflow

Every change follows this. Steps scale with the change: a typo fix passes most of them in a
line, a new kernel does not. Skipping a step is a decision that must be stated.

| # | Step | What it requires |
|---|---|---|
| 1 | **Understand the request** | Restate the requirement in one sentence. If two readings differ materially, ask before building |
| 2 | **Read the existing code** | Read the files you will touch and their callers. Never patch code you have not read |
| 3 | **Review nearby architecture** | Which layer owns this? What already exists that does something similar? |
| 4 | **Assess the abstractions** | Can what exists carry this feature cleanly? Usually yes — say so in a line and continue. See §6 for the escalation signals |
| 5 | **Improve first if needed** | If step 4 says no, fix the abstraction in its own commit, *then* add the feature |
| 6 | **Design** | For anything expensive to reverse, write the design down first. If it changes an invariant, a public contract, or was chosen against an obvious alternative, it needs an ADR (§9) |
| 7 | **Implement** | To §4 and §5 |
| 8 | **Tests** | Written with the change, not after (§8) |
| 9 | **Validate** | `scripts/check_layering.py`, build clean with `-Werror`, full test suite. For numerics, the oracle chain (§8) |
| 10 | **Benchmark** | Only if performance is the point — and then to §7, which is not optional |
| 11 | **Documentation** | Update what the change invalidated (§9) |
| 12 | **Review** | Run §10 against your own work before calling it done |
| 13 | **Commit** | Propose logical commits to §11 |

**Report honestly.** If tests fail, say so with the output. If a step was skipped, say which and
why. Never state a benchmark you did not run or a test result you did not observe. A confident
wrong number survives review; this project has lost days to that nine times over.

---

## 3. vkML invariants

Hard constraints. Violating one is a defect regardless of how good the code looks.

### 3.1 Layering

Each layer may include only from layers strictly *below* it. Siblings at the same level may not
include each other. Enforced by `scripts/check_layering.py`, which has already caught a real
violation.

```
7  autograd          backward rules, written in terms of forward ops
6  api               public C++ surface — Tensor, ops
5  dispatch, plan    op×device tables, CPU fallback, memory planning
4  backend/cpu       backend/vulkan        (siblings — must not include each other)
3  backend/api       Device / Buffer / Stream interfaces
2  graph             Node, Graph, topological build
1  core              Shape, Storage, DType
0  util              error, log, assert
```

`backend/vulkan` must not know what an `nn.Linear` is. `autograd` must not know Vulkan exists.

### 3.2 Hardware (RX 5600M / RDNA1 / RADV) — see `ARCHITECTURE.md` §1

| Constraint | Consequence |
|---|---|
| `shaderBufferFloat32AtomicAdd = false` | **No global float atomics anywhere.** LDS accumulation within a workgroup; deterministic two-pass across workgroups |
| No cooperative matrix | GEMM is hand-tiled. Do not write code that assumes tensor cores |
| 5.75 GiB VRAM, 256 MiB host-visible, no resizable BAR | Staging uploads are mandatory; avoiding host sync per step is worth real complexity |
| `bufferDeviceAddress = true` | No descriptor sets. Pointers travel in push constants |
| 256-byte push constants | Fixes `kMaxDims = 4`. Raising it taxes every kernel |
| fp16 yes, bf16 no | fp16 is a *capacity* win here (~1.34×), not a throughput win |

### 3.3 The numerical contract

- **Results are bit-reproducible.** Same input, same output, every run. This is vkML's defining
  guarantee and no studied production library offers it.
- **Reductions are pairwise with block 32** (`kPairwiseBlock`, mirrored in
  `tests/python/tolerance.py`). Changing a fold order changes results — it requires a
  re-derived error bound *before* implementation and re-pinned goldens, never a loosened
  tolerance.
- **fp32 accumulation always.** fp16 storage is a separate, permitted question.
- **Tolerance is a property of the operation**, declared once in `tests/python/tolerance.py`
  with a citable source (Vulkan spec ULP allowance, IEEE-754, or Higham's backward bound).
  Never chosen at a call site. Never widened to make a test pass — a failure is a bug until
  proven otherwise, and twice here the *check* was wrong rather than the code.

Kinds: `EXACT` (bit movement, or built only from correctly-rounded IEEE ops) · `ULP`
(transcendentals, where the Vulkan spec permits driver divergence) · `RELATIVE` (composites) ·
`BACKWARD` (`|computed − exact| ≤ γ·Σ|terms|`, for sums and dot products, where the result may
be far smaller than the terms producing it).

### 3.4 Correctness oracle

CPU backend → validated against PyTorch. Vulkan → validated against CPU. Never write the Vulkan
kernel first: a mismatch against an oracle that shares our exact semantics is unambiguously a
kernel bug, which is the whole point of the chain.

### 3.5 Mechanically enforced — do not restate, just satisfy

`.clang-format` (LLVM base, 100 cols, 4 spaces) · `.clang-tidy` · `check_layering.py` · CI.
If a tool can decide it, the tool decides it.

Naming: `lower_case` functions and variables, `CamelCase` types, `trailing_` for private
members, `UPPER_CASE` macros, `kConstantName` for compile-time constants.

---

## 4. Modern C++ standards

**vkML is C++20.** `std::expected` is C++23 and **not available** — use exceptions (§4.7) for
failure and `std::optional` for legitimate absence. Revisit if the project moves to C++23.

### 4.1 Ownership and lifetime

- **Rule of Zero by default.** A class that manages no resource declares no destructor, copy,
  or move. Most types qualify; a type that does not is a design signal worth a second look.
- **Rule of Five when you must.** If you write any of destructor / copy ctor / copy assign /
  move ctor / move assign, write or `= delete` all five. A half-set is a silent bug.
- **RAII for every resource** — memory, `VkDevice*` handles, file descriptors, timers. No
  manual `free`, no `vkDestroy*` outside a destructor, no "remember to release" comments.
- **Ownership is stated in the type.**

  | Intent | Type |
  |---|---|
  | Sole owner | `std::unique_ptr<T>` — the default owning pointer |
  | Genuinely shared, unpredictable lifetime | `std::shared_ptr<T>` |
  | Break a `shared_ptr` cycle; observe without extending | `std::weak_ptr<T>` |
  | Non-owning, must outlive the callee | `T&` or `T*` |
  | Non-owning contiguous range | `std::span<T>` |

- `shared_ptr` is a decision, not a default. vkML uses it for `Node` deliberately — Python holds
  tensors for arbitrary lifetimes and refcounting makes that correct for free (ADR-0001, which
  measured the alternatives). Do not "optimise" it to raw pointers or an arena without reading
  that ADR; the arena lowering is planned for M5 and is additive.
- **Never** `new`/`delete` in application code. `std::make_unique` / `std::make_shared`.
- Raw pointers and references are **non-owning, always**. A function taking `T*` does not free it.

### 4.2 Value semantics and moves

- Prefer values. Copy is fine until measured otherwise; a shared mutable object is a permanent
  cost.
- Return by value. NRVO and move make this free; out-parameters obscure dataflow.
- Take sink parameters **by value and move**: `explicit Foo(std::string s) : s_(std::move(s)) {}`.
  Take read-only parameters by `const&`, or by value for trivially-copyable types ≤ 2 words.
- Mark move operations `noexcept` or containers will silently copy instead of moving.
- A moved-from object is valid but unspecified. Assign to it or destroy it; never read it.
- Do not `std::move` a return value — it blocks copy elision.

### 4.3 const, constexpr, noexcept

- **`const` by default** on locals, parameters, member functions, and references. Non-const is
  the exception that needs a reason.
- `const` member functions must be **thread-safe for concurrent readers**. That is what callers
  assume; `mutable` caches break it unless synchronised.
- `constexpr` for anything computable at compile time — table sizes, tile geometry, dtype sizes.
  Moves work from run time to build time and enables `static_assert`.
- `noexcept` where you can genuinely guarantee it, and always on moves, swaps, and destructors.
  A `noexcept` function that throws calls `std::terminate` — do not use it as decoration.
- `[[nodiscard]]` on anything whose result is the point of calling it (every `Shape` transform
  in this codebase carries it).

### 4.4 Types that carry meaning

- **`enum class` always.** Scoped, non-converting, forward-declarable.
- **`std::optional<T>`** for "may legitimately be absent". Not for errors. `Shape::reshaped`
  returns `nullopt` when the view cannot be expressed as strides, forcing the caller to
  materialise a copy *visibly* rather than hiding it.
- **`std::variant`** for closed sets of alternatives; visit exhaustively so adding a case is a
  compile error, not a runtime surprise.
- **`std::span`** for contiguous ranges — replaces pointer+length pairs and cannot go out of
  sync. Never store one longer than the data it views.
- **Strong types over primitives** where a mix-up is plausible. Two `int64_t`s named `dim` and
  `stride` are one transposition away from a silent bug; a distinct type makes it a build error.
  Use judgement — do not wrap every integer.

### 4.5 Templates and concepts

- Templates for genuine parametric reuse, not to avoid writing a function twice.
- **Constrain every template parameter with a concept.** An unconstrained template fails deep
  inside instantiation with an unreadable error; a constrained one fails at the call site with a
  readable one.
- Keep templates thin: a template wrapper over a non-template implementation keeps compile times
  and binary size sane.
- Prefer `if constexpr` to tag dispatch and SFINAE.
- Compile time is a feature. Instantiating a heavy template in a widely-included header taxes
  every translation unit.

### 4.6 Polymorphism and interfaces

- **Composition over inheritance.** Inherit only for a genuine "is-a" *interface* relationship.
- **Static polymorphism** (templates, CRTP) when the type is known at compile time and the call
  is hot. **Dynamic polymorphism** (virtual) at genuine runtime boundaries — `backend/api`'s
  Device / Buffer / Stream is correct, because the backend is chosen at run time and the
  dispatch cost is amortised over a whole kernel.
- Interfaces are **narrow**. A `Device` that also formats logs serves two masters. Prefer several
  small interfaces to one broad one, so implementers are not forced to stub methods.
- A polymorphic base needs a `virtual` destructor, or a `protected` non-virtual one if deletion
  through the base is never intended.
- Do not add a virtual to "keep options open". Add it when a second implementation exists.

### 4.7 Errors and exception safety

vkML **throws, never aborts** (`include/vkml/util/error.h`). ggml's abort-on-assert is
defensible for a CLI; it is not for a library driven from Python, where it would take down the
user's interpreter. Every exception maps to a natural Python exception in the bindings.

Hierarchy: `Error` (base) · `ShapeError` · `DTypeError` · `DeviceError` · `IndexError` ·
`NotImplementedError` · `InternalError` · `OutOfMemoryError`.

Assertion vocabulary (`util/assert.h`) — three macros, three meanings:

| Macro | Meaning | Active in Release |
|---|---|---|
| `VKML_CHECK(cond, ExcType, ...)` | **User error** — bad arguments to a public API | yes |
| `VKML_ASSERT(cond, ...)` | **Internal invariant** — a failure means vkML is broken | yes, deliberately |
| `VKML_DEBUG_ASSERT(cond, ...)` | Internal invariant on a hot path | no |

Because any invariant check can throw, code between an allocation and its owner must be
exception-safe. Guarantees, in preference order:

1. **No-throw** — destructors, moves, swaps, deallocation. Required, not optional.
2. **Strong** — the operation either succeeds or leaves state unchanged. Aim for this on
   anything mutating shared state. Technique: do the work on a copy, then `swap` (no-throw).
3. **Basic** — invariants hold, no leaks, state unspecified. The minimum acceptable.

RAII gets you most of this automatically, which is the main reason it is non-negotiable.
Error messages name the operation, the actual values, and the expectation:
`"matmul: shape mismatch, lhs (4, 8) vs rhs (16, 2); inner dimensions must agree"`.

### 4.8 Undefined behaviour

UB is not a performance tool. Signed overflow, out-of-bounds indexing, strict-aliasing
violations, misaligned loads, use-after-move, uninitialised reads, and data races are all
defects even when the binary happens to work.

- Use `std::bit_cast` (C++20), never a pointer cast, to reinterpret bits.
- Index with the container's own size type; compare signed to signed.
- Watch alignment at the C++/GLSL boundary. `scalarBlockLayout` makes push-constant structs lay
  out identically on both sides *only if* the struct is written to respect it — `static_assert`
  the size and offsets.
- Build and test with the `asan` preset. A clean ASan/UBSan run is evidence; the absence of a
  crash is not.

### 4.9 Thread safety

vkML is currently single-threaded above the driver. That is a **stated policy, not an
accident** — do not introduce threads, `std::atomic`, or locks without a design decision
recorded in an ADR.

When concurrency does arrive, the rules already hold: `const` methods are safe for concurrent
readers; `Node` is immutable after construction except for realisation fields (ADR-0001), which
is precisely what would make sharing safe; document every type as thread-safe, thread-compatible
(safe for distinct instances), or thread-hostile.

### 4.10 Performance-aware design

Design for performance; do not micro-optimise without measurement (§7).

- **Data layout beats instruction selection.** Prefer contiguous arrays over pointer chases;
  ADR-0001 measured a 64× traversal gap between a `shared_ptr` graph and a flat arena, driven
  almost entirely by cache locality.
- Keep hot structs small and hot fields adjacent. Cold data belongs in a side table.
- Allocation is a design concern, not a micro-optimisation. Reserve capacity; reuse buffers on
  repeated paths; the steady-state training step should allocate nothing.
- Pass large read-only data by `const&` or `span`. Return by value only when the type is cheap
  to move.
- Prefer the better algorithm before the better constant factor.
- SIMD: let the compiler vectorise by writing clean, aliasing-free loops over contiguous data.
  Reach for intrinsics only when a profile and a disassembly justify it.

### 4.11 Headers and includes

- `include/vkml/` is the **public** surface. `src/**/*.h` is internal. Nothing in `include/`
  exposes an internal type — ADR-0001's guardrail: `Node` must not appear in the public Tensor
  API, so the representation can change without breaking the ABI or the bindings.
- **Include what you use**, directly. Do not lean on transitive includes.
- **Forward-declare** when only a reference, pointer, or return type is needed. Include when you
  need the size, a member, or a base.
- Every header is self-contained and compiles alone. `#pragma once` at the top.
- Headers include the minimum. A heavy include in a widely-used header is a build-time tax on
  everyone; PImpl is the escape hatch when an implementation detail would otherwise leak.
- Definitions in `.cpp` unless the function is genuinely tiny, a template, or `constexpr` and
  needed at compile time.
- **Circular dependencies are an architecture failure, not an include problem.** Fix the layering
  (§3.1); do not paper over it with forward declarations.

### 4.12 Naming and comments — the maturity standard

This is what separates a codebase a stranger can maintain from one only its author can. Hold it
to the standard of the projects in §12: LLVM, Chromium, the Linux kernel. Neither of these is
cosmetic — a wrong name and a stale comment both actively mislead, which is worse than silence.

**Names state what a thing is, in the domain's own vocabulary.**

- Use the word the field uses. `logits`, `stride`, `subgroup`, `workgroup`, `epilogue`,
  `spilled_vgprs`. Never invent a synonym for an established term, and never use two words for
  one concept — if it is a `workgroup` in one file it is not a `block` in the next.
- **Length scales with scope.** `i` inside a three-line loop is correct and `index_of_current_row`
  there is noise. A member, a parameter, or anything living more than a screen gets a full name.
- **Abbreviate only what the domain already abbreviates.** `gemm`, `lds`, `spv`, `vgpr`, `esz`
  next to an `element size` comment — fine, these are how practitioners write. `mgr`, `hdlr`,
  `tmp2`, `res`, `val`, `do_stuff` — never.
- **Booleans read as assertions**, so the call site reads as English: `is_contiguous()`,
  `has_broadcast_stride()`, `supports(node)`, `binary_srcs_are_f32(node)`.
- **No type in the name.** Not `float_value`, `p_node`, `vec_dims`. The type is already there.
- **A name that needs a comment to be understood is the wrong name.** Rename first; comment only
  what a better name cannot carry.
- **Banned outright**: `data2`, `temp`, `foo`, `helper`, `utils`, `misc`, `manager`, `process()`,
  `handle()`, and any name whose only meaning is "the other one".

**Comments explain *why*. The code already says *what*.**

The test is: **would a competent reader be surprised?** If yes, explain, and explain the
*reason* — not the mechanics. If no, write nothing. A comment restating its line is worse than
no comment, because it drifts out of date and then lies.

What earns a comment:

- **A non-obvious constant, with its derivation.** Not `// 32 elements`, but why 32 — where the
  number came from and what breaks if it changes.
- **A rejected alternative**, and the cost that rejected it. This is the single most valuable
  kind here: it stops the next person redoing the analysis, or "fixing" the code back to the
  version that was wrong. `error.h` explains why vkml throws where ggml aborts;
  `shape.h` spends twenty lines justifying row-major order against ggml's convention.
- **A deliberate divergence** from the obvious implementation, from PyTorch, or from a
  specification — always with the reason attached.
- **A hardware or driver constraint** that makes otherwise-odd code necessary.
- **An invariant a caller must uphold** that the type system cannot express.

What must not appear:

- Restating the line: `// increment i`, `// loop over elements`, `// return the result`.
- **Commented-out code.** Git has it. Delete it.
- Decorative banners with no content, and `// -----` separators around a single function.
- A `TODO` with no owner and no trigger. Deferral is allowed but must be *recorded* (§1) — say
  what is deferred, why, and what would make it worth doing.
- Anything that will be false after the next edit. If a comment must track a value stated
  elsewhere, cite the source rather than copying it.

**The standing test for both**, applied before every commit: could a contributor who has never
spoken to you read this file and make a correct change? If a name or a comment would send them
the wrong way, it is a defect, not a style preference.

---

## 5. Architecture

- **Dependencies point one way** — down the layer stack (§3.1), always. If a lower layer needs
  something from above, invert it with an interface the lower layer owns.
- **Module boundaries are contracts.** A module states what it guarantees and what it requires.
  Everything else is private and may change without notice.
- **Separation of concerns.** One module, one reason to change. A file that changes for two
  unrelated reasons is two files.
- **Encapsulate representation.** Expose behaviour, not fields. Public data members are a
  permanent commitment.
- **Extension points are deliberate.** `backend/api` and `supports_op()` exist so a new backend
  is an addition rather than a redesign, and so unimplemented ops fall back to CPU
  transparently. Keep that seam clean; do not let backend-specific concepts leak upward.
- **Do not build plugin machinery you do not need.** vkML deliberately omits ggml's `reg` layer
  and PyTorch's dispatcher: for two backends and ~64 ops, a flat table is the right size. Add
  indirection when a second case exists, not before.
- **New code goes in the layer that owns the responsibility**, even when that is less convenient
  than the layer you happen to be editing.

---

## 6. Refactoring

**Before adding a feature, decide whether what exists can carry it.** Usually it can — say so
in a line and move on. Escalate to a real analysis only when you see one of these signals:

- the feature needs a special case inside an existing abstraction to be accepted;
- a parameter exists only to select behaviour for one caller;
- the change forces an edit in a layer that should not have known about it;
- the same conditional appears in a third place.

**When a signal fires, fix the abstraction first, in its own commit.** Building on a known-bad
abstraction is how a codebase becomes unmaintainable, and it is never cheaper later.

**When *not* to refactor** — this matters as much:

- The code is ugly but stable, well-tested, single-caller, and no feature is pending.
- You are mid-feature and the refactor is unrelated. Note it, finish, do it separately.
- You cannot state the improvement in one sentence.
- Behaviour would change. That is a redesign, and it needs §2 step 6.

**Refactoring preserves behaviour, and here that is checkable rather than a matter of
judgement:** a refactor that changes no fold order must leave goldens **byte-identical**. If
they move, you did not refactor — you changed the result, and §3.3 applies. Refactor in small
steps with the suite green between each.

---

## 7. Performance engineering

**No optimisation is accepted without evidence.** Not because it looks faster, not because it
is theoretically better, not because another project does it.

### Gate zero — is this the right work at all?

Before profiling anything, answer: **is the functionality this optimisation serves actually
complete?** In the current phase the answer is usually no, and the honest response is to
implement what is missing instead. An operator with no GPU kernel is infinitely slower than a
suboptimal one, and no amount of tuning elsewhere closes that gap.

Optimise now only when: the path is on a real workload's critical path, the functionality
around it is complete and tested, and a profile — not intuition — identifies it. Otherwise
record the opportunity and move on.

### The loop

1. **Profile.** Identify the actual bottleneck. Intuition about which line is hot is wrong more
   often than it is right.
2. **Hypothesise.** State what you expect to change, by how much, and *what result would prove
   you wrong* — before measuring. A prediction made afterwards is not a prediction.
3. **Check the constraints.** Does it alter fold order (§3.3)? Do the laws in `THEORY.md`
   already forbid it? Cheapest possible refutation first.
4. **Gate on resources before timing.** Compile the candidate and read `PipelineStats` — VGPRs,
   spills, scratch bytes, occupancy. Reject on non-zero scratch or spilled registers *without
   ever benchmarking it*. Stage 8's regression was fully visible in the statistics before a
   single benchmark ran, and no comparable production library can do this.
5. **Measure**, to the rules below.
6. **Verify correctness.** Full suite plus goldens.
7. **Accept or roll back.** A hypothesis that failed is a result — record it in the stage
   document rather than deleting it. Failed predictions are how `THEORY.md`'s laws were found.

### Measurement rules — from `MEASUREMENT-AUDIT.md`, learned the hard way

Nine times an apparent implementation failure turned out to be a measurement error, and once the
profiler built to prevent that class of error caused it.

1. Effects below ~2 % need **GPU timestamps** and repeated independent trials. Timestamps are
   ~20× more reproducible than wall clock here.
2. Wall clock is admissible only when the measured operation **dominates the window** — check
   `GPU/wall > 0.5` first, whatever the effect size.
3. Report the **minimum**, never the mean. The tail measures the machine, not the kernel.
4. **Never sum per-dispatch timestamps** across independent dispatches — use the `submit` window.
   Concurrent dispatches each report to a global drain point and the sum counts the same time
   repeatedly.
5. Never compare a profiled time against an unprofiled one.
6. **Never benchmark with validation layers enabled.** They change what they measure,
   substantially.
7. Warm pipelines before timing; compilation is setup, not measurement.
8. An A/B is valid only if the A arm is the **frozen, unmodified** baseline.
9. Prefer acceptance criteria that **compare bytes** — they cannot be perturbed by measuring.
10. Check every correctness gate for **vacuity** before trusting a pass.
11. Two independent calculations agreeing is not confirmation — it is a warning the experiment
    cannot distinguish them.

Benchmark with the `relwithdebinfo` preset. Baselines live in `bench/baselines/`; updating one
requires the same evidence as any other performance claim.

### What actually matters on this GPU

Small-batch training is **bandwidth-bound**, not FLOP-bound (~185 GB/s at N=1 against a 288 GB/s
peak). Fusion and avoiding host round-trips beat GEMM tuning for those shapes. Occupancy is
governed by independent barrier domains, not raw wave count. Do not assume; read `THEORY.md`
before reasoning about registers, occupancy, or spilling — the laws there are measured, scoped,
and several are counter-intuitive.

---

## 8. Testing

Tests ship with the change. A feature without tests is unfinished.

| Tier | What it covers |
|---|---|
| **Unit** | Every op: every dtype, contiguous + strided + broadcast, edge shapes (empty, size-1, non-power-of-2, rank 0 and 4) |
| **Oracle** | CPU vs PyTorch, then Vulkan vs CPU (§3.4). Both, in that order |
| **Autograd** | Analytical gradient vs central finite differences *and* vs `torch.autograd.grad`, for every input and parameter |
| **Integration** | Layers, optimisers over 100 steps (compare trajectories, not just endpoints), full model parity |
| **Regression** | Golden hashes. Exact, because results are deterministic. Every fixed bug gets one |
| **Property** | Randomised shapes, strides and dtypes for shape algebra and broadcasting, where hand-written cases miss combinations |
| **Resource** | Every allocation freed at exit; peak VRAM within 10 % of prediction; no device loss under stress |

- Run correctness tests under `VKML_EAGER=1` so a failure points at the offending op rather than
  at the realise boundary.
- Run Vulkan tests with validation layers on. Any validation error fails the build.
- **A test that cannot fail is worse than no test** — it manufactures confidence. Verify a new
  test fails against deliberately broken code before trusting its pass.
- Tolerances come from `tests/python/tolerance.py` and nowhere else (§3.3).
- Test behaviour through the public API, not private internals — otherwise every refactor breaks
  the suite and the suite stops meaning anything.

Commands: `ctest --preset debug` · `python -m pytest tests/python -q` · `python scripts/check_layering.py`

---

## 9. Documentation

Documentation is a deliverable, not a byproduct. vkML's is its main asset.

- **Comments explain *why*, and name the alternative and its cost.** The code already says what.
  `shape.h` spends twenty lines justifying row-major order against ggml's convention; `error.h`
  explains why throwing beats aborting. That habit is the standard — match it.
- **Public API**: every public type and function gets a doc comment covering purpose,
  parameters, return value, what it throws, and any lifetime or ownership requirement.
- **Non-obvious code** gets a comment. Obvious code gets none — a comment restating the line is
  noise that will drift out of date.
- **Design rationale** goes in `docs/`, not in a commit message where nobody will find it.
  A decision that is expensive to reverse, changes a public contract or an invariant, or was
  taken against an obvious alternative → an **ADR** in `docs/adr/`, following the existing
  format: *Context → Measurements → Options considered, each with a verdict → Decision →
  Guardrails adopted now → Rejected alternatives, recorded so they are not rediscovered.*
- **Performance experiments** get a stage record: hypothesis, quantified prediction, and
  falsification criteria stated *before* the results.
- **Update what your change invalidated.** Stale documentation is worse than none, because it is
  trusted. If behaviour changed, the doc changes in the same commit.
- Examples must build and run in CI, or they will rot.

---

## 10. Review checklist

Run this against your own work before declaring anything complete. Scale it to the change:
the mechanical items always, the rest as applicable.

**Mechanical** — no judgement required, so just run them:
`scripts/check_layering.py` · clang-format clean · builds with `-Werror` · `ctest` green ·
`pytest` green · ASan preset clean for memory-touching changes.

**Always:**
- [ ] Every claim I am about to make is supported by something I actually ran.
- [ ] The change is one logical unit.
- [ ] Anything left unfinished is stated explicitly, not implied.
- [ ] The surrounding code is no worse than I found it.

**Architecture:** correct layer · one responsibility · right abstraction level · no new coupling ·
no circular dependency · extension points still clean.

**Correctness:** edge cases (empty, rank 0, size-1, broadcast, non-contiguous, aliased) ·
integer overflow · error paths tested, not just the happy path.

**C++:** ownership stated in types · Rule of Zero, or all five · RAII for every resource · no
raw `new`/`delete` · moves `noexcept` · `const` correct · `[[nodiscard]]` where the result is the
point · no UB · includes minimal and direct · headers self-contained.

**Exception safety:** no-throw where required · strong guarantee where state is mutated · no
leak on any throw path.

**Numerics:** fold order unchanged, or bound re-derived and goldens re-pinned · tolerances from
the policy · determinism preserved.

**Performance:** no unnecessary allocation on hot paths · layout cache-friendly · any
optimisation backed by measurement to §7 · no regression.

**Testing:** new tests can actually fail · regression test for every fixed bug · oracle chain
followed.

**Documentation:** public API documented · non-obvious decisions explained · invalidated docs
updated · ADR if the decision warrants one.

**Maintainability (§4.12):** every name states what the thing is in domain vocabulary · no
`tmp`/`data2`/`helper`/`process()` · booleans read as assertions · every comment explains *why*
and none restates its line · non-obvious constants carry their derivation · rejected
alternatives recorded · no commented-out code · no ownerless TODO · no duplication that wants an
abstraction · no speculative abstraction without a second caller.

The question that decides it: **could a contributor who has never spoken to you read this and
make a correct change?**

---

## 11. Git workflow

- **One logical change per commit.** A refactor and a feature are two commits, always. A commit
  that does two things cannot be reverted, reviewed, or bisected cleanly.
- **Every commit builds and passes tests.** Broken intermediate commits destroy bisection, which
  is the main reason history is worth keeping.
- **Commit messages explain *why*.** Subject in the imperative, ≤ 72 characters, no trailing
  period. Blank line. Body covering what changed and, more importantly, the reasoning — the
  alternative considered, the measurement that justified it, the constraint that forced it.
  Reference documents and ADRs by path. Every commit must be understandable without opening the
  diff.
  Good: `Add tensor broadcasting support` · `Implement Adam optimizer` ·
  `Refactor Vulkan descriptor cache` · `Fix gradient accumulation bug`.
  Never: `fix` · `update` · `changes` · `working` · `final` · `test` · `misc`.
- **Never commit broken code** — failing to compile, breaking tests, a known regression, or
  incomplete functionality that is not clearly isolated. Build, test and format first. Every
  commit should raise confidence in the codebase, not lower it.
- **Keep history linear and readable.** Squash experimental or noisy commits before merging. A
  reader years later should see how the architecture evolved and when features, optimisations
  and refactors landed.
- **One objective per pull request.** Do not combine unrelated features, refactors, formatting
  and optimisations.
- **The history is engineering documentation.** A contributor should be able to read it and
  understand how vkML evolved: decisions, milestones, refactors, optimisation phases.
- **No artificial history.** Never fabricate commits for states that never existed. Truthfulness
  beats a tidy log.
- **No AI attribution.** No `Co-Authored-By` for tools, no "generated with" trailers. The commit
  records the change, not what typed it.
- **No unnecessary rewriting.** Amend or rebase only unpushed work, and only to fix a genuine
  mistake.
- Never commit build output, `.venv`, generated SPIR-V, or the reference clones under
  `third_party/reference/` — `.gitignore` covers these; keep it that way.
- Propose commits as a list with messages; let the author decide when to run them.

---

## 12. Learning from open source

Before proposing a significant architectural change or optimisation, find out how mature
projects solved the same problem — **and why**.

Relevant prior art: **ggml / llama.cpp** (the only production Vulkan GEMM; same GPU class) ·
**tinygrad** (lazy execution, search-based autotuning) · **CLBlast** (tuned parameters for
gfx1010 — vkML's exact chip, and the only project that has them) · **CUTLASS** (the reference
vocabulary for GEMM structure) · **rocBLAS / Tensile** · **PyTorch** (API shape) · **oneDNN**,
**XLA**, **TVM** (compiler and fusion design) · **Eigen** (expression templates) · **LLVM**,
**Chromium**, **Qt** (large-scale C++ engineering practice).

Six are already cloned read-only, at pinned revisions, under `third_party/reference/`. They are
study material — never linked, never vendored, never modified.

**How to study:**

1. Read for **intent**, not implementation. Ask why the design is shaped that way and what
   constraint produced it.
2. **Convergence is evidence about the problem.** A choice made independently by four teams with
   different languages and hardware is telling you something about the problem, not about the
   teams. Where they *disagree*, the question is genuinely open — treat it as such.
3. **Apply the filter.** Does the constraint that motivated it hold for vkML? Their target GPU,
   scale, workload, and guarantees are usually different. llama.cpp accumulates in fp16 by
   default and clamps to avoid overflow — a deliberate speed-for-accuracy trade that is
   **incompatible** with §3.3 and is therefore rejected outright, however fast it is.
4. **Never copy code.** Understand the idea, then implement it for vkML's constraints. Copying
   also imports licence obligations.
5. **Record provenance.** A borrowed idea is documented with its source, why it exists there, and
   why it applies here — see `ARCHITECTURE.md` Appendix A for the format.

vkML already leads the field in one respect — driver-level pipeline statistics as a pre-benchmark
filter (§7). Do not assume the reference projects are ahead on everything.

---

## Quick reference

```bash
cmake --preset debug && cmake --build build/debug -j$(nproc)   # develop
ctest --preset debug --output-on-failure                        # C++ tests
python -m pytest tests/python -q                                # parity suite
python scripts/check_layering.py                                # layering
cmake --preset asan  && cmake --build build/asan -j$(nproc)     # ASan/UBSan
cmake --preset relwithdebinfo                                   # benchmarking ONLY
VKML_EAGER=1 python -m pytest tests/python -q                   # per-op realisation
```

**If you remember nothing else:** read before writing · fix the abstraction before building on
it · never break determinism · measure before claiming · test what you changed · name things so
the next reader needs no explanation · comment the *why*, never the *what* · say what you
actually did.
