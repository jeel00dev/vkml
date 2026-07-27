---
name: cpp_spec
description: The vkML engineering specification. Use for every implementation, modification, optimisation, review, or refactor in this repository — C++, GLSL, CMake, Python bindings, or tests. Defines how to behave, the development philosophy, the mandatory implementation workflow, project invariants, modern C++20 standards, architecture, backend and dispatcher design, API evolution, refactoring policy, performance and measurement discipline, testing requirements, documentation duties, the pre-completion review checklist, and the Git workflow.
---

# vkML Engineering Specification

vkML is a Vulkan-first machine-learning framework in modern C++, built to be correct, fast, and
maintained for a decade. This document is its engineering standard. Follow it on every change;
where it conflicts with habit, this wins.

**This document states how to work. It deliberately states no project state.**

Operator counts, device capabilities, measured throughput, the current milestone, tuned
constants and roadmap position all change, and a rule that embeds them expires. They live in
`docs/`, which is authoritative for *what is true*:

| Document | Authoritative for |
|---|---|
| `ARCHITECTURE.md` | design, layering, measured device capabilities, op inventory |
| `PHASE2-MANIFESTO.md` | mission, phase plan, current priorities |
| `THEORY.md` | measured performance laws, with confidence and scope |
| `MEASUREMENT-AUDIT.md` | instrument validity and the measurement rules' derivations |
| `PERFORMANCE-MODEL.md`, `GAP_ANALYSIS.md` | performance analysis |
| `adr/*.md` | decisions, their alternatives, and their guardrails |

Cite them. Never copy a number out of them into code, into a comment, or into this file.

---

## 0. How to behave

This section is for the agent reading the file. It is first because it governs everything after.

**Never assume; read.** Before changing a file, read it and its callers. Before using an API,
read its declaration. Before repeating a claim you found in a comment, verify it against the
code — a comment can describe an intended design that was never built.

**Never invent.** Not an API, not a function signature, not a flag, not a benchmark, not a test
result. If you did not run it, you did not measure it. If you cannot find it, it does not exist.
Say so instead of producing something plausible.

**State assumptions out loud.** When a decision depends on something unverified, say which
assumption you took and what would change if it were wrong.

**Report faithfully.** If tests fail, say so with the output. If you skipped a step, say which
and why. "Done" means verified, not written.

**Do not silently change design philosophy.** A recorded decision — an ADR, a documented
divergence, a deliberate non-goal — is binding until explicitly revisited. If you believe one is
wrong, say so and make the case; do not quietly implement the alternative.

**When proposing architecture, give trade-offs, not just a recommendation.** What it costs, what
it forecloses, what would make it the wrong call.

**Ask when genuinely blocked.** Two readings that lead to materially different work, or a
decision that is the author's to make, are worth one question. Routine judgement calls are not —
make them, state them, continue.

**Prefer the smallest change that is complete.** Do not widen scope silently, and do not narrow
it either: if part of a task is blocked, finish the rest and say plainly what you left.

---

## 1. Principles

Stated once, here. Later sections refer to them by number rather than restating them.

**P1 — Correctness first.** A wrong answer fast is worthless. Determinism and the numerical
contract are part of correctness, not a separate concern.

**P2 — Completeness before optimisation.** A library that cannot do the job has no users,
however fast its inner loop is. Do not tune one component while others are missing.

**P3 — Architecture before code.** Decide where something belongs before writing it. Structure
is expensive to change; code is cheap.

**P4 — Maintainability over cleverness.** The next reader has no context. Obvious beats clever,
always.

**P5 — Evidence before claims.** No optimisation, no performance statement, and no "it works"
without something you actually ran.

**P6 — Generic before specialised.** One correct implementation that works everywhere comes
first. Specialisation is an optimisation, and P5 governs it.

**P7 — No unrecorded debt.** Deferral is fine; silence is not. A deliberate compromise carries a
rationale, an owner, and the trigger for revisiting it.

**Ordering.** When two conflict, the lower number wins. Three rules override even that:

- **The numerical contract (§3.3) is never traded for speed**, by any margin, without a
  documented re-derivation.
- **No feature is built on an abstraction that cannot carry it.** Fix the abstraction first
  (§6). "Ship now, clean later" is not available.
- **A recorded decision is binding** until explicitly revisited (§0).

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
| 9 | **Validate** | Layering, a clean warning-free build, the full suite. For numerics, the oracle chain (§8) |
| 10 | **Benchmark** | Only if performance is the point — and then to §7, which is not optional |
| 11 | **Documentation** | Update what the change invalidated (§9) |
| 12 | **Review** | Run §10 against your own work before calling it done |
| 13 | **Commit** | Propose logical commits to §11 |

---

## 3. Invariants

Hard constraints. Violating one is a defect regardless of how good the code looks.

### 3.1 Layering

The codebase is a stack of layers. **A layer may depend only on layers strictly below it, and
never on a sibling.** Higher-level concepts must not leak downward: a backend must not know what
a neural-network layer is, and the autograd rules must not know which backend exists.

This is a policy, enforced automatically. A dependency check runs in CI; when it rejects a
change, the fix is the design, not the check. Consult `ARCHITECTURE.md` for the current layer
list — it grows.

### 3.2 Target hardware

**Query capabilities at runtime; never assume them.** The implementation must respect the
capabilities of the active device, not of the machine it was written on.

- Never assume support for cooperative matrix / tensor cores, float atomics, a particular
  subgroup size, a specific shared-memory size, `bufferDeviceAddress`, or any extension.
- Every capability the code depends on is queried, and its absence produces a clear failure or a
  supported alternative — never undefined behaviour.
- A device query that returns "unknown" (a vendor-specific property on another vendor's
  hardware) is a reason to **decline** the decision it would have informed, not to guess.
- Push-constant budget, workgroup limits and shared-memory size are device properties. Fit
  within what is reported; if metadata cannot fit, move it to a buffer rather than assuming
  headroom.

Measured capabilities of the current development device are recorded in `ARCHITECTURE.md`. They
are context for understanding decisions, not values to hardcode.

### 3.3 The numerical contract

- **Results are bit-reproducible.** Same input, same output, every run, on a given device and
  build.
- **Reduction order is fixed and deterministic.** Atomic accumulation is forbidden — not because
  hardware may lack it, but because its ordering varies run to run and float addition is not
  associative. This also makes the code portable to devices without it.
- **Accumulate in at least fp32**, whatever the storage precision.
- **A tolerance is a property of the operation**, declared once in the central policy with a
  citable source, and derived *before* the implementation. Never chosen at a call site, and
  never widened to make a test pass — a failure is a bug until proven otherwise.
- **Changing a fold order requires a re-derived error bound and re-pinned goldens**, decided
  before any code is written.

Prefer an acceptance criterion that compares bytes. It cannot be perturbed by the act of
measuring, and it detects an ordering change that a tolerance would hide.

### 3.4 The correctness oracle

Every operation is validated through a chain, in this order:

```
reference backend  vs  established framework   →  "is the maths right?"
optimised backend  vs  reference backend       →  "is this implementation right?"
```

**Never write the optimised path first.** Each comparison must have exactly one candidate cause;
that is the whole value. A mismatch against an oracle sharing our own semantics is unambiguously
an implementation bug, whereas a mismatch against an external framework could be either.

The reference backend is written to be *obviously correct*, not fast. It is a measuring
instrument that also ships.

### 3.5 Mechanical enforcement

Formatting, static analysis and layering are decided by tooling, not by review. **If a tool can
decide it, the tool decides it** — configure the tool, satisfy it, and do not restate its rules
here or argue them in review. Naming conventions are part of that configuration.

---

## 4. Modern C++

This project is C++20. Use the standard it targets; do not reach for a later one. Check the
build configuration before assuming a feature is available — `std::expected` is C++23 and is not
available here.

This section states how *this project* writes C++. It is not a summary of the Core Guidelines or
of Effective Modern C++; those are references, and this is a house style.

### 4.1 Ownership and lifetime

- **Rule of Zero by default.** A class managing no resource declares no destructor, copy or
  move. A type that cannot follow it is a design signal worth a second look.
- **Rule of Five when you must.** Writing any of the five means writing or `= delete`-ing all
  five. A half-set is a silent bug.
- **RAII for every resource** — memory, device handles, file descriptors, timers. No manual
  release, no "remember to free" comments.
- **Ownership is stated in the type.**

  | Intent | Type |
  |---|---|
  | Sole owner | `std::unique_ptr<T>` — the default owning pointer |
  | Genuinely shared, unpredictable lifetime | `std::shared_ptr<T>` |
  | Break a cycle; observe without extending | `std::weak_ptr<T>` |
  | Non-owning, must outlive the callee | `T&` or `T*` |
  | Non-owning contiguous range | `std::span<T>` |

- `shared_ptr` is a decision, not a default. Where this codebase uses it, an ADR records why;
  read it before "optimising" it away.
- **Never** `new`/`delete` in application code. Use `std::make_unique` / `std::make_shared`.
- Raw pointers and references are **non-owning, always**.

### 4.2 Value semantics and moves

- Prefer values. A copy is fine until measured otherwise (P5); shared mutable state is a
  permanent cost.
- Return by value. Out-parameters obscure dataflow and defeat elision.
- Sink parameters **by value and moved**; read-only parameters by `const&`, or by value for
  trivially-copyable types of a word or two.
- Mark moves, swaps and destructors `noexcept` — containers silently copy instead of moving
  otherwise.
- A moved-from object is valid but unspecified: assign to it or destroy it, never read it.
- Do not `std::move` a return value; it blocks copy elision.

### 4.3 const, constexpr, consteval, noexcept

- **`const` by default** on locals, parameters, member functions and references. Non-const needs
  a reason.
- `const` member functions must be **thread-safe for concurrent readers** — that is what callers
  assume, and a `mutable` cache breaks it unless synchronised.
- **`constexpr` wherever computation can move to build time.** It also enables `static_assert`.
- **`consteval`** when a function must *never* be called at run time — a compile-time-only
  factory or check. Use it to make that a compile error rather than a convention.
- **`static_assert` for invariants** the type system can check: struct sizes and offsets shared
  with a shader, table sizes matching an enum, assumptions about type widths.
- `noexcept` only where you can genuinely guarantee it, and always on moves, swaps and
  destructors. A `noexcept` function that throws calls `std::terminate`; it is not decoration.

### 4.4 Attributes

| Attribute | Use when |
|---|---|
| `[[nodiscard]]` | The return value is the point of the call. Default for any pure function, factory, or error code |
| `[[maybe_unused]]` | A parameter or variable is used only in some build configurations — a debug-only assertion argument |
| `[[fallthrough]]` | A `switch` case deliberately falls through. Without it the reader cannot tell intent from bug |
| `[[likely]]` / `[[unlikely]]` | A branch is overwhelmingly one-sided **and** on a measured hot path (P5). Not a guess |
| `[[noreturn]]` | A function never returns — a throw helper, a fatal handler. Lets the compiler drop unreachable paths |
| `[[deprecated("use X")]]` | A public API is on its way out (§5.4). Always with the replacement named |

### 4.5 Types that carry meaning

- **`enum class` always.** Scoped, non-converting, forward-declarable.
- **`std::optional<T>`** for "may legitimately be absent". Not for errors.
- **`std::variant`** for a closed set of alternatives; visit exhaustively so adding a case is a
  compile error rather than a runtime surprise.
- **`std::span`** for contiguous ranges — it replaces pointer+length pairs that can go out of
  sync. Never store one outliving the data it views.
- **`std::string_view`** for non-owning string parameters. Never store one pointing at a
  temporary.
- **Strong types over primitives** where a mix-up is plausible — two same-typed integers meaning
  different things are one transposition from a silent bug, and a distinct type makes it a build
  error. Use judgement; not every integer needs a wrapper.

### 4.6 Conversions and casts

**Never use a C-style cast.** `(int)x` is unsearchable, silently selects among four different
operations, and can quietly become a `reinterpret_cast` when a type changes. Prefer, in
decreasing order of acceptability:

| Cast | Use |
|---|---|
| `static_cast<T>` | Ordinary, intended conversions. The overwhelming default |
| `std::bit_cast<T>` | Reinterpreting the bits of an object. Replaces the pointer-cast idiom entirely |
| `const_cast` | Almost never. Casting away const to *write* is undefined if the object is genuinely const. Needing it usually means an interface is wrong |
| `reinterpret_cast` | Rare, and always with a comment justifying it. Byte-level buffer access is the legitimate case |
| `dynamic_cast` | Only across a polymorphic hierarchy where the alternative is worse. A `dynamic_cast` chain is usually a missing virtual function |

**Narrowing is explicit or it is a bug.** No implicit `int64_t` → `int32_t`, no
`size_t` ↔ signed mixing left to the compiler. Where a narrowing is intended, `static_cast` it
and, if the value could exceed the target, check first. Use braced initialisation where a
narrowing conversion should be a compile error. Compare signed to signed and unsigned to
unsigned; index containers with their own size type.

### 4.7 Macros

**Macros ignore scope and namespaces, so they are a last resort.** Permitted only for:

- header guards (`#pragma once` preferred),
- platform and compiler detection,
- compile-time configuration,
- assertions and logging, where capturing `__FILE__`, `__LINE__` and the source expression is
  the entire point and no function can do it.

Everything else uses `constexpr`, `inline` functions, templates or `enum class`. A macro that
does survive is `UPPER_CASE`, prefixed to avoid collisions, and wrapped so it behaves as one
statement at any call site.

### 4.8 Standard library first

**Prefer the standard library to a custom implementation.** `std::vector`, `std::array`,
`std::unordered_map`, `std::span`, `std::optional`, `std::string_view`, `std::chrono`,
`std::filesystem`, `std::bit_cast`, `<algorithm>`.

A hand-rolled container or algorithm needs a profile showing the standard one is the bottleneck
(P5) plus a comment recording that measurement. "It could be faster" is not a reason.

**Prefer standard algorithms to hand-written loops where they say more.** `std::find_if`,
`std::transform`, `std::accumulate`, `std::sort`, `std::any_of`, `std::ranges::*` — an algorithm
names the intent, and a raw loop makes the reader infer it. Do not force it: a loop doing several
things at once, or one whose index arithmetic is the point, is clearer written out. Clarity
decides, not purity.

**`std::pmr`** is worth considering only for an allocation-heavy subsystem, after a profile
shows allocation is the cost. Not a default.

### 4.9 Templates, concepts and compile-time work

- Templates for genuine parametric reuse, not to avoid writing a function twice.
- **Constrain every template parameter with a concept.** An unconstrained template fails deep
  inside instantiation with an unreadable error; a constrained one fails at the call site.
- Keep templates thin — a template wrapper over a non-template implementation keeps compile time
  and binary size sane.
- Prefer `if constexpr` to tag dispatch and SFINAE.
- **Avoid template metaprogramming unless it buys correctness or measured performance.** Clever
  type-level computation is the hardest code in any codebase to debug, and P4 applies with force.
- Compile time is a feature. A heavy template instantiated in a widely-included header taxes
  every translation unit.

### 4.10 Lambdas

- Keep them short. A lambda past a dozen lines wants to be a named function.
- **Capture explicitly.** `[&]` and `[=]` hide lifetime bugs: `[&]` on a lambda that outlives the
  scope is a dangling reference, and `[=]` silently copies more than intended. Name what you
  capture. A blanket capture is acceptable in a small, immediately-invoked lambda where the scope
  is obvious.
- Never capture `this` into something that outlives the object; capture the members you need.
- Prefer a stateless lambda where one will do — it converts to a function pointer and is easier
  to reason about.
- An immediately-invoked lambda is the right way to initialise a `const` that needs several
  statements.

### 4.11 Polymorphism and interfaces

- **Composition over inheritance.** Inherit only for a genuine interface relationship.
- **Static polymorphism** (templates, CRTP) when the type is known at compile time and the call
  is hot; **dynamic polymorphism** at genuine runtime boundaries — where the implementation is
  chosen at run time and the dispatch cost is amortised over real work.
- Interfaces are **narrow**. Prefer several small ones to a broad one, so implementers are not
  forced to stub methods.
- A polymorphic base needs a `virtual` destructor, or a `protected` non-virtual one if deletion
  through the base is never intended.
- Do not add a virtual to "keep options open". Add it when a second implementation exists.

### 4.12 Errors and exception safety

vkML **throws; it does not abort.** Aborting is defensible for a command-line binary, not for a
library embedded in someone else's process — it would take down their session. Every exception
maps to a natural exception in the binding layer.

The exception hierarchy is rooted at one base type with distinct derived types per failure class
(shape, dtype, device, index, unimplemented, internal, out-of-memory), so callers can catch
precisely. Assertions come in three levels, and the distinction matters:

| Level | Meaning | Active in release |
|---|---|---|
| User error | Bad arguments to a public API | yes |
| Internal invariant | A failure means the library itself is broken | yes, deliberately |
| Hot-path invariant | Checked inside per-element loops | no |

Keeping internal invariants on in release is a deliberate trade: a silently corrupted result
costs far more to debug than a branch costs to execute.

Because any invariant check can throw, code between an allocation and its owner must be
exception-safe. Guarantees, in preference order:

1. **No-throw** — destructors, moves, swaps, deallocation. Required.
2. **Strong** — the operation succeeds or leaves state unchanged. Aim for this wherever shared
   state is mutated; do the work on a copy, then swap.
3. **Basic** — invariants hold, nothing leaks, state unspecified. The minimum.

RAII gets most of this automatically, which is the main reason §4.1 is non-negotiable.

**Error messages name the operation, the actual values, and the expectation.** A message that
says only that something is invalid has wasted the throw.

### 4.13 Undefined behaviour

UB is not a performance tool. Signed overflow, out-of-bounds indexing, strict-aliasing
violations, misaligned loads, use-after-move, uninitialised reads and data races are defects even
when the binary happens to work.

Two that bite repeatedly:

- **Dangling from a temporary.** A function returning by value, called twice in one expression to
  get a begin and an end, yields iterators into two different dead objects. Bind to a named
  value.
- **Layout assumptions across a language boundary.** A struct shared with shader code must have
  its size and offsets `static_assert`-ed, not assumed.

Build and test under sanitizers. A clean sanitizer run is evidence; the absence of a crash is
not.

### 4.14 Thread safety

Concurrency is a design decision recorded in an ADR, not something introduced incidentally. Do
not add threads, atomics or locks without one.

Every type is documented as thread-safe, thread-compatible (safe for distinct instances), or
thread-hostile. `const` methods are safe for concurrent readers (§4.3). Objects immutable after
construction are safe to share by construction, which is the cheapest way to get there.

### 4.15 Performance-aware design

Design for performance; do not micro-optimise without measurement (P5, §7).

- **Data layout beats instruction selection.** Contiguous arrays over pointer chases; cache
  locality is usually the dominant term.
- Keep hot structs small and hot fields adjacent; cold data belongs in a side table.
- Allocation is a design concern, not a micro-optimisation. Reserve capacity, reuse buffers on
  repeated paths, and aim for a steady state that allocates nothing.
- Pass large read-only data by `const&` or `span`.
- Prefer the better algorithm before the better constant factor.
- Let the compiler vectorise by writing clean, aliasing-free loops over contiguous data. Reach
  for intrinsics only when a profile and a disassembly justify it.

### 4.16 Headers and includes

- The public include directory is the **public** surface; headers beside sources are internal.
  Nothing public exposes an internal type.
- **Include what you use**, directly. Do not lean on transitive includes.
- **Forward-declare** when only a reference, pointer or return type is needed; include when you
  need the size, a member or a base.
- Every header is self-contained and compiles alone, with `#pragma once` at the top.
- Headers include the minimum — a heavy include in a widely-used header taxes every build. PImpl
  is the escape hatch when an implementation detail would otherwise leak.
- Definitions in the source file unless the function is genuinely tiny, a template, or
  `constexpr` and needed at compile time.
- **Circular dependencies are an architecture failure** (§3.1), not an include problem. Fix the
  layering; do not paper over it with forward declarations.

### 4.17 Naming and comments — the maturity standard

This is what separates a codebase a stranger can maintain from one only its author can. Hold it
to the standard of the projects in §13. Neither half is cosmetic: a wrong name and a stale
comment both actively mislead, which is worse than silence.

**Names state what a thing is, in the domain's own vocabulary.**

- Use the word the field uses, and use one word per concept across the whole codebase — if it is
  a *workgroup* in one file it is not a *block* in the next.
- **Length scales with scope.** `i` in a three-line loop is correct; `i` as a member is not.
- **Abbreviate only what the domain abbreviates.** Established acronyms are fine; `mgr`, `tmp2`,
  `res`, `val`, `do_stuff` are not.
- **Booleans read as assertions**, so call sites read as English: `is_contiguous()`,
  `has_broadcast_stride()`, `supports(x)`.
- **No type in the name.** Not `float_value`, not `p_node`, not `vec_dims`.
- **A name needing a comment to be understood is the wrong name.** Rename first.
- Banned outright: `data2`, `temp`, `foo`, `helper`, `utils`, `misc`, `manager`, `process()`,
  `handle()`, and any name whose only meaning is "the other one".

**Comments explain *why*. The code already says *what*.**

The test: **would a competent reader be surprised?** If yes, explain the reason, not the
mechanics. If no, write nothing — a comment restating its line drifts out of date and then lies.

What earns a comment:

- **A non-obvious constant, with its derivation** — where the number came from, what breaks if it
  changes.
- **A rejected alternative and the cost that rejected it.** The most valuable kind here: it stops
  the next person redoing the analysis, or "fixing" the code back to the version that was wrong.
- **A deliberate divergence** from the obvious implementation, from the reference framework, or
  from a specification — with the reason attached.
- **A hardware or driver constraint** that makes otherwise-odd code necessary.
- **An invariant a caller must uphold** that the type system cannot express.

What must not appear: restatement of the line; commented-out code (Git has it); decorative
banners with no content; a `TODO` with no owner and no trigger (P7); anything that will be false
after the next edit — cite a source rather than copying a value.

**The standing test for both:** could a contributor who has never spoken to you read this file
and make a correct change?

### 4.18 Features not yet adopted

- **Modules.** Do not migrate until toolchain support is stable and uniform across every
  supported compiler and the build system handles them without special-casing. The benefit is
  build time; the cost of being early is a build nobody else can reproduce.
- **Coroutines.** Do not introduce them without a genuine asynchronous use case that is awkward
  to express otherwise. They bring a lifetime model of their own, and adding it for style is a
  poor trade.

Revisit both by proposal and ADR, not by drift.

---

## 5. Architecture

### 5.1 Dependencies and boundaries

- **Dependencies point one way**, down the layer stack (§3.1). If a lower layer needs something
  from above, invert it with an interface the lower layer owns.
- **Module boundaries are contracts.** A module states what it guarantees and what it requires;
  everything else is private and may change without notice.
- **One module, one reason to change.** A file that changes for two unrelated reasons is two
  files.
- **Encapsulate representation.** Expose behaviour, not fields.
- **New code goes in the layer that owns the responsibility**, even when that is less convenient
  than the layer you happen to be editing.

### 5.2 Backend philosophy

vkML targets more than one execution backend and will target more. The rule that keeps that from
fragmenting the codebase:

```
    generic implementation  →  correct everywhere
            ↓
    verified against the oracle
            ↓
    profiled on a real workload
            ↓
    specialised path  →  only where the profile justifies it
```

- **One generic implementation must work on every supported device.** It is the contract; a
  specialised path is an optimisation on top of it, never a replacement for it.
- **Detect capabilities, not vendors** (§3.2). Branch on what the device reports it can do, not
  on who made it. Vendor identity is at best a proxy, and it goes stale with every new part.
- A specialised path must produce results the generic path would have produced, within the
  numerical contract (§3.3) — and where it changes a fold order, §3.3's re-derivation applies.
- **Every specialised path is a permanent maintenance obligation**: another combination to test,
  another thing to keep correct. It earns its place with a measurement (P5) and keeps it by
  staying measurably ahead.
- Removing a specialised path that no longer pays is a normal, encouraged change.

An unimplemented operation on a backend must fail clearly, naming the operation — or be routed
to one that can run it, if that routing exists. It must never silently produce a wrong result.

### 5.3 Dispatcher philosophy

```
    operator  →  dispatcher  →  backend  →  kernel
```

**The dispatcher owns backend selection. Operators never know which backend executes them.**

- An operator describes *what* to compute — shapes, dtypes, parameters. It contains no branch on
  device, backend or vendor. A `if (device == ...)` in operator code is a layering violation.
- The dispatcher decides *where*, using the backend's own declaration of what it supports.
- The backend decides *how*, selecting among its kernels.
- A kernel does the work and knows nothing above it.

This is what makes adding a backend an addition rather than a rewrite, and it is why the
capability predicate is asked rather than inferred. Keep the seam clean even when a shortcut
would be convenient: the shortcut is paid for once per backend, forever.

**Do not build dispatch machinery beyond the need.** A flat table sized to the actual operator
and backend count is the right structure until it is not; a plugin registry, dynamic loading or a
multi-key dispatch lattice are answers to problems this project may never have (P2, P6). Add
indirection when a second case exists, not before.

### 5.4 API evolution

**Public APIs are stable. Internal APIs may evolve freely.** The boundary is explicit (§4.16),
and knowing which side you are on decides how much care a change needs.

- **Internal**: change it. Update the callers, keep the tests green, done.
- **Public**: a breaking change needs a migration path — a deprecation period with
  `[[deprecated("use X")]]` naming the replacement, or an ADR recording why a clean break is the
  lesser cost.
- **Prefer additive change.** A new overload or a defaulted parameter breaks nobody.
- **Widening what a function accepts is safe; narrowing is not.** Same for return values in
  reverse.
- A deliberate divergence from the framework being mirrored is a public contract too: document
  it at the declaration and pin it with a test, so it stays a decision rather than drifting into
  a surprise.
- Removing a public symbol is a release-note event.

Until the first stable release, "public" means "documented as public and covered by tests" rather
than "frozen" — but the discipline starts now, because retrofitting it is what makes libraries
ossify.

### 5.5 Code smells — stop and rethink

These are not style complaints. Each one reliably predicts a design that will be expensive later.
On seeing one, stop and reconsider before adding to it (§6):

- A function past ~100 lines, or one that needs a section comment to be navigable.
- More than about five parameters — usually a missing struct, or a function doing two jobs.
- Nesting past three levels; usually an early return or an extracted function is waiting.
- A **boolean parameter that selects behaviour**. `f(x, true)` is unreadable at the call site and
  the function is really two functions.
- The same logic in a third place. Twice can be coincidence; three times is a missing
  abstraction.
- A `switch` that keeps growing, especially the same set of cases switched over in several
  places — that is a virtual function or a table waiting to exist.
- A parameter that exists for exactly one caller.
- A comment explaining what a block does, where extracting a named function would say it better.
- A test that must be rewritten whenever the implementation changes — it is testing the
  implementation, not the behaviour (§8).

---

## 6. Refactoring

**Before adding a feature, decide whether what exists can carry it.** Usually it can — say so in
a line and move on. Escalate to a real analysis when you see:

- the feature needs a special case inside an existing abstraction to be accepted;
- a parameter exists only to select behaviour for one caller;
- the change forces an edit in a layer that should not have known about it;
- the same conditional appears in a third place.

**When a signal fires, fix the abstraction first, in its own commit.** Building on a known-bad
abstraction is never cheaper later.

**When *not* to refactor**, which matters as much:

- The code is ugly but stable, well-tested, single-caller, and no feature is pending.
- You are mid-feature and the refactor is unrelated. Note it, finish, do it separately.
- You cannot state the improvement in one sentence.
- Behaviour would change — that is a redesign, and it needs §2 step 6.

**Refactoring preserves behaviour, and here that is checkable rather than a judgement**: a change
that alters no fold order must leave goldens byte-identical (§3.3). If they move, you did not
refactor. Work in small steps with the suite green between each.

---

## 7. Performance engineering

**No optimisation is accepted without evidence** (P5). Not because it looks faster, not because
it is theoretically better, not because another project does it.

### Gate zero — is this the right work at all?

Before profiling anything: **is the functionality this optimisation serves complete?** (P2.) If
not, implement what is missing instead — a missing operation is infinitely slower than a
suboptimal one. Optimise when the path is on a real workload's critical path, the functionality
around it is complete and tested, and a profile identifies it. Otherwise record the opportunity
(P7) and move on.

### The loop

1. **Profile.** Find the actual bottleneck. Intuition about which line is hot is wrong more often
   than right.
2. **Hypothesise.** State what you expect to change, by how much, and **what result would prove
   you wrong** — before measuring. A prediction made afterwards is not a prediction.
3. **Check the constraints.** Does it alter fold order (§3.3)? Do the recorded laws already
   forbid it? Cheapest possible refutation first.
4. **Gate on resources before timing.** Where the toolchain can report register pressure, spills,
   scratch or occupancy for a candidate, read them and reject on those *before* benchmarking. A
   candidate rejected without ever being timed is the cheapest possible rejection.
5. **Measure**, to the rules below.
6. **Verify correctness.** Full suite plus goldens.
7. **Accept or roll back.** A failed hypothesis is a result — record it (P7) rather than deleting
   it. Failed predictions are where laws come from.

### Measurement rules

These are hard-won; the derivations and the measured figures behind them are in
`MEASUREMENT-AUDIT.md`.

1. **Characterise the noise floor before claiming an effect.** An effect smaller than the spread
   of repeated identical runs is not an effect.
2. **Device-side timers are far more reproducible than wall clock.** Prefer them; use wall clock
   only when the measured operation dominates the measured window, and check that ratio rather
   than assuming it.
3. **Report the minimum** of a timing distribution, never the mean. The tail measures the
   machine, not the work.
4. **Never sum per-dispatch timers across concurrent work.** Use an enclosing window; overlapping
   work reports overlapping intervals and the sum counts the same time repeatedly.
5. **Never compare a profiled run against an unprofiled one.**
6. **Never benchmark with validation or debug layers enabled.** They change what they measure.
7. **Warm caches and pipelines before timing.** Compilation is setup, not measurement.
8. **An A/B is valid only if the A arm is the frozen, unmodified baseline.**
9. **Prefer criteria that compare bytes** — they cannot be perturbed by measuring.
10. **Check every gate for vacuity before trusting a pass.**
11. **Two independent calculations agreeing is not confirmation** — it is a warning the
    experiment cannot distinguish them.

Benchmark in the configuration intended for it, and treat a stored baseline as a claim: updating
one needs the same evidence as any other performance statement.

---

## 8. Testing

Tests ship with the change. A feature without tests is unfinished.

**Test behaviour, not implementation.** Exercise the public contract, not private internals.
A test coupled to internals breaks on every refactor, and a suite that cries wolf stops being
read — which costs more than the coverage was worth. If behaviour is hard to test without
reaching inside, the design is telling you something (§5.5).

| Tier | What it covers |
|---|---|
| **Unit** | Every operation: every dtype, contiguous + strided + broadcast, edge shapes (empty, size-1, non-power-of-2, minimum and maximum rank) |
| **Oracle** | The chain in §3.4, in order |
| **Gradient** | Analytical against numerical differentiation *and* against the reference framework, for every input and parameter |
| **Integration** | Layers, optimisers over many steps (compare trajectories, not just endpoints), whole-model parity |
| **Regression** | Golden values. Exact, because results are deterministic. Every fixed bug gets one |
| **Property** | Randomised shapes, strides and dtypes, where hand-written cases miss combinations |
| **Resource** | Every allocation freed; peak usage within prediction; no device loss under stress |

- Run correctness tests in the mode that makes a failure point at the offending operation rather
  than at a deferred execution boundary.
- Run device tests with validation layers on; any validation error fails the build.
- **A test that cannot fail is worse than no test** — it manufactures confidence. Verify a new
  test fails against deliberately broken code before trusting its pass.
- Tolerances come from the central policy and nowhere else (§3.3).
- Cover the cases a random sweep cannot reach: NaN and infinity, exact equality, empty inputs,
  contention, and the numerically awkward tails.

---

## 9. Documentation

Documentation is a deliverable. Comment philosophy is §4.17; this is about the artifacts.

- **Public API**: every public type and function documents purpose, parameters, return value,
  what it throws, and any lifetime or ownership requirement.
- **Design rationale goes in `docs/`**, not in a commit message where nobody will find it. A
  decision that is expensive to reverse, changes a public contract or an invariant, or was taken
  against an obvious alternative → an **ADR**, following the existing format: *Context →
  Measurements → Options considered, each with a verdict → Decision → Guardrails adopted now →
  Rejected alternatives, recorded so they are not rediscovered.*
- **Experiments get a record** stating the hypothesis, the quantified prediction and the
  falsification criteria *before* the results.
- **Update what your change invalidated**, in the same commit. Stale documentation is worse than
  none, because it is trusted.
- Examples must build and run in CI, or they rot.

---

## 10. Review checklist

Run this against your own work before declaring anything complete. Scale it to the change: the
mechanical items always, the rest as applicable.

**Mechanical** — no judgement required, so just run them: layering check, formatter, static
analysis, a warning-free build in every configuration, the full test suite, sanitizers for
memory-touching changes.

**Always:**
- [ ] Every claim I am about to make is supported by something I actually ran (P5, §0).
- [ ] The change is one logical unit.
- [ ] Anything left unfinished is stated explicitly, not implied (P7).
- [ ] The surrounding code is no worse than I found it.

**Architecture:** correct layer · one responsibility · right abstraction level · no new coupling ·
no circular dependency · operators contain no backend branch (§5.3) · no smell from §5.5.

**Correctness:** edge cases (empty, minimum rank, size-1, broadcast, non-contiguous, aliased) ·
overflow · error paths tested, not just the happy path.

**C++:** ownership stated in types · Rule of Zero, or all five · RAII for every resource · no raw
`new`/`delete` · moves `noexcept` · `const` correct · attributes used where they carry meaning ·
no C-style casts, no implicit narrowing · no macro that a function could be · standard library
preferred · includes minimal and direct · headers self-contained · no UB.

**Exception safety:** no-throw where required · strong guarantee where shared state is mutated ·
no leak on any throw path.

**Numerics:** fold order unchanged, or bound re-derived and goldens re-pinned · tolerances from
the policy · determinism preserved.

**Performance:** no unnecessary allocation on hot paths · layout cache-friendly · any
optimisation backed by measurement (§7) · no regression.

**Testing:** new tests can actually fail · tests exercise behaviour, not internals · regression
test for every fixed bug · oracle chain followed.

**Documentation:** public API documented · non-obvious decisions explained · invalidated docs
updated · ADR where the decision warrants one.

**Maintainability (§4.17):** every name states what the thing is in domain vocabulary · booleans
read as assertions · every comment explains *why* and none restates its line · non-obvious
constants carry their derivation · rejected alternatives recorded · no commented-out code · no
ownerless TODO · no duplication that wants an abstraction · no speculative abstraction without a
second caller.

The question that decides it: **could a contributor who has never spoken to you read this and
make a correct change?**

---

## 11. Git workflow

- **One logical change per commit.** A refactor and a feature are two commits, always. A commit
  doing two things cannot be reverted, reviewed or bisected cleanly.
- **Every commit builds and passes tests.** Broken intermediate commits destroy bisection, which
  is the main reason history is worth keeping.
- **Commit messages explain *why*.** Subject in the imperative, ≤72 characters, no trailing
  period. Blank line. Body covering what changed and, more importantly, the reasoning — the
  alternative considered, the measurement that justified it, the constraint that forced it.
  Reference documents and ADRs by path. Every commit must be understandable without opening the
  diff.
  Good: `Add tensor broadcasting support` · `Implement Adam optimizer` ·
  `Refactor descriptor cache` · `Fix gradient accumulation bug`.
  Never: `fix` · `update` · `changes` · `working` · `final` · `test` · `misc`.
- **Never commit broken code** — failing to compile, breaking tests, a known regression, or
  incomplete functionality that is not clearly isolated. Build, test and format first.
- **The history is engineering documentation.** A contributor should be able to read it and
  understand how the project evolved: decisions, milestones, refactors, optimisation phases.
- **No artificial history.** Never fabricate commits for states that never existed.
- **No AI attribution.** No `Co-Authored-By` for tools, no generated-by trailers. The commit
  records the change, not what typed it.
- **Keep history linear and readable.** Squash experimental or noisy commits before merging.
  Rewrite only unpushed work, and only to fix a genuine mistake.
- **One objective per pull request.** Do not combine unrelated features, refactors, formatting
  and optimisations.
- Never commit build output, virtual environments, generated artifacts or vendored study
  material.
- Propose commits as a list with messages; let the author decide when to run them.

---

## 12. Build system

- **Modern, target-based CMake.** Everything attaches to a target: `target_link_libraries`,
  `target_include_directories`, `target_compile_features`, `target_compile_definitions`,
  `target_compile_options`.
- **No directory-scoped or global commands** — `include_directories`, `link_libraries`,
  `add_definitions`, or mutating `CMAKE_CXX_FLAGS` globally. They leak into every target and make
  a build impossible to reason about locally.
- **Get the visibility right**: `PRIVATE` for implementation needs, `PUBLIC` for what consumers
  also require, `INTERFACE` for header-only. Over-broad visibility propagates dependencies nobody
  asked for.
- **Named presets for every configuration** anyone is expected to build, so a build is
  reproducible from one command and CI runs what a developer runs.
- **Generated sources are build steps with declared dependencies**, including dependency files
  where a generator supports them — an edit to a shared include must trigger the rebuilds that
  depend on it.
- Warnings-as-errors available in every configuration and on in CI.
- **Dependency admission is a decision, not a convenience.** A new dependency needs a reason, a
  licence compatible with the project's, and an owner. Prefer the standard library (§4.8), then a
  small well-maintained library, then vendoring. Reference material studied but never linked is
  kept clearly separate from code that ships.

---

## 13. Learning from open source

Before proposing a significant architectural change or optimisation, find out how mature projects
solved the same problem — **and why**.

Relevant prior art spans the frameworks in the same problem space, the compiler and kernel
projects that solved dispatch and portability, and the large C++ codebases worth imitating for
engineering practice rather than for design. Where study material is kept in the repository it is
read-only, at pinned revisions, never linked and never modified.

**How to study:**

1. Read for **intent**, not implementation. Ask what constraint produced the shape.
2. **Convergence is evidence about the problem.** A choice made independently by several teams
   with different languages and hardware tells you something about the problem, not about the
   teams. Where they *disagree*, the question is genuinely open — treat it as such.
3. **Apply the filter.** Does the constraint that motivated it hold here? Their target hardware,
   scale, workload and guarantees are usually different. A technique that trades accuracy for
   speed may be right for an inference engine and incompatible with §3.3 — such a case is
   rejected outright however fast it is.
4. **Never copy code.** Understand the idea, then implement it for this project's constraints.
   Copying also imports licence obligations.
5. **Record provenance.** A borrowed idea is documented with its source, why it exists there, and
   why it applies here.

Do not assume the reference projects are ahead on everything. Where this project has capability
they lack, use it.

---

## Quick reference

Exact commands and preset names live in the repository's build configuration and CI definition;
read those rather than memorising them here. The shape of the loop:

```
configure + build (debug)      →  develop
run the C++ and Python suites  →  correctness
layering + format + tidy       →  mechanical gates
sanitizer build + suite        →  memory safety
release-with-debug-info build  →  benchmarking only
eager / per-op mode            →  make a failure point at the operation
```

**If you remember nothing else:** read before writing · fix the abstraction before building on
it · generic before specialised · never break determinism · measure before claiming · test
behaviour, not internals · name things so the next reader needs no explanation · comment the
*why*, never the *what* · say what you actually did.
