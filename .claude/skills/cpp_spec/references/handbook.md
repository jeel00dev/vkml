# vkML Engineering Handbook

The detailed rules. The philosophy and judgement that decide *why* these rules exist are in the
constitution (`../SKILL.md`), which is assumed read — principles are cited as P1–P7 throughout.

**States no project state.** Operator counts, device capabilities, measured throughput and tuned
constants live in `docs/`: `ARCHITECTURE.md`, `PHASE2-MANIFESTO.md`, `THEORY.md`,
`MEASUREMENT-AUDIT.md`, `adr/*.md`. Cite them; never copy a number out.

**Parts.** A Invariants · B Architecture · C Design · D Writing · E Verifying · F Shipping.

---

# Part A — Invariants

## 1. Invariants

Hard constraints. Violating one is a defect regardless of how good the code looks.

### 1.1 Layering

**A layer may depend only on layers strictly below it, never on a sibling.** Higher-level
concepts must not leak downward: a backend must not know what a neural-network layer is.

Policy, enforced automatically in CI. When the check rejects a change, the fix is the design.
Current layers are in `ARCHITECTURE.md`; the list grows.

### 1.2 Portability and target hardware

**Query capabilities at runtime; never assume them.** Respect the active device, not the one this
was written on.

- Never presume cooperative matrix, float atomics, a subgroup size, a shared-memory size, an
  address model, or any extension.
- A query returning "unknown" — a vendor property on another vendor's hardware — is a reason to
  **decline** the decision it would have informed, not to guess.
- Push-constant budget, workgroup limits and shared memory are device properties. Fit what is
  reported; if metadata will not fit, move it to a buffer rather than assume headroom.

**Portability of the code itself follows the same rule.** No reliance on undefined behaviour
(§7.10), compiler extensions, byte order, struct padding, or platform assumptions — unless
isolated behind a named abstraction with the assumption documented at it, so a port has one place
to look.

### 1.3 The numerical contract

- **Results are bit-reproducible** for a given device and build.
- **Reduction order is fixed.** Atomic accumulation is forbidden — its ordering varies run to run
  and float addition is not associative. This also makes the code portable to devices lacking it.
- **Accumulate in at least fp32**, whatever the storage precision.
- **A tolerance is a property of the operation**, declared once centrally with a citable source
  and derived *before* implementation. Never chosen at a call site, never widened to make a test
  pass — a failure is a bug until proven otherwise.
- **Changing a fold order requires a re-derived bound and re-pinned goldens**, decided first.

Prefer an acceptance criterion that compares bytes: it cannot be perturbed by measuring, and it
catches an ordering change a tolerance would hide.

### 1.4 The correctness oracle

```
reference backend  vs  established framework   →  is the maths right?
optimised backend  vs  reference backend       →  is this implementation right?
```

**Never write the optimised path first.** Each comparison must have exactly one candidate cause;
that is the entire value. The reference backend is written to be *obviously correct*, not fast —
a measuring instrument that also ships.

### 1.5 Mechanical enforcement

**If a tool can decide it, the tool decides it.** Formatting, static analysis, layering and
naming conventions are configured once and satisfied, never argued in review and never restated
here.

# Part B — Architecture

## 2. Architecture

### 2.1 Dependencies and boundaries

- **Dependencies point one way**, down the stack (§1.1). If a lower layer needs something from
  above, invert it with an interface the lower layer owns.
- **Module boundaries are contracts**: what it guarantees, what it requires. Everything else is
  private and may change without notice.
- **Encapsulate representation.** Expose behaviour, not fields.
- **New code goes in the layer that owns the responsibility**, even when that is less convenient
  than the layer you happen to be editing.

### 2.2 Backend philosophy

```
generic implementation → verified against the oracle → profiled on a real workload
                                                      → specialised path, only where justified
```

- **One generic implementation must work on every supported device.** It is the contract; a
  specialised path is an optimisation on top, never a replacement.
- **Detect capabilities, not vendors** (§1.2). Vendor identity is a proxy that goes stale with
  every new part.
- A specialised path must produce what the generic path would, within §1.3.
- **Every specialised path is a permanent maintenance obligation.** It earns its place with a
  measurement (P5) and keeps it by staying measurably ahead. Removing one that no longer pays is
  a normal change.

An unsupported operation fails clearly, naming it, or is routed to a backend that can run it. It
never silently produces a wrong result.

### 2.3 Dispatcher philosophy

```
operator  →  dispatcher  →  backend  →  kernel
```

**The dispatcher owns backend selection. Operators never know which backend executes them.**

- An operator describes *what* to compute. A branch on device, backend or vendor in operator code
  is a layering violation.
- The dispatcher decides *where*, from the backend's own declaration of what it supports.
- The backend decides *how*; the kernel knows nothing above it.

This is what makes adding a backend an addition rather than a rewrite. **Do not build dispatch
machinery beyond the need** — a flat table sized to the actual counts is right until it is not
(YAGNI).

### 2.4 API, versioning and stability

**Public APIs are stable. Internal APIs may evolve freely.** The boundary is explicit (§7.11);
knowing which side you are on decides how much care a change needs.

- **Internal**: change it, update callers, keep tests green.
- **Public**: a breaking change needs a deprecation period naming the replacement
  (`[[deprecated("use X")]]`), or an ADR recording why a clean break costs less.
- **Prefer additive change.** A new overload or defaulted parameter breaks nobody. Widening what
  a function accepts is safe; narrowing is not.
- A deliberate divergence from a mirrored framework is a public contract: document it at the
  declaration and pin it with a test.

**Versioning is semantic.** Major for a breaking change, minor for additions, patch for fixes.
The version states what the *API* promises — a performance change is not a major bump, and a
behaviour change that breaks a caller is, even if the signature is untouched.

**ABI is a separate promise from API** and a stricter one: adding a virtual, reordering members,
or changing a struct's size breaks it without touching a single signature. Until a release
commits to ABI stability, say so explicitly rather than leave it assumed.

Before the first stable release, "public" means documented and tested rather than frozen — but
the discipline starts now, because retrofitting it is what ossifies libraries.

### 2.5 Configuration and constants

**A constant that depends on something must be derived from it, not written down.**

| Kind | Where it belongs |
|---|---|
| Device-dependent (tile sizes, workgroup dims, limits) | Queried, or selected at run time from a tuned table keyed on what was queried |
| Build-dependent | A build-system option with a documented default |
| Genuinely universal (a mathematical bound, a format's magic number) | `constexpr`, with its derivation in a comment |
| Tuned by measurement | A data file or table, versioned, with the measurement recorded |

A bare `constexpr int kTile = 64;` for a hardware-dependent quantity is a portability bug waiting
for a different device — the value is not wrong, its *location* is.

**No magic numbers.** A literal appearing in logic gets a named constant and a reason.

### 2.6 Concurrency model

Concurrency is a design decision recorded in an ADR, not something introduced incidentally. Do
not add threads, atomics or locks without one.

Every type is documented as thread-safe, thread-compatible (safe for distinct instances), or
thread-hostile. `const` methods must be safe for concurrent readers (§7.3). Objects immutable
after construction are safe to share by construction — the cheapest way to get there.

# Part C — Design

## 3. Design principles

Applied pragmatically. Each usually pays; none is a law, and following one past the point where
it helps is its own defect. Most rules later are one of these applied to a situation.

| | Principle | The part that matters here |
|---|---|---|
| **SRP** | One reason to change | The test is not size but *why* it would be edited. Two unrelated reasons means two units |
| **Open/closed** | Extend without editing | Adding a backend, operator or device is an addition, not surgery — through the seams in §2, not speculative hooks |
| **Liskov** | A subtype is usable as its base | An implementation that throws on half the interface, or needs callers to know which one they hold, is a broken abstraction |
| **Interface segregation** | Narrow interfaces | Several small beat one broad; implementers should not stub what they cannot support |
| **Dependency inversion** | Depend on abstractions the *lower* layer owns | This is what makes §1.1's one-way rule achievable |
| **Composition over inheritance** | Inherit only for a genuine interface relationship | Composition is testable in isolation, swappable at run time, and does not couple lifetimes |

**DRY — one authoritative definition per fact.** Applies hardest to *knowledge*: a constant, a
rule, an invariant. Two code paths that merely look alike but change for different reasons are
not duplication, and merging them couples them. Duplicate code is cheap; a duplicated *decision*
is not.

**KISS — few concepts, not few lines.** A dense one-liner is not simple.

**YAGNI — build for the requirement in front of you.** Speculative generality costs certainly and
pays uncertainly, and it is usually generalised along the wrong axis. Add indirection when the
second case exists (P6).

## 4. Complexity

**Choose the algorithm before tuning the constant.** Most large performance failures are the
wrong algorithm; instruction-level work cannot rescue one.

Four costs, all of which matter and only the first of which is usually considered:

| | Ask |
|---|---|
| **Time** | How does work grow with input? An accidental quadratic in a loop nest is invisible at test scale and fatal at real scale |
| **Memory** | Peak, not total. Allocating a full intermediate where a streaming pass would do is what turns a working size into an out-of-memory |
| **Cache** | How many times is a byte re-read, and is access sequential? On memory-bound work this dominates the instruction count |
| **I/O and transfer** | Every crossing of a slow boundary — host to device, disk, network. One avoided round-trip usually beats any amount of local tuning |

State non-obvious complexity in a comment, in the terms that matter — a kernel's cost is more
usefully given in bytes moved than operations. A worse asymptotic bound can still win at the
sizes actually used: legitimate **when measured** (P5) and recorded with the range it holds for.

## 5. Code smells, and deleting code

**Smells are not style complaints.** Each reliably predicts a design that gets expensive. On
seeing one, stop and reconsider before adding to it (§14):

- A function past ~100 lines, or one needing section comments to navigate.
- More than about five parameters — usually a missing struct, or two functions in one.
- Nesting past three levels; an early return or an extracted function is waiting.
- **A boolean parameter selecting behaviour.** `f(x, true)` is unreadable at the call site, and
  the function is really two functions.
- The same logic in a third place. Twice is coincidence; three times is a missing abstraction.
- A growing `switch`, especially the same cases switched in several places — a virtual function
  or a table wants to exist.
- A parameter existing for exactly one caller.
- A comment explaining what a block does, where extracting a named function would say it better.
- A test needing rewriting whenever the implementation changes — it tests the implementation
  (§10).

**Deleting code is an improvement.** Not every contribution is an addition. Delete on sight:
unused functions, dead branches, a configuration option nobody sets, an abstraction with one
implementation that was built for a second that never came, a specialised path no longer ahead of
the generic one (§2.2), and a test that cannot fail.

Git holds the history — deleted code is recoverable, and code kept "just in case" is code that
must be read, compiled, and kept correct forever. **The burden of proof is on keeping it.**

---

# Part D — Writing

## 6. Readability

The constitution's rule — *good code does not make the reader think* — as concrete practice.

**Function shape**

- **One screen.** A function should usually fit in one, not because of an arbitrary limit but
  because an algorithm you cannot see whole is one you cannot reason about.
- **One responsibility.** A function that validates, allocates, logs, dispatches and updates a
  cache is five functions. This is about mental load, not purity.
- **Guard clauses over nesting.** Handle the exceptional cases first and return; then the main
  path is unindented and reads straight down. An `if/else if/else if/else` chain that assigns to
  one variable usually wants to be early returns or a table.
- **Read top to bottom.** Structure a body as: validate, prepare, do the work, return. A reader
  should not have to jump backwards.

**One level of abstraction per function.** This is the strongest single indicator of mature code.
A function that mixes high-level policy with low-level API calls forces the reader to hold two
mental models at once:

```
train_step()        // policy: forward, loss, backward, update
  → backend call    // one level down
    → kernel        // one level down again
```

Each layer names *what* happens; the layer below decides *how*. If a high-level function contains
a raw device API call, a layer is missing.

**Names carry the meaning.** `foo(a, b, c)` tells the reader nothing; `compile_shader(source,
options)` tells them everything without opening the definition. A call site should be readable
without following it — that is what makes local reasoning possible.

**Prefer self-documenting code to a comment.** A comment saying *convert bytes* above a line is
worse than a function named `bytes_to_elements()`. Reach for a better name before reaching for an
explanation; comments are for what a name cannot carry (§9).

**No hidden work.** A call must not secretly allocate, synchronise, copy, compile a shader, block,
or throw unless the name says so. A function named `realize()` may allocate and synchronise — that
is what it means. A function named `size()` may not.

**Make illegal states impossible.** Prefer a factory or constructor that cannot produce an invalid
object over a default-constructed one plus an `initialized` flag. A state that cannot be
represented needs no test, no check and no comment.

**Explicit types where they inform.** `auto` is right when the type is obvious from the
initialiser or unspeakable; it is wrong when naming the type is what tells the reader what is
happening. `auto x = compute();` hides the thing the reader most needs to know.

**Every abstraction must reduce complexity.** A wrapper that adds a name and forwards its
arguments has made the codebase larger and no simpler. Before adding one, see the constitution's
three questions.

## 7. C++ language

C++20. Use the standard it targets; check the build configuration before assuming a feature
exists. This is a house style, not a summary of the Core Guidelines.

### 7.1 Ownership and lifetime

- **Rule of Zero by default.** A class managing no resource declares no destructor, copy or move.
- **Rule of Five when you must.** Writing any of the five means writing or `= delete`-ing all
  five; a half-set is a silent bug.
- **RAII for every resource** — memory, device handles, descriptors, timers. No manual release.

| Intent | Type |
|---|---|
| Sole owner | `std::unique_ptr<T>` — the default owning pointer |
| Genuinely shared, unpredictable lifetime | `std::shared_ptr<T>` |
| Break a cycle; observe without extending | `std::weak_ptr<T>` |
| Non-owning, must outlive the callee | `T&` or `T*` |
| Non-owning contiguous range | `std::span<T>` |

`shared_ptr` is a decision, not a default; where this codebase uses it an ADR says why. Never
`new`/`delete` in application code. Raw pointers and references are **non-owning, always**.

### 7.2 Value semantics and moves

- Prefer values; shared mutable state is a permanent cost.
- Return by value — out-parameters obscure dataflow and defeat elision.
- Sink parameters by value and moved; read-only by `const&`, or by value when trivially copyable
  and small.
- Mark moves, swaps and destructors `noexcept`, or containers copy instead of moving.
- A moved-from object is valid but unspecified: assign or destroy, never read.
- Do not `std::move` a return value; it blocks elision.

### 7.3 const, constexpr, consteval, noexcept

- **`const` by default**; non-const needs a reason.
- `const` methods must be **thread-safe for concurrent readers** — a `mutable` cache breaks that
  unless synchronised.
- **`constexpr`** wherever computation moves to build time. **`consteval`** when a function must
  *never* run at run time, making that a compile error rather than a convention.
- **`static_assert` for invariants the type system can check**: struct sizes and offsets shared
  with shader code, table sizes matching an enum, type widths.
- `noexcept` only where genuinely guaranteed. A `noexcept` function that throws terminates.

### 7.4 Attributes

| Attribute | Use when |
|---|---|
| `[[nodiscard]]` | The return value is the point of the call. Default for pure functions and factories |
| `[[maybe_unused]]` | Used only in some build configurations — a debug-only assertion argument |
| `[[fallthrough]]` | A `switch` case deliberately falls through; without it, intent is indistinguishable from bug |
| `[[likely]]`/`[[unlikely]]` | A branch is overwhelmingly one-sided **and** on a measured hot path (P5) |
| `[[noreturn]]` | A throw helper or fatal handler; lets the compiler drop unreachable paths |
| `[[deprecated("use X")]]` | A public API is on its way out (§2.4). Always name the replacement |

### 7.5 Types that carry meaning

- **`enum class` always** — scoped, non-converting, forward-declarable.
- **`std::optional<T>`** for legitimate absence; not for errors.
- **`std::variant`** for a closed set; visit exhaustively so a new case is a compile error.
- **`std::span` / `std::string_view`** for non-owning ranges — they replace pointer+length pairs
  that can disagree. Never store one outliving what it views.
- **Strong types over primitives** where a mix-up is plausible: two same-typed integers meaning
  different things are one transposition from a silent bug. Judgement, not every integer.

### 7.6 Conversions and casts

**Never use a C-style cast.** It is unsearchable and silently picks among four operations.

| Cast | Use |
|---|---|
| `static_cast<T>` | Intended conversions. The overwhelming default |
| `std::bit_cast<T>` | Reinterpreting an object's bits; replaces the pointer-cast idiom |
| `const_cast` | Almost never. Casting away const to write is UB if the object is const; needing it means an interface is wrong |
| `reinterpret_cast` | Rare, always with a justifying comment. Byte-level buffer access is the legitimate case |
| `dynamic_cast` | Only where the alternative is worse. A chain of them is a missing virtual function |

**Narrowing is explicit or it is a bug.** Braced initialisation where a narrowing should not
compile; `static_cast` with a range check where the value could exceed the target. Compare signed
to signed, and index containers with their own size type.

### 7.7 Macros

**Macros ignore scope and namespaces**, so they are a last resort. Permitted only for: header
guards; platform and compiler detection; compile-time configuration; and assertions and logging,
where capturing `__FILE__`, `__LINE__` and the source expression is the entire point.

Everything else uses `constexpr`, `inline` functions, templates or `enum class`. A surviving
macro is `UPPER_CASE`, prefixed, and wrapped to behave as one statement at any call site.

### 7.8 Standard library first

**Prefer the standard library to a custom implementation** — containers, `std::span`,
`std::optional`, `std::chrono`, `std::filesystem`, `std::bit_cast`, `<algorithm>`. A hand-rolled
replacement needs a profile showing the standard one is the bottleneck (P5), plus a comment
recording that measurement.

**Prefer standard algorithms to hand-written loops where they say more** — `find_if`,
`transform`, `accumulate`, `sort`, `any_of`, `ranges::*`. An algorithm names the intent; a raw
loop makes the reader infer it. Do not force it: a loop doing several things at once, or whose
index arithmetic is the point, is clearer written out. Use `std::ranges` where it improves
clarity, not everywhere.

**`std::pmr`** is worth considering only for an allocation-heavy subsystem after a profile shows
allocation is the cost.

### 7.9 Templates and compile-time work

- Templates for genuine parametric reuse, not to avoid writing a function twice.
- **Constrain every template parameter with a concept** — an unconstrained one fails deep inside
  instantiation instead of at the call site.
- Keep templates thin: a template wrapper over a non-template implementation keeps compile time
  and binary size sane.
- Prefer `if constexpr` to tag dispatch and SFINAE.
- **Prefer runtime polymorphism when compile-time polymorphism costs compile time without a
  measured runtime benefit.** Templates are not free, and the cost is paid by everyone building.
- **Avoid template metaprogramming unless it buys correctness or measured performance** — it is
  the hardest code in any codebase to debug (P4).

### 7.10 Lambdas

- Keep them short; past a dozen lines it wants a name.
- **Capture explicitly.** `[&]` on a lambda outliving its scope dangles; `[=]` copies more than
  intended. A blanket capture is acceptable only in a small immediately-invoked lambda.
- Never capture `this` into something outliving the object — capture the members needed.
- Prefer stateless lambdas; an immediately-invoked lambda is the right way to initialise a
  `const` needing several statements.

### 7.11 Undefined behaviour, headers and includes

UB is not a performance tool. Signed overflow, out-of-bounds indexing, aliasing violations,
misaligned loads, use-after-move, uninitialised reads and data races are defects even when the
binary works. Two that bite repeatedly: **a function returning by value, called twice in one
expression** to get a begin and an end, yields iterators into two different dead temporaries; and
**a struct shared with shader code** must have its size and offsets `static_assert`-ed, not
assumed. Build and test under sanitizers — a clean run is evidence; no crash is not.

Headers:

- The public include directory is the **public** surface; headers beside sources are internal.
  Nothing public exposes an internal type.
- **Include what you use**, directly; do not lean on transitive includes.
- **Forward-declare** for a reference, pointer or return type; include when you need the size, a
  member or a base.
- Every header is self-contained, compiles alone, `#pragma once` at the top, and includes the
  minimum — a heavy include in a widely-used header taxes every build. PImpl is the escape hatch.
- Definitions in the source file unless tiny, a template, or `constexpr` and needed at compile
  time.
- **Circular dependencies are an architecture failure** (§1.1), not an include problem.

### 7.12 Features not yet adopted

- **Modules** — not until toolchain support is stable across every supported compiler and the
  build handles them without special-casing. The benefit is build time; the cost of being early
  is a build nobody can reproduce.
- **Coroutines** — not without a genuine asynchronous use case awkward to express otherwise. They
  bring a lifetime model of their own.

Revisit by proposal and ADR, not by drift.

## 8. Errors, assertions and logging

**Throw; never abort.** A library embedded in someone else's process must not kill it. Every
exception maps to a natural one in the binding layer.

The hierarchy is rooted at one base type with a derived type per failure class — shape, dtype,
device, index, unimplemented, internal, out-of-memory — so callers can catch precisely.
Assertions come in three levels:

| Level | Meaning | Active in release |
|---|---|---|
| User error | Bad arguments to a public API | yes |
| Internal invariant | A failure means the library is broken | yes, deliberately |
| Hot-path invariant | Checked inside per-element loops | no |

Keeping internal invariants on in release is a deliberate trade: a silently corrupted result
costs far more to debug than a branch costs to execute.

**Exception safety.** Any check can throw, so code between an allocation and its owner must be
safe. In preference order: **no-throw** (destructors, moves, swaps — required); **strong**,
succeed or leave state unchanged, wherever shared state is mutated (work on a copy, then swap);
**basic**, invariants hold and nothing leaks, as the minimum. RAII gets most of this
automatically — why §7.1 is non-negotiable.

**Error messages name the operation, the actual values, and the expectation.** A message saying
only that something is invalid has wasted the throw.

**Logging.** Log what someone can act on.

| Level | For |
|---|---|
| `error` | The operation failed and the caller needs to know why |
| `warn` | Something is wrong but recoverable, or a deprecated path was taken |
| `info` | Significant lifecycle events — device selected, backend registered. Rare by construction |
| `debug` | Diagnostic detail, off by default |

- **Never log in a hot loop** unless explicitly diagnosing, and then behind a switch that is off
  by default and costs one predictable branch when off.
- **Format nothing unless the message will be emitted.** Building a string that is then discarded
  can cost more than the work being logged.
- A log line that fires on every normal operation is noise, and noise trains people to ignore the
  channel — which costs you the one message that mattered.
- Logging is not error handling. A logged-and-swallowed failure is a silent failure.

## 9. Naming and comments

A wrong name and a stale comment both actively mislead, which is worse than silence.

**Names state what a thing is, in the domain's vocabulary.**

- Use the word the field uses, and **one word per concept** across the codebase — if it is a
  *workgroup* in one file it is not a *block* in the next.
- **Length scales with scope.** `i` in a three-line loop is correct; `i` as a member is not.
- **Abbreviate only what the domain abbreviates.** Established acronyms are fine; `mgr`, `tmp2`,
  `res`, `do_stuff` are not.
- **Booleans read as assertions**, so call sites read as English: `is_contiguous()`,
  `supports(x)`.
- **No type in the name** — not `float_value`, `p_node`, `vec_dims`.
- **A name needing a comment to be understood is the wrong name.** Rename first.
- Banned: `data2`, `temp`, `foo`, `helper`, `utils`, `misc`, `manager`, `process()`, `handle()`.

**Comments explain *why*. The code says *what*.** The test: **would a competent reader be
surprised?** If not, write nothing — a comment restating its line drifts and then lies.

**Earns a comment:**

- **An invariant** a caller or the surrounding code must uphold that the type system cannot
  express.
- **An assumption** the code depends on, especially one that could silently stop being true.
- **The reasoning** behind a non-obvious approach — why *this* way.
- **A rejected alternative and the cost that rejected it.** The most valuable kind: it stops the
  next person redoing the analysis, or "fixing" the code back to the version that was wrong.
- **A hardware, driver or specification constraint** making otherwise-odd code necessary.
- **A complexity or cost** that is not obvious from the shape of the loop.
- **A mathematical derivation** — where a constant or a bound came from.

**Never appears** — these are worse than nothing, because they add reading cost and then drift:

```
i++;                    // increment i
return result;          // return the value
for (...)               // loop through the elements
```

Also never: commented-out code (§5); decorative banners; an ownerless `TODO` (P7); anything that
will be false after the next edit — cite a source rather than copy a value.

**The standing test:** could a contributor who has never spoken to you read this and make a
correct change?

---

# Part E — Verifying

## 10. Testing

Tests ship with the change. A feature without tests is unfinished.

**Test behaviour, not implementation.** Exercise the public contract, not internals. A test
coupled to internals breaks on every refactor, and a suite that cries wolf stops being read. If
behaviour is hard to test without reaching inside, the design is telling you something (§5).

| Tier | Covers |
|---|---|
| **Unit** | Every operation: every dtype, contiguous + strided + broadcast, edge shapes (empty, size-1, non-power-of-2, minimum and maximum rank) |
| **Oracle** | The chain in §1.4, in order |
| **Gradient** | Analytical against numerical differentiation *and* against the reference framework |
| **Integration** | Layers, optimisers over many steps (trajectories, not endpoints), whole-model parity |
| **Regression** | Golden values, exact because results are deterministic. Every fixed bug gets one |
| **Property** | Randomised shapes, strides, dtypes — where hand-written cases miss combinations |
| **Resource** | Every allocation freed; peak within prediction; no device loss under stress |

- Run correctness tests in the mode that makes a failure point at the offending operation rather
  than a deferred execution boundary.
- Run device tests with validation layers on; any validation error fails the build.
- **A test that cannot fail is worse than none** — it manufactures confidence. Verify a new test
  fails against deliberately broken code before trusting its pass.
- Tolerances come from the central policy and nowhere else (§1.3).
- Cover what a random sweep cannot reach: NaN and infinity, exact equality, empty inputs,
  contention, numerically awkward tails.

## 11. Debugging

**Never guess.** A fix applied without understanding the cause moves the symptom and leaves a
change nobody can justify.

```
Reproduce  →  Reduce  →  Observe  →  Hypothesise  →  Verify  →  Fix  →  Regression test
```

1. **Reproduce** deterministically, with a fixed seed and a recorded command. An intermittent
   failure is a reproduction problem to solve first, not a bug to guess at.
2. **Reduce** to the smallest input and shortest path that still fails. Most bugs become obvious
   at minimum size, and the reduced case is the regression test.
3. **Observe** rather than infer. Print the actual values, dump the intermediate, run the
   sanitizer, enable the validation layer. What you believe the code does is the thing under
   suspicion.
4. **Hypothesise** one specific cause that explains *all* the evidence — including anything that
   looked irrelevant. A hypothesis explaining only some of it is wrong.
5. **Verify** by prediction: state what you expect to see if the hypothesis holds, then look.
   Cheapest discriminating test first.
6. **Fix the cause.** If the fix works and you cannot say why, you have not finished.
7. **Regression test**, from the reduced case. Confirm it fails against the unfixed code (§10).

**Before assuming an implementation bug, rule out two things** that have wasted more time here
than real defects: **a wrong measurement** (§13) and **a wrong test**. Both look exactly like a
code bug from the outside.

**Bisect when the cause is not localised** — which is most of why every commit must build and
pass (§17). Record what a hard bug taught you, as a note in `docs/` or a comment at the trap.

## 12. When *not* to optimise

Optimisation has a permanent cost — complexity, another path to test, code that resists change —
paid whether or not the speedup materialises.

**Not a reason to optimise:** it feels slow · it looks inefficient · another project does it ·
it saves allocations in code that runs once · the assembly could be tighter · a microbenchmark of
the function in isolation improved · it seems wasteful.

**A reason to optimise:** a profile of a real workload names it · a benchmark on a representative
input shows the gain · a measured regression against a recorded baseline · a user-visible
bottleneck · a resource limit actually being hit.

**Gate zero, before profiling anything: is the functionality this serves complete?** (P2.) A
missing operation is infinitely slower than a suboptimal one. If it is not complete, implement
what is missing and record the opportunity (P7).

## 13. Performance engineering

**No optimisation is accepted without evidence** (P5).

### The loop

1. **Profile.** Find the actual bottleneck; intuition about which line is hot is wrong more often
   than right.
2. **Hypothesise.** State what you expect to change, by how much, and **what result would prove
   you wrong** — before measuring. A prediction made afterwards is not one.
3. **Check the constraints.** Does it alter fold order (§1.3)? Do recorded laws forbid it?
   Cheapest refutation first.
4. **Gate on resources before timing.** Where the toolchain reports register pressure, spills,
   scratch or occupancy for a candidate, read them and reject on those *before* benchmarking — a
   candidate rejected without ever being timed is the cheapest rejection available.
5. **Measure**, to the rules below.
6. **Verify correctness.** Full suite plus goldens.
7. **Accept or roll back.** A failed hypothesis is a result: record it (P7). Failed predictions
   are where laws come from.

### Measurement rules

Derivations and measured figures are in `MEASUREMENT-AUDIT.md`.

1. **Characterise the noise floor first.** An effect smaller than the spread of repeated
   identical runs is not an effect.
2. **Device-side timers beat wall clock.** Use wall clock only when the measured operation
   dominates the window — check that ratio rather than assume it.
3. **Report the minimum**, never the mean. The tail measures the machine.
4. **Never sum per-dispatch timers across concurrent work** — overlapping work reports
   overlapping intervals and the sum counts the same time repeatedly. Use an enclosing window.
5. **Never compare a profiled run against an unprofiled one.**
6. **Never benchmark with validation or debug layers on.** They change what they measure.
7. **Warm caches and pipelines first.** Compilation is setup, not measurement.
8. **An A/B is valid only if the A arm is the frozen, unmodified baseline.**
9. **Prefer criteria that compare bytes** — they cannot be perturbed by measuring.
10. **Check every gate for vacuity before trusting a pass.**
11. **Two independent calculations agreeing is not confirmation** — it warns that the experiment
    cannot distinguish them.

Benchmark in the configuration intended for it. A stored baseline is a claim: updating one needs
the same evidence as any other performance statement.

---

# Part F — Shipping

## 14. Refactoring

**Before adding a feature, decide whether what exists can carry it.** Usually it can — say so in
a line and move on. Escalate when: the feature needs a special case inside an existing
abstraction; a parameter exists only to select behaviour for one caller; the change forces an
edit in a layer that should not have known about it; the same conditional appears in a third
place.

**When a signal fires, fix the abstraction first, in its own commit.** Building on a known-bad
abstraction is never cheaper later.

**When *not* to refactor**, which matters as much: the code is ugly but stable, well-tested,
single-caller, with no feature pending · you are mid-feature and it is unrelated (note it, finish,
do it separately) · you cannot state the improvement in one sentence · behaviour would change,
which is a redesign needing §2 step 6.

**Refactoring preserves behaviour, and here that is checkable**: a change altering no fold order
must leave goldens byte-identical (§1.3). If they move, you did not refactor. Small steps, suite
green between each.

## 15. Documentation

Comment philosophy is §10. This is the artifacts, and where each kind of knowledge belongs:

```
code            what it does now — the only thing that cannot be out of date
comments        why this line is as it is, and what was rejected
API docs        the contract: parameters, returns, throws, lifetimes
architecture    how the pieces fit and why the shape is this shape
ADRs            one decision, its alternatives, its measurements, its guardrails
manifesto       mission, priorities, what is deliberately not being done
```

Put knowledge at the **narrowest** level that holds it: a reason affecting one line is a comment,
not an ADR; a decision constraining the project is an ADR, not a comment.

- **Public API**: purpose, parameters, return, what it throws, any lifetime or ownership
  requirement.
- **ADR format**: *Context → Measurements → Options considered, each with a verdict → Decision →
  Guardrails adopted now → Rejected alternatives, recorded so they are not rediscovered.*
- **Experiments get a record** stating hypothesis, quantified prediction and falsification
  criteria *before* the results.
- **Update what your change invalidated**, in the same commit. Stale documentation is worse than
  none, because it is trusted.
- Examples build and run in CI, or they rot.

## 16. Review checklist

Against your own work before declaring anything complete. Mechanical always; the rest as
applicable. Each line cites where the rule lives rather than restating it.

**Mechanical** — no judgement, just run them: layering, formatter, static analysis, a
warning-free build in every configuration, the full suite, sanitizers for memory-touching changes.

- [ ] **Always** — every claim supported by something I ran (P5, constitution §7) · one logical unit · anything
      unfinished stated (P7) · surrounding code no worse than found · nothing added that could
      have been deleted (§5)
- [ ] **Design** (§4–7) — correct layer · one reason to change · no new coupling or cycle · no
      backend branch in an operator (§2.3) · generic before specialised (P6) · no smell from §5 ·
      complexity suits the real input sizes (§4)
- [ ] **Correctness** — edge cases (empty, min rank, size-1, broadcast, non-contiguous, aliased) ·
      overflow · error paths tested, not just the happy path · outside input validated (§19)
- [ ] **C++** (§7) — ownership in types · Rule of Zero or all five · RAII · moves `noexcept` ·
      `const` correct · no C-style cast, no implicit narrowing · no macro a function could be ·
      standard library preferred · headers self-contained · no UB
- [ ] **Exception safety** (§8) — no-throw where required · strong where shared state is mutated ·
      no leak on any throw path
- [ ] **Numerics** (§1.3) — fold order unchanged, or bound re-derived and goldens re-pinned ·
      tolerances from the policy · determinism preserved
- [ ] **Performance** (§12–14) — justified by §12, backed by measurement · no needless allocation
      on hot paths · no regression
- [ ] **Testing** (§10) — new tests can actually fail · behaviour not internals · regression test
      for every fixed bug · oracle chain followed
- [ ] **Docs & naming** (§9, §15) — public API documented · decisions recorded at the right level
      · invalidated docs updated · names state what things are · comments explain *why* · no
      commented-out code, no ownerless TODO

The question that decides it: **could a contributor who has never spoken to you read this and
make a correct change?**

## 17. Git workflow

- **One logical change per commit.** A refactor and a feature are two commits. A commit doing two
  things cannot be reverted, reviewed or bisected cleanly.
- **Every commit builds and passes tests.** Broken intermediate commits destroy bisection (§11),
  which is most of why history is worth keeping.
- **Messages explain *why*.** Imperative subject, ≤72 characters, no trailing period; blank line;
  body covering what changed and the reasoning — the alternative considered, the measurement that
  justified it, the constraint that forced it. Reference documents by path. Understandable
  without opening the diff.
  Good: `Add tensor broadcasting support` · `Fix gradient accumulation bug`.
  Never: `fix` · `update` · `changes` · `working` · `final` · `misc`.
- **Never commit broken code** — failing to compile, breaking tests, a known regression, or
  incomplete functionality not clearly isolated.
- **History is engineering documentation.** A reader should see how the project evolved:
  decisions, milestones, refactors, optimisation phases.
- **No artificial history.** Never fabricate commits for states that never existed.
- **No AI attribution.** No `Co-Authored-By` for tools, no generated-by trailers.
- **Keep history linear.** Squash noisy commits before merging; rewrite only unpushed work.
- **One objective per pull request.**
- Never commit build output, environments, generated artifacts or vendored study material.
- Propose commits as a list with messages; let the author decide when to run them.

## 18. Build and reproducibility

- **Modern, target-based CMake.** Everything attaches to a target: `target_link_libraries`,
  `target_include_directories`, `target_compile_features`, `target_compile_definitions`.
- **No directory-scoped or global commands** — `include_directories`, `add_definitions`, mutating
  `CMAKE_CXX_FLAGS`. They leak into every target and make a build impossible to reason about
  locally.
- **Correct visibility**: `PRIVATE` for implementation, `PUBLIC` for what consumers need,
  `INTERFACE` for header-only. Over-broad visibility propagates dependencies nobody asked for.
- **Named presets for every configuration** anyone builds, so CI runs what a developer runs.
- **Generated sources are build steps with declared dependencies**, including dependency files —
  an edit to a shared include must trigger the rebuilds that depend on it.
- Warnings-as-errors available everywhere, on in CI.

**Reproducibility** — a build that cannot be reproduced cannot be debugged from a report.

- **Pin what you can**: dependency versions, and toolchain versions in CI.
- **No hidden local dependencies.** If it only builds on the author's machine, it is broken.
- **Nothing depends on wall-clock time, absolute paths, or leftover environment.** Embedded
  timestamps and build paths defeat binary comparison.
- **A bug report should be answerable from a commit hash plus a preset name.**

**Dependency admission is a decision, not a convenience.** A new dependency needs a reason, a
compatible licence, and an owner. Prefer the standard library (§7.8), then a small well-maintained
library, then vendoring. Study material that is read but never linked is kept clearly separate
from code that ships.

## 19. Security

Modest but not optional — a framework that loads model files loads *other people's* files.

- **Validate everything crossing a trust boundary**: file contents, tensor shapes and strides,
  indices, sizes, counts. Never trust a value because your own writer produced it.
- **Never deserialise into code execution.** A model format that can invoke arbitrary code on load
  is a remote-code-execution vector — the well-known failure in this field. Formats are data;
  loading is parsing.
- **Check sizes and bounds before allocating or indexing**, including that a declared size matches
  the bytes actually present. A length field from a file is an attacker-controlled integer.
- **Integer overflow in size arithmetic is a memory-safety bug.** Multiplying dimensions to get an
  allocation size overflows silently; check before allocating, not after.
- **Fail closed.** A malformed input produces a clear error, never a partially-initialised object
  the caller can use.
- **Do not put secrets or absolute paths in artifacts** — logs, error messages, embedded strings.

## 20. Learning from prior art

Before proposing a significant architectural change or optimisation, find out how mature projects
solved the same problem — **and why**.

1. Read for **intent**, not implementation. Ask what constraint produced the shape.
2. **Convergence is evidence about the problem.** A choice made independently by several teams
   with different languages and hardware says something about the problem, not the teams. Where
   they *disagree*, the question is genuinely open.
3. **Apply the filter.** Does the motivating constraint hold here? A technique trading accuracy
   for speed may be right for an inference engine and incompatible with §1.3 — rejected outright,
   however fast.
4. **Never copy code.** Understand the idea, implement it for this project's constraints. Copying
   also imports licence obligations.
5. **Record provenance**: source, why it exists there, why it applies here.

Study material in the repository is read-only, pinned, never linked, never modified. Do not
assume the reference projects are ahead on everything.

---

## Quick reference

Exact commands and preset names live in the build configuration and CI definition — read those.
The shape of the loop:

```
configure + build (debug)      →  develop
run the full test suite        →  correctness
layering + format + tidy       →  mechanical gates
sanitizer build + suite        →  memory safety
release-with-debug-info        →  benchmarking only
eager / per-op mode            →  make a failure point at the operation
```

**If you remember nothing else:** read before writing · fix the abstraction before building on it
· generic before specialised · choose the algorithm before the constant · never break determinism
· never guess when debugging · measure before claiming · test behaviour, not internals · delete
what is not earning its place · name things so the next reader needs no explanation · comment the
*why* · say what you actually did.
