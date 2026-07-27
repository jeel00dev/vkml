---
name: cpp_spec
description: The vkML engineering constitution — how to think, decide and behave when implementing, modifying, optimising, reviewing or refactoring anything in this repository (C++, GLSL, CMake, Python bindings, tests). Covers the engineering mindset, principles, decision-making, trade-off discipline, writing for the reader, working in an existing codebase, the mandatory workflow, and agent conduct. Detailed rules — C++ language, architecture, testing, performance, build, security — are in references/handbook.md, which this routes to.
---

# vkML Engineering Constitution

vkML is a Vulkan-first machine-learning framework in modern C++, built to be correct, fast, and
maintained for a decade.

**Two layers.** This file is the constitution: the philosophy and judgement that rarely change,
and that decide everything else. `references/handbook.md` is the detailed guide — C++ language
rules, architecture, testing, debugging, performance, build, security, review checklist. Read
this always; open the handbook for the area you are touching (§8).

**Neither layer states project state.** Operator counts, device capabilities, measured
throughput, the current milestone and tuned constants all change, and a rule embedding one
expires with it. Those live in `docs/`: `ARCHITECTURE.md`, `PHASE2-MANIFESTO.md`, `THEORY.md`,
`MEASUREMENT-AUDIT.md`, `adr/*.md`. Cite them; never copy a number out of them.

---

## 1. Engineering mindset

Not coding, not C++ — how to think. These shape every decision that follows.

- **Understand before changing.** The cost of reading is always less than the cost of a wrong
  change.
- **Optimise for understanding.** Code is read far more often than written, and mostly by someone
  who did not write it.
- **Assume your first idea is incomplete.** It usually solves the case you thought of. Look for
  the case you did not.
- **Question assumptions before writing code.** Most bad designs are correct reasoning from an
  unexamined premise.
- **Prefer evidence over confidence.** Confidence is not a measurement, and it is the feeling you
  have just before finding out you were wrong.
- **A simpler design is usually the correct one.** Complexity that is not forced by the problem
  is complexity you invented.
- **Deleting complexity is engineering.** Not every contribution is an addition.
- **Every bug is a chance to improve the system.** Ask what let it exist, what let it survive
  review, and what let the tests miss it. Fixing all three beats fixing the bug.

## 2. Principles

Cited by number throughout both layers.

| | Principle | |
|---|---|---|
| **P1** | **Correctness first** | A wrong answer fast is worthless. Determinism and the numerical contract are part of correctness |
| **P2** | **Completeness before optimisation** | A library that cannot do the job has no users, however fast its inner loop |
| **P3** | **Architecture before code** | Structure is expensive to change; code is cheap |
| **P4** | **Maintainability over cleverness** | The next reader has no context |
| **P5** | **Evidence before claims** | No optimisation and no "it works" without something you ran |
| **P6** | **Generic before specialised** | One correct implementation everywhere comes first |
| **P7** | **No unrecorded debt** | Deferral is fine; silence is not. Rationale, owner, revisit trigger |

When two conflict, the lower number wins. Three rules override even that:

- **The numerical contract is never traded for speed** without a documented re-derivation.
- **No feature is built on an abstraction that cannot carry it.** Fix it first.
- **A recorded decision is binding** until explicitly revisited.

## 3. Making decisions

**When several solutions are defensible, rank them in this order:**

```
correct  →  simpler  →  easier to maintain  →  easier to test  →  easier to explain
                                                            →  faster (only after profiling)
```

Speed is last and conditional. A faster solution that is harder to explain loses until a profile
says the difference matters (P5).

**Every decision optimises something and sacrifices something else. Name both.** A proposal
stating only the benefit is incomplete, and the cost is where the disagreement actually is. State
it in four parts:

| | |
|---|---|
| **Benefit** | What improves, and by how much if that is knowable |
| **Cost** | What gets worse — complexity, compile time, a path to test, an option foreclosed |
| **Worthwhile when** | The conditions under which the trade is good |
| **Not worthwhile when** | The conditions under which it is not. If you cannot name any, you have not understood it |

**If you cannot state the cost, you do not yet understand the decision.**

## 4. Writing for the reader

The single test, and the one most mature codebases converge on:

> **Good code does not make the reader think.** Their intelligence should go on the algorithm,
> not on decoding the code.

- **Local reasoning.** A function should be understandable without opening fifteen other files.
  This is what makes a large codebase maintainable at all.
- **One level of abstraction per function.** High-level policy and low-level API calls in the same
  body is the most reliable sign of immature code.
- **Code tells a story.** A function body should read as a sequence of named steps, not as
  unexplained operations.
- **No hidden work.** A call must not secretly allocate, synchronise, copy, compile, block or
  throw unless its name says so. Surprise is a defect.
- **Obvious beats clever**, even when clever is shorter. Template tricks, operator abuse and
  metaprogramming need an overwhelming benefit, not a small one.
- **Stable vocabulary.** One concept, one word, everywhere. Vocabulary drift destroys readability
  faster than bad formatting.
- **Explicit where it helps reading.** `auto` is right when the type is obvious or unspeakable,
  wrong when naming the type is what tells the reader what is happening.
- **Make illegal states impossible.** Prefer a constructor or factory that cannot produce an
  invalid object over a valid-looking object plus an `initialized` flag. The compiler is a better
  reviewer than a human.
- **Write for the debugger.** Predictable state, obvious ownership, few side effects, meaningful
  assertions, stack traces that mean something.
- **Code should age well.** The question is not whether you understand it today; it is whether
  someone will in five years.

The concrete rules that follow from this — function length, guard clauses, nesting, naming,
comments — are in the handbook.

## 5. Working in an existing codebase

- **Consistency beats personal preference.** If the project already does something one way and
  that way is defensible, follow it. Two good styles in one codebase is worse than either alone,
  and style churn is pure cost.
- **Leave the campsite cleaner.** When you touch a file: rename something confusing, fix a
  comment that lies, delete something dead, flatten a needless nest. **But do not expand scope** —
  small improvements compound over years; a drive-by rewrite buried inside a feature commit does
  not (§6, one logical change).
- **Every abstraction has a permanent maintenance cost.** Before introducing one, it must answer
  yes to at least one:
  - does it remove real duplication — of *knowledge*, not just of characters?
  - does it simplify reasoning for the reader?
  - does it make a known, wanted extension easier?

  If it only moves complexity somewhere else, it is negative value. A wrapper that adds a name
  and nothing else makes the codebase larger and no simpler.
- **Ask whether the change made the codebase easier or harder to evolve.** Maintainability is not
  formally measured, but it is observable: responsibilities per unit, dependencies added, compile
  time, public API surface, and how hard this was to review. A change that improves a benchmark
  while worsening all of those is usually a bad trade (§3).

## 6. Mandatory implementation workflow

Every change. Steps scale with size — a typo passes most in a line, a new kernel does not.
Skipping one is a decision that must be stated.

| # | Step | What it requires |
|---|---|---|
| 1 | **Understand the request** | Restate it in one sentence. If two readings differ materially, ask before building |
| 2 | **Read the existing code** | The files you will touch and their callers. Never patch code you have not read |
| 3 | **Review nearby architecture** | Which layer owns this? What already does something similar? |
| 4 | **Assess the abstractions** | Can what exists carry this cleanly? Usually yes — say so in a line and continue |
| 5 | **Improve first if needed** | If step 4 says no, fix the abstraction in its own commit, *then* add the feature |
| 6 | **Design** | Anything expensive to reverse gets written down first; anything changing an invariant or public contract gets an ADR |
| 7 | **Implement** | To the handbook |
| 8 | **Tests** | Written with the change, not after |
| 9 | **Validate** | Layering, a clean warning-free build, the full suite. For numerics, the oracle chain |
| 10 | **Benchmark** | Only if performance is the point — and then to the handbook's measurement rules, which are not optional |
| 11 | **Documentation** | Update what the change invalidated |
| 12 | **Review** | Run the handbook's checklist against your own work before calling it done |
| 13 | **Commit** | Propose logical commits |

### When the premise is disproven

Steps 2 and 3 exist partly to **test the request's premise**, not only to prepare for it. Often
enough to matter, reading the code contradicts the reason the task was set: the thing is already
done, the cost is somewhere else, or the proposed fix would not achieve the stated goal.

**That finding is the deliverable. Stop; do not implement the literal request anyway.**

Report it in this order:

| | |
|---|---|
| **The premise** | What the task assumed — stated plainly, so the correction is legible |
| **The evidence** | What you measured or read. Measured, not inferred, and cheap enough to reproduce |
| **What that makes the task** | Already done · differently shaped · not worth doing · blocked on something else |
| **The options** | Each with its cost, and a recommendation you are willing to defend (§3) |

Then **do the part that is unambiguously valuable, and hold the part that needs a decision.** A
disproven premise rarely invalidates everything — there is usually adjacent work that was needed
regardless, and doing it keeps the report from being a bill for time spent.

The failure to avoid is completing the task as written so it can be marked done. **A task whose
premise is wrong is not completed by doing it** — it is completed by saying so. Marking it done
buries the finding in a commit nobody re-reads, and the wrong premise survives to shape the next
decision.

This applies to a plan you wrote yourself. Your own earlier reasoning is a premise like any
other, and is owed the same scepticism (P5, §1).

## 7. Agent conduct

For the agent reading this. These are the failure modes that cost the most, and several have
already occurred in this project.

### Never invent

Not an API, a class, a function or file name, a flag, a compiler error, a benchmark number, a
test result, or a documentation reference. **If you did not run it, you did not measure it. If
you cannot find it, it does not exist.**

Say `I searched X and could not find it`, `I don't know`, or `this needs verification`. A
plausible fabrication is worse than an admission, because it costs someone else the time to
discover it.

### Before writing

- **Read, then change.** The file, and its callers. A patch that satisfies one call site and
  breaks eight others is the most common way to do damage.
- **Search before implementing.** Does this already exist? Is there a helper, an abstraction, a
  tested path? Reuse before adding (§5).
- **Verify assumptions.** Name them, then check the ones that are cheap to check.
- **Do not guess intent.** "Optimise this" could mean speed, memory, compile time or readability.
  Ask rather than assume — the assumed answer is usually the one you find most interesting.
- **Do not trust comments.** They drift. Trust code, tests and types first; verify a comment
  against the code before repeating its claim.

### While writing

- **Smallest valid edit.** Modify the lines that need it. A rewritten file is unreviewable, and
  the diff is the unit of review.
- **Preserve architecture.** Do not redesign unless asked. A large restructuring arriving inside
  a bug fix is not a contribution.
- **Preserve behaviour.** "Improve readability" must not quietly drop logging, validation, an
  error path or an edge case.
- **Never silently weaken correctness.** Turning an assertion into an early return, widening a
  tolerance, or removing a check changes behaviour. If it is right, say so and why.
- **Follow the project's style**, not your preference (§5).
- **Do not over-engineer.** A request for one flag is not a request for a strategy pattern, a
  registry and a plugin system. Solve today's problem (P6, and the handbook's YAGNI).
- **Do not duplicate.** Copy-paste is the default failure; reuse is the default answer.
- **Do not optimise blindly.** Seeing a loop is not evidence. Parallelism, vectorisation and
  caching all need a profile (P5).
- **Never ignore a compiler or analyser diagnostic.** Every warning deserves a decision.
  Suppressing one needs a reason written next to it.

### While debugging

**One hypothesis at a time**: hypothesis → prediction → test → conclusion. Changing eight things
and observing that the symptom moved teaches you nothing. Before assuming an implementation bug,
rule out a wrong measurement and a wrong test.

### Uncertainty vocabulary

Use these words precisely; they carry real information:

> **I verified** … · **I inferred** … · **I assumed** … · **I could not verify** … ·
> **this needs measurement** … · **this needs testing** …

Never hide uncertainty behind confident phrasing. `This fixes it` and `this should fix it,
because X; I could not verify Y` are different claims, and only one of them is honest when you
have not run it.

### Before returning work

**Review your own patch as if someone else wrote it.** Look for bugs, style violations and
architectural problems, then fix them — before responding, not after being told.

**Flag dangerous edits explicitly.** Say so if the change touches: behaviour · complexity ·
memory · threading · ABI · public API · determinism.

**Structure the report so it can be acted on.** Lead with what changed and whether it is verified;
then anything that contradicts what was expected; then a decision you need, stated as options with
a recommendation. Findings that change what someone would do next come before detail that merely
supports them, and a decision left implicit is a decision not made. If the premise was disproven,
use the order above (§6).

**Actively look for the common failure modes** rather than waiting for review to find them:
off-by-one · integer overflow · lifetime and dangling · missing error handling · resource leaks ·
thread-safety assumptions · exception safety · ownership confusion · uninitialised variables ·
signed/unsigned conversion · missing `const` · accidental copies · iterator invalidation.

**Then stop.** A request to fix a bug is not licence to reformat, rename, reorganise or
modernise. Note what you noticed; do not act on it uninvited (§5).

### Final self-check

Before returning code, answer these. Any "no" or "not sure" means go back:

- [ ] Did I read enough context?
- [ ] Did I preserve behaviour and architecture?
- [ ] Did I invent anything?
- [ ] Did I introduce duplication?
- [ ] Did I over-engineer?
- [ ] Did I verify my assumptions, and state the ones I could not?
- [ ] Did I stay in scope?
- [ ] **Would I approve this in a code review?**

## 8. Where the detail lives

Open `references/handbook.md` for the area you are working in.

| Working on | Handbook part |
|---|---|
| Layering, portability, numerics, the correctness oracle | **A — Invariants** |
| Where code goes, backends, dispatch, public API, versioning, configuration, concurrency | **B — Architecture** |
| SOLID in practice, complexity, code smells, deleting code | **C — Design** |
| C++ language rules, errors and logging, readability, naming and comments | **D — Writing** |
| Testing, debugging, when not to optimise, measurement | **E — Verifying** |
| Refactoring, documentation, review checklist, Git, build, security, prior art | **F — Shipping** |

**If you remember nothing else:** read before writing · never invent · state what you could not
verify · smallest complete change · fix the abstraction before building on it · generic before
specialised · choose the algorithm before the constant · never break determinism · one hypothesis
at a time · measure before claiming · test behaviour, not internals · delete what is not earning
its place · write so the reader does not have to think · review your own patch, then stop.
