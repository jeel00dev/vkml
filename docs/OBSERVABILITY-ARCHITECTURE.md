# Observability architecture — design for review (revision 2)

**Status: approved; increments 1-3 implemented.** To become an ADR when O1 and O2 land.

| Increment | State |
|---|---|
| 1 · fact model (`util/observe.h`) | **done** |
| 2 · decision publication at the GEMM fallback | **done** — the site's one-shot flag and inline logging deleted |
| 3 · recorder + Python query surface (`util/decisions.h`) | **done** — `record_decisions()`, `decisions()`, `decisions_published()`; `vkvalidate.py`'s category-3 threshold deleted |
| 4 · cross-owner validation | **done** — `tests/python/test_decision_facts.py`; one signal implemented, one found not to exist yet |
| 5 · observation-cost measured | **done** — `bench_observation_cost` in `bench/cpp`; gate is remaining work, tracker #117 |
| O1 · configuration snapshot | tracker #96 |
| O2 · allocator decision site | tracker #97 |

This exists because an audit found that observability has never been planned
work in this project. All four observability tasks were discovered reactively;
`planned + Observability` is an empty cell in a tracker of 98 tasks. Correctness
and verification each have an architecture. This is the case that observability
needs one too, and that three separate ones would be worse than none.

**Revision 2** separates the fact model from the surfaces that render it, adds
the ownership rule, adds users as a consumer, restates the cost constraint as a
measurable property, and requires every fact to be checkable against reality.

---

## 1. The evidence, measured

| Fact | Measured |
|---|---|
| Runtime switches that change behaviour | **18** |
| ...observable in a running process | **0** |
| ...documented (`docs` + site) | 18 of 18 — the documentation is complete |
| Cost of one unreported switch | `VKML_GEMM_NOVEC=1` → **+14.6%** on 1024³ matmul, three processes per arm, non-overlapping |
| Kernel-name string literals, no registry | **18** distinct, in `pipes.get("…")` |
| Files independently referencing `max_workgroup_invocations` | **14** |
| Capability-dependent branches | 9 — **7 are `VKML_CHECK` and throw with the numbers** |
| One-shot "report once then forget" flags | **1** (`reported_gemm_fallback`) — since deleted; dedup moved to the consumer |

The last two rows are why this document is narrow. Observability here is not
broadly absent — most capability branches fail loudly and well, and the
documentation of the switches is complete and accurate. **The gap is that a
running process cannot describe its own configuration or explain its own
choices, even though both are fully known to it.**

### What that costs, in incidents already paid for

- **#76.** A 38.6% "regression" that was GPU clock state. An exported switch
  produces the identical signature: stable across processes, invisible to every
  instrument, indistinguishable from a code change.
- **#71.** `bench/baselines/rx5600m.json` records commit, driver, `recorded_at`,
  `warmed` — and no switches. Principle 4 says a baseline without its conditions
  is not a baseline; the one condition that is invisible and worth 15% is missing.
- **N1.** `requires_register_kernel` in `tests/python/vkvalidate.py` cannot ask
  the backend whether it fell back, so it reimplements the selection rule as
  `max_workgroup_invocations < 256` — **category 3 inside the test suite**,
  written by someone who had just authored the principle forbidding it.

---

## 2. Source of truth: the decision sites

**The thing making the decision is the thing describing the decision.**

This is a deliberate rejection of a subsystem that describes the engine from
outside. Such a subsystem is a second model of vkML — hand-maintained, drifting
silently, authoritative-looking throughout: category 3 wearing an architecture
diagram.

So nothing here *describes* the system. Instead:

```
    env_flag("VKML_GEMM_NOVEC", false)      →  a registered switch
    pipes.get("gemm_reg", …)                →  a named kernel from a table
    if (!reg_fits) use_naive = true;        →  a recorded decision
```

Each site keeps doing exactly what it does, and additionally becomes enumerable.
The registry is *populated by* the sites, so it cannot disagree with them: the
failure mode is a compile error, not a stale document. That makes observability
**category 1, generated from the source of truth**.

---

# PART I — THE ARCHITECTURE

The architecture is the fact model and its rules. It ends at §5. Everything
after that is consumption and presentation, and can change without changing this.

## 3. The four fact types

| Fact | Question | Lifetime |
|---|---|---|
| **Configuration** | what was I asked to do? | fixed at startup |
| **Capability** | what can this device do? | fixed after device init |
| **Decision** | what did I choose, and instead of what? | per operation |
| **Measurement** | what did it cost? | per operation |

Measurement exists and is good. Capability exists, with a known gap: 3 of 5
min-spec-clamped limits are not exposed, which `check_min_spec.py` reports every
run. **Decision now exists** (`util/observe.h`, recorded by `util/decisions.h`).
**Configuration does not yet** — tracker #96.

A Decision carries: the site, what was chosen, the alternatives rejected, the
reason, and the numbers that forced it. *"The register-blocked kernel needs 256
invocations and the device allows 128"* is already exactly this — it is merely
formatted as English and thrown at stderr.

### What a subsystem must expose

Not everything it knows. **Any fact that changes what executes and is not
derivable from the inputs.** A tensor's shape is derivable; the kernel chosen
for it is not.

## 4. Ownership — exactly one authoritative owner per fact

| Fact | Owner | Everyone else |
|---|---|---|
| Configuration | the configuration subsystem (`util/env`) | consumes |
| Capability | device discovery (`vk_device`) | consumes |
| Decision | the planner/backend site that makes it | consumes |
| Measurement | the profiler / benchmark harness | consumes |

**No fact may have two authoritative owners.** This is the rule that stops the
architecture growing sideways, and it is not abstract — it names precisely what
is wrong today:

> `vkvalidate.py` read `max_workgroup_invocations` (a Capability, owned by
> device discovery) and from it *derived a Decision* — whether the backend fell
> back. It was not consuming a fact; it had become a **second owner of a fact
> the backend already owns**, and the two could disagree silently.

That is the diagnosis the first revision could not state. The fix was not "expose
more capability"; it was that Decision has one owner and the test consumes it.
**Done:** `_matmul_falls_back_to_naive()` now runs a matmul and reads what the
backend published. Verified by silencing the publication, at which point the
marker stops skipping and 13 tests fail — the threshold version would have kept
skipping and said nothing.

Corollary for the 14 files referencing `max_workgroup_invocations`: consuming a
capability is fine and expected. *Deriving a decision from it* is the violation.

## 4a. The implementation constraint: no subsystem knows the recorder exists

The fact model is the architectural centre. Everything else hangs off it, and
nothing hangs off anything else.

```
    Decision site
        ↓  publishes
      Fact                      ← the architecture is here
        ↓  consumed by
     Recorder
        ↓  rendered as
    snapshot · log · CI artifact · diagnostics · user queries
```

**Not** this:

```
    Decision site → Recorder → everything else
```

A decision site publishes a fact and learns nothing in return. It must not know
about storage, buffering, rendering, transport, logging, or any consumer — not
even whether anyone is listening. If `vulkan_backend.cpp` ends up including a
recorder header, taking a recorder handle, or branching on whether recording is
enabled, the architecture has been inverted and the coupling will spread along
exactly the paths that made the log load-bearing in the first place.

The precedent is already in this codebase: `util/log.h` decision sites call
`VKML_LOG_INFO` and know nothing about sinks, and `callback_storage()` lets a
consumer install itself without any caller changing. Publication follows that
shape.

Practical test of the constraint: **it must be possible to delete the recorder
entirely and still compile.** Publication is a call into layer 0 that may go
nowhere.

## 5. Every fact must be checkable against reality

The highest-leverage systems in this project all became bidirectional —
documentation ↔ source, gates ↔ controls, baselines ↔ conditions. Observability
adopts the same property, or it becomes the next thing that drifts.

For each fact type: **can we verify it still corresponds to reality?** If yes,
automate it. If no, say so.

| Fact | Verifiable? | How |
|---|---|---|
| Configuration | **yes, automate** | every `VKML_*` literal in `src/` must resolve to a registered switch; a gate compares the two sets, in both directions |
| Capability | **partly, today** | `check_min_spec.py` cross-checks 2 of 5 clamps against the C++ that applies them. The other 3 are not exposed — recorded as a gap it prints every run, not hidden |
| Decision | **yes, in part — see below** | falsifiable against facts the *driver* reports, not against vkML's own bookkeeping |
| Measurement | **yes, exists** | warm-up, noise floor, `NOISE_SIGMA`, frozen baselines |

### The Decision check is only partly independent, and the boundary matters

§13 of revision 1 flagged the risk that this check could be circular. Checked
before relying on it, and **it is circular in one half**:

- **Pipeline identity is NOT an independent check.** Whether `gemm_reg` appears
  in `vulkan_pipeline_stats` is decided by vkML calling `pipes.get("gemm_reg")`
  — the same call a decision record would report. Both derive from one code
  path, so agreement proves nothing. A recorder that faithfully echoes
  `pipes.get` cannot be caught here, and at that point the two are one fact.

- **Pipeline properties ARE independent.** `query_stats` calls
  `vkGetPipelineExecutablePropertiesKHR` / `…StatisticsKHR`: the numbers come
  from *the driver's* view of the compiled pipeline. If a decision claims the
  register-blocked kernel at 256 invocations with RM=RN=2, the driver's VGPR
  count, LDS bytes and scratch bytes are derived from the compiled SPIR-V and
  can contradict it.

- **Dispatch structure is independent.** If a decision claims split-K with 8
  partitions, `vulkan_last_profile` must show 8 dispatches. That is the
  profiler's own count, and `test_submit_window_bounds_concurrent_dispatches`
  already relies on exactly this signal.

**What actually shipped**, after measuring which producers exist: a decision's
`chose` must name something the driver successfully compiled and reported
statistics for, and its `instead_of` must name something the driver did *not*
compile. The independence comes from the driver attesting the compilation —
`available` and the register counts are the shader compiler's, not vkML's — and
from the dispatch path fetching its pipeline ~140 lines after, and separately
from, the site that published the claim.

Both directions are proven to fail: making the decision name the wrong kernel,
and making it name an alternative that was in fact compiled, each turn the check
red. A third proposed check was dropped because its producer does not exist yet
(above), which is the honest outcome rather than a check kept for comfort.

This works *because* of the ownership rule — but only where the owners are
genuinely different. "Different accessor" is not "different owner".

---

# PART II — CONSUMPTION AND PRESENTATION

Implementation detail. New surfaces may be added here without touching Part I.

## 6. Presentation surfaces

Renderings of the facts above. **None is a source.** Today's design fails
exactly here: a decision exists *only* in the log, so the log is load-bearing.

| Surface | Renders | For |
|---|---|---|
| snapshot | Configuration + Capability | tests, benchmarks, CI, artifacts |
| structured query | Decision + Measurement | tests, users, debugging |
| log | whatever is worth noticing unasked | humans, live |
| recorded artifact | snapshot embedded in baselines | offline, months later |
| CI diff | snapshot vs snapshot | drift detection |

A decision worth logging is worth recording; a decision worth recording is not
always worth logging.

### Runtime versus offline

- **Runtime:** all of it. Cheap to ask, safe to ask in a loop.
- **Offline:** the snapshot, embedded in every recorded artifact. A baseline from
  a year ago must still answer "under what configuration?" without its machine.
  This closes principle 4's hole, at `check_baselines.stamp()`, in one edit.
- Decisions are **not** persisted by default. Persisting them makes this a
  tracing system (§8).

## 7. The five consumers

| Consumer | Reads | Replaces |
|---|---|---|
| **Tests** | Decision | reimplemented thresholds (`vkvalidate.py`) |
| **Benchmarks** | Configuration, stamped into artifacts | a `recorded` block that omits switches |
| **CI gates** | Configuration diffs | nothing |
| **Docs site** | the switch registry, generated | a hand-written reference that happens to be correct |
| **Users** | Decision, in their own vocabulary | reading `vulkan_backend.cpp` |

### Users are a first-class consumer, and this constrains the design

The questions are not developer questions, and they come in **two distinct
classes** that arise together during debugging but are different information:

**Decision questions — "why did this happen?"**

> Why did this run on the CPU? Why wasn't split-K selected? Why wasn't the
> register kernel used? Why did this operation fall back?

Answered from **Decision** facts. Specific to one operation, produced at the
moment of choosing, and meaningless without the alternatives that were rejected.

**Provenance questions — "what environment produced this?"**

> Which runtime switches were enabled? Which driver? Which Vulkan capabilities?
> Which backend? Which build?

Answered from **Configuration** and **Capability** facts. Process-wide, fixed
early, and the same for every operation in the run.

The distinction is not cosmetic. Provenance is what makes a *recorded artifact*
interpretable months later and is cheap enough to be always-on; decisions are
per-operation and bounded. Conflating them is how a system ends up either
persisting far too much or being unable to explain a benchmark it recorded last
year. "Why is this slower than yesterday?" is the question that needs both — a
provenance diff to find what changed, then decisions to find what that changed.

Three consequences follow, and none is optional:

1. **The vocabulary must be the user's.** A user asks about `a @ b` at
   `(512, 512)`. `gemm_reg:wg256_sg0_lv4_256_32_32_32_2_2_6_1_1_0_0` is an
   answer to a different question. Decisions must be addressable by operation
   and shape, with the internal key available underneath rather than instead.

2. **The answer must survive the question being asked afterwards.** A user
   notices the surprise and *then* asks why. If the only way to learn is to set
   an environment variable and reproduce, the architecture is developer-only.
   This argues for a small always-on ring of recent decisions — which collides
   with §8 and is resolved there, not waved away.

3. **It must not require a debug build, a rebuild, or a preceding decision to
   observe.** "Why is this feature unavailable" must be answerable on the
   installed wheel.

Adding this consumer is what turns "expose internals" into "explain behaviour",
and it is the difference between diagnostics and an API.

## 8. The cost of observation must itself be observable

Stronger than "must not perturb", and the reason is that it can be checked
continuously rather than asserted once.

**The recorder reports its own overhead as a Measurement fact.** If enabling
decision recording costs X% of dispatch time, X is a measured, queryable
property of the recorder — not an invisible side effect.

**And it is treated as an ordinary benchmark, not a special internal number.**
Being "internal" is not a reason for weaker discipline; if anything it is a
reason for more, because nobody else will notice it drifting. So observation cost
gets exactly what every other measurement in this project gets:

| | Same rule as every benchmark |
|---|---|
| **baseline** | recorded in `bench/baselines/`, alongside the others |
| **conditions** | the `recorded` block — commit, driver, warmed, and now switches (principle 4) |
| **variance** | the measured noise floor, `gpu_run_sd` and `NOISE_SIGMA` |
| **regression criteria** | the same threshold as any other benchmark entry |
| **warm-up** | `measure()` warms immediately before timing (principle 3) |

### Measured, 2026-08-02

| | ns per publication |
|---|---|
| publish, nobody recording | **58.8** |
| publish, recorded | **131.7** |

Conditions: release build, `bench/cpp` harness, minimum of the run, development
machine. The recorder adds **~73 ns per decision** on top of publication itself.

**The end-to-end measurement failed, and that failure is the more useful
result.** A min-spec matmul with recording on and off, three processes per arm,
warmed, produced overlapping arms — the recording arm even had the *lower*
minimum, which is physically impossible. One mutex and one deque push against a
~2 ms matmul is roughly four orders of magnitude under process-to-process
variance. Principle 1 exactly: the comparison was precise about the wrong
phenomenon, and a number taken from it would have been noise wearing a decimal
point. The primitive is what can be measured honestly, so that is what is
measured.

**What that answers.** §7.2 asked whether an always-on ring is affordable, and
the answer is *it depends on the publishing rate, and the rate is the thing to
watch*:

- at today's rate — one publication per matmul, only on a min-spec device —
  131.7 ns against ~2 ms is **0.007%**, which is why nothing could detect it.
- at one publication per small dispatch (~5 µs), it would be **~2.6%**, which is
  well above the noise floor and would need to be opt-in.

So the always-on ring is affordable *for coarse decisions* and not for
per-dispatch ones. That is a design constraint discovered by measuring rather
than assumed in either direction, and it means per-dispatch attribution (#99)
must not simply reuse this path.

It also inherits principle 1: before believing the overhead number, prove the
measurement is measuring the recorder and not the execution environment — which
is what the failed end-to-end attempt above establishes. The regression gate
remains to be built (#117), and must prove its own failure path by forcing an
allocation per publication.

This resolves the tension in §7.2. The always-on ring is permitted *because* its
cost is measured and gated, not because it is assumed cheap. If the measurement
says it is not free, the design changes — see §13.

## 9. Where observability ends

Prohibitions, because this kind of work grows without anyone deciding it should.

**This is not:** a logging framework (no sinks, appenders, per-module levels,
formatters — `util/log.h` stays as small as it is); a metrics or telemetry system
(no time series, export protocol, daemon, network anything); a tracing system (no
spans, no timeline format, no Perfetto/OpenTelemetry); data inspection (tensor
*values* are debugging, and `VKML_VULKAN_DUMP` already covers that, capped at 256
elements).

**The boundary rule: it records decisions and configuration, never data.**
Bounded memory, bounded vocabulary, no per-element and no per-iteration growth.

If a proposal cannot be phrased as *"which of these did the system choose, and on
what evidence"*, it does not belong here.

## 10. What must be impossible afterwards

1. Adding an environment switch that does not appear in Configuration.
2. Adding a kernel whose name exists only as a string literal.
3. Taking a silent fallback — a capability branch that neither throws nor records.
4. Recording a baseline without its active configuration.
5. A test asserting on a backend decision by reimplementing the rule.
6. A decision record that claims something the profiler contradicts (§5).

1–3 are enforceable at the decision site. 4 is `stamp()`. 5 is a gate over the
test tree. 6 is the cross-owner check.

---

## 11. Order of work: mechanism first, then migrate the instances

**O3 before O1.** Decision recording creates the abstraction; configuration
reporting and the allocator become its first two consumers. This is the ordering
that has worked repeatedly here — build the mechanism, then migrate.

| | Work | After |
|---|---|---|
| **O3** decisions | the fact model + recorder + cross-owner check | tests assert instead of reimplement; `vkvalidate`'s category-3 marker deleted |
| **O1** switches | registry, as a Configuration consumer | switches enumerable; baselines stamped; principle 4's hole closed |
| **O2** allocator | *one call site* | ceases to be separate work — a decision site once O3 exists |

It reaches further than the O-tasks: six deferred Product Evolution tasks (#17,
#18, #32, #38, #39, #40) have "a measurement showing X is not hot" as their
deletion evidence, and #17's title is literally *"once profiling justifies it"*.
Those are blocked on an observability capability, not on anyone's time.

---

## 12. Alternative designs considered

**Add more logging at the decision points.** Rejected: it is what exists, and O3
is the finding that it is insufficient. Text cannot be asserted on, aggregated or
attributed to an operation — and CLAUDE.md's own trap list records that pytest
captures stderr, so under test the decision record does not exist at all.

**Extend `vulkan_last_profile` / `vulkan_pipeline_stats`.** Rejected: they are
Vulkan-backend surfaces and half the facts are not Vulkan facts. `VKML_EAGER`
changes graph realization; build identity is not a device property. It would put
"what am I running?" behind `init_vulkan`, wrong for a CPU-only build. Under §4
it also gives Configuration an owner that does not own it.

**Parse the existing log output.** Rejected: category 3 with extra steps — the
parser becomes a second model of the message format, drifts silently, and is
defeated by the stderr capture above.

**Adopt a tracing framework (Perfetto, Chrome trace, OpenTelemetry).** Rejected:
solves timeline visualization, which is not the problem — the timings are already
good and already structured. Adds a dependency and an export format to answer
"which kernel and why". §9 exists to keep that door shut.

**A declarative model of the system that tools read.** Rejected, and the most
important rejection to record because it is the most attractive design and would
almost certainly be wrong within a year. A hand-maintained description of vkML's
switches and kernels would drift from the code with nothing able to compare the
two — the same failure as the rejected GEMM selector, and as PRE-COMMIT-CHECKLIST
§6, which claimed to list the gates CI runs and was wrong in both directions.

**Do nothing; the log is adequate for a solo project.** Not rejected on
principle — it is adequate for someone who wrote the code and remembers it. It is
inadequate for the five consumers in §7, for the six deferred tasks in §11, and
for a bug report from hardware this project does not own, which CLAUDE.md names
as the main way device-specific assumptions get found.

---

## 13. How this architecture could fail

Not trade-offs and not rejected alternatives — the conditions under which this
document becomes the wrong answer, written down now while there is no incentive
to be generous about them.

### The assumptions it rests on

1. **Decisions are few and coarse.** One per operation, not per element or per
   workgroup. If vkML acquires a per-tile or per-wave decision — an autotuner
   choosing per block, say — the bounded ring becomes a sampling problem and §9's
   "never data" boundary is under real pressure rather than rhetorical pressure.
2. **A decision's alternatives are enumerable at the site.** "Register-blocked,
   rejected because 256 > 128" is a closed set. A search-based or learned
   selector has no such list, and `instead_of` degrades to a score nobody can
   interpret.
3. **Publication is free.** §8 measures this rather than assuming it, but the
   design assumes the *answer* will be "free". If it is not, §7.2 loses and the
   user-facing consumer is the casualty.
4. **One owner per fact is achievable.** It is clean today because vkML is
   single-process, single-device-at-a-time, and synchronous at the API boundary.
5. **The driver tells the truth.** The cross-owner check in §5 rests on
   `vkGetPipelineExecutableStatisticsKHR`. On a driver that does not implement it
   — or implements it approximately — the independent half of the check silently
   becomes unavailable, not wrong. It must report unavailability rather than pass.

### What would invalidate it

| Future requirement | What breaks |
|---|---|
| **Multi-device or multi-queue execution** | Capability stops being process-wide. Provenance becomes per-device and the snapshot is no longer one object. Assumption 4 fails first. |
| **An autotuner or learned kernel selection** | Assumptions 1 and 2 both fail. Decisions become numerous and their alternatives become a search space; this architecture would describe the *outcome* while the interesting fact is the *search*. |
| **Asynchronous or reordered execution** | "Per operation" stops being well-defined ordering, and a ring buffer records interleavings rather than a sequence. |
| **A second frontend or an out-of-process client** | Delivery surfaces (Part II) must serialise, which is fine — but if the fact model itself needs a wire format, Part I has acquired a representation and is one step from the declarative model rejected in §12. |
| **Users needing history, not the last N** | Persisting decisions makes this a tracing system, which §9 forbids. That prohibition would need revisiting deliberately, not by growth. |

### Evidence that the abstraction is wrong

- **A fact that genuinely needs two owners.** The ownership rule is the load-
  bearing idea; one honest counter-example that cannot be resolved by splitting
  the fact means the model is too coarse.
- **Consumers reaching past the facts.** If tests or tooling start reading
  internal state directly because the fact model cannot express what they need,
  the vocabulary in §3 is too narrow, and the response is to widen it explicitly
  rather than tolerate the bypass.
- **Decisions that cannot state a reason.** If sites routinely publish "chose X"
  with no `because`, then what is being modelled is not a decision, it is a
  dispatch log, and §3 is claiming more structure than exists.
- **The cross-owner check never failing.** A validation that has never gone red
  is unproven, by this project's own principle 2. If it cannot be made to fail
  under a deliberately lying recorder, it is decoration.

### How we would recognise it in time

The gates in §10 are the early-warning system, not just enforcement. Item 6 going
permanently unavailable means assumption 5 has failed. §8's baseline drifting
upward means assumption 3 is failing. A rising exemption count in the §10 gates
means the model is being worked around, which is how architectures die — not by
being rejected, but by accumulating exceptions until they describe nothing.

## 14. What would change my mind, today

- If §8's measurement shows the always-on ring is not free, §7.2 loses and
  decisions become opt-in — with the user-facing consequence stated plainly
  rather than hidden.
- If the switch registry cannot be made compile-enforced, item 1 of §10 degrades
  to a grep-based gate, which is weaker and must be labelled as such.
- If the cross-owner check in §5 turns out to be circular — if the profiler's
  pipeline list is derived from the same code path as the decision — it is not an
  independent check and must be replaced or dropped, not kept for comfort.
- If no consumer beyond tests ever reads Decision, §6's structured query is
  over-built and a test-only accessor was the right size.
