# vkML engineering principles

Principles the project has actually paid for. Each names the defect that
produced it, because a principle without a scar is advice and gets ignored.

`docs/DOCUMENTATION-PRINCIPLES.md` covers the documentation platform
specifically. These apply to the whole repository.

Every principle here should progress: **observation → principle → tooling →
gate → continuous verification.** A principle that can only be remembered will
eventually be forgotten. The "enforced by" line on each says how far it has got.

---

## 1. Never optimise before proving the measurement measures what you think

*A precise measurement of the wrong phenomenon is still the wrong measurement.*

A 38.6% regression on `dispatch 1 element` was tracked, investigated and
refuted. The same binary produces 0.01052 ms cold and 0.00572 ms after sixty
1024-cubed matmuls, in one process; the baseline it was compared against was
0.00664, inside that range. The code was 16.3% *faster* than the thing it was
accused of regressing against.

What made it convincing was that it had survived a noise control. Ten runs in
ten separate processes gave a 3.3% spread — because all ten were cold in the
same way. **A consistent measurement of the wrong thing looks exactly like a
good measurement.**

Before optimising, rule out the measurement itself:

- Is it reproducible, and what is the variance across *processes*?
- Is the instrument adequate? (Here: timestampPeriod 10 ns, so 9.2 µs is 920
  ticks — quantisation ruled out with a number.)
- What is being measured — kernel time, fixed overhead, transfer, host cost?
  (`gpu_min` for relu is flat from 1 to 4096 elements, so at one element it is
  dispatch overhead, not kernel work.)
- What is the execution environment doing? Clock state, thermal, scheduling.
- What evidence rules out each alternative?

**Enforced by:** `measure()` in `bench/gpu_bench.py` warms before every timing;
`docs/MEASUREMENT-AUDIT.md` records the figures. Not yet a gate — the effect is
a distribution, not a pass/fail.

## 2. Every mechanism proves itself by failing under controlled conditions

A gate, test, benchmark or validator that has never been demonstrated to fail
has never been demonstrated to work.

Three mechanisms here were doing nothing while reporting green:
`check_docs_links.py` printed the broken links it found and exited 0 for its
entire life; `check_docs_examples` reported PASS over content it never opened;
the mutation campaign reported SURVIVED for seventeen mutations it never
executed, because it rebuilt one extension and pytest imported another.

None was visible from a green run. All three were found by breaking something
on purpose.

It keeps happening, including to gates written *by* someone applying this
principle. `check_min_spec.py` was written to catch a CI arm going missing; its
negative control removed `VKML_MIN_SPEC=1` from the command and the gate stayed
green, because the step's own comment still explained `VKML_MIN_SPEC=1` and the
check was reading the whole step. **It was matching the prose about the thing
instead of the thing.** Eleven minutes old, already dead, and it would have
looked correct forever. Assume this of every new gate until its control has
been run.

### Verification code deserves the same skepticism as production code

This is the general lesson, and it took a run of failures to see it as one
pattern rather than a run of bad luck:

| The verifier | What was wrong with it |
|---|---|
| `check_docs_links.py` | printed its findings, never called exit |
| `check_docs_examples` | reported PASS over content it never opened |
| `mutation_check.py` | rebuilt one extension, tested another |
| the typography probe | measured the sidebar, not the reading column |
| a red test, twice | read `$?` from the end of a pipe, not from the gate |
| `check_min_spec.py` | matched the step's comment, not its command |
| `check_gate_coverage.py` | let an exemption cancel the requirement it explained |
| the min-spec gate's skip path | treated "I could not run this" as "nothing to check" |

Every one of these was written *by someone who knew this principle*, and every
one looked correct. Verification code gets less scrutiny than production code —
no user runs it, no test covers it, and its output is a single line most people
only read when it is red. That is exactly the environment in which a thing
quietly stops working.

So: a gate is not a place to relax, it is a place to be more careful than usual.
Write the control before believing the check. When a verifier reports success,
ask what it would have done if the answer were unavailable — because reporting
"I cannot verify this" is worth far more than assuming success, and the failure
is silent in both directions.

### A different accessor is not a different owner

The subtlest form of a verifier that cannot fail: two sources agree, and the
agreement means nothing because they were never independent.

It was caught in the observability design before anything was built on it. The
proposed check was that a recorded decision naming `gemm_reg` could be validated
against `vulkan_pipeline_stats` showing that pipeline exists. Two APIs, two
owners, apparently a cross-check — except that *both* derive from the same
`pipes.get("gemm_reg")` call. They cannot disagree, so the check proves nothing.
What **is** independent is what the driver reports about the compiled pipeline
(`vkGetPipelineExecutableStatisticsKHR`) and what the profiler counts
(8 dispatches for 8 split-K partitions): different producers, capable of
contradiction.

The pattern predates observability, and recognising it is the reason several
things here were rebuilt:

- the mutation campaign rebuilt one extension and tested another — two paths that
  looked like one system
- `check_min_spec.py` compared a step's comment against its command, both
  written by the same hand at the same moment
- the coverage matrix and the tests it measured shared their notion of what an
  operator was

**Before calling something a validation, name the two producers and show they can
disagree.** If a single change would move both sides together, there is one
source of truth wearing two hats, and the comparison is decoration.

The general form is worth keeping in mind wherever independence is claimed:
two accessors over the same state, two documents generated from the same
template, two tests exercising the same helper.

**Enforced by:** `scripts/verify_gates.py`, in CI. It damages a file, runs the
gate, checks the exit status and restores. A gate that cannot fail fails the
build. A gate with no control is reported as unverified rather than omitted.

## 3. Warm-up is a precondition of each measurement, not a phase

Corollary of 1, and it cost a wrong fix to learn. A warm-up added at the top of
the benchmark left the number unchanged:

    warm → measure                 0.00612 ms
    warm → transfers → measure     0.00932    the transfers undo it
    transfers → warm → measure     0.00560

Anything between the warm-up and the measurement can undo it.

**Enforced by:** `measure()` warms immediately before timing. Suite-wide spread
fell from a median of 50.2% to 10.2%.

## 4. A baseline without its conditions is not a baseline

`bench/baselines/rx5600m.json` records GPU, subgroup sizes and memory. It does
not record clock state, driver version, timestamp period, or the commit it was
taken at — so nothing in it could rule out the explanation that turned out to be
correct in principle 1.

**Enforced by:** `scripts/check_baselines.py` — every recorded artifact that
something compares against must carry a `recorded` block.

## 5. Three categories, in order of preference

1. **Generated from the source of truth.** It cannot drift; drifting means the
   build is broken.
2. **Observed from the real system.** Running it and recording what happened.
3. **Reimplemented logic.** Assume it is wrong until proven otherwise.

An interactive GEMM kernel selector was rejected under this: the choice depends
on shape, on device limits read at runtime, on three environment overrides and
on a split-K planner. A JavaScript version would be a second implementation with
nothing able to compare the two — and the intuitive model is wrong anyway, since
a 1×512 by 512×1 matmul selects `gemm_reg`, not `gemv`.

**"Can it be generated?" is not enough.** A thing can be generated from a
hand-written model and still be a lie.

## 6. A gate guards a class, not the instance that prompted it

The unrendered-markup gate checked `*emphasis*` but not `` `code` ``, so a
generated heading shipped with backticks intact and the gate passed. The
design-system gate used a budget of ten sizes against a nine-step scale, so the
*first* new size passed silently — exactly the step that starts an accumulation.

Ask: does this check the whole class, or only what I happened to hit?

## 7. One example is a hypothesis about a population

When you find one instance, immediately ask whether it is the only one — and
answer it by measuring, not by feeling.

Both answers have been useful. The CI audit found `benchmark` was the *only* job
asserting nothing, which stopped a pointless sweep. The baseline audit found the
missing-conditions problem in three artifacts, not one. Neither was predictable
in advance.

## 8. Write down why you believe something is correct

When concluding "this is correct", record what convinced you and what evidence
would change your mind. Conclusions without their evidence cannot be re-examined
when the surrounding facts move, and this project has repeatedly found that the
first explanation was wrong while the investigation was valuable.

Applied: every closed task here answers what assumption turned out to be wrong,
what class of bugs the change prevents, and what would justify revisiting.

---

## Closing an engineering task

Before considering it done:

1. What assumption turned out to be wrong?
2. What class of bugs does this prevent in future?
3. Can this knowledge become tooling instead of documentation?
4. Can that tooling become a gate?
5. Can that gate prove its own failure path?

If any answer is "yes", do that work before moving on.
