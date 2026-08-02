# ADR 0006 — Lazy assign, and how work is batched into submissions

**Status:** accepted. **Stage A implemented and measured (7 below). Stage B is
part built**: multi-root realize shipped and is measured in 9 — `realize()` takes
a list of tensors, `dispatch/executor.h` takes `std::span<const NodePtr>` — and
**the optimisers now use it, measured in 10** — but the Assign node did **not**,
and is blocked on the two findings in 9. Stage C is still proposed.

This line said "stages B and C still proposed" while 9 below described stage B's
first half as built and measured, so the summary contradicted the body and a
reader who stopped at the status would conclude multi-root realize did not exist.
**Date:** 2026-07-29
**Covers:** task #19 ("batch optimiser realize calls so a step is one submission"), whose
premise this document corrects.
**Hardware:** AMD RX 5600M (RDNA1), RADV. Every measurement below was run on it.

---

## 1. Context, and the premise that turned out to be wrong

Task #19 said the optimiser costs too many submissions because of the explicit
`.realize()` calls in `python/vkml/optim.py`, and that batching them into one submission
would fix it. After the H11 fix (`docs/BACKWARD-PERF-INVESTIGATION.md`) the optimiser
became the largest single cost in a training step, so #19 came up for implementation.

**Reading `Tensor::assign_` first disproved the premise.** `src/api/tensor.cpp` does this,
unconditionally:

```cpp
realize();
const Tensor flat = src.contiguous();
flat.realize();
...
std::vector<std::byte> staging(nbytes);
backend.copy_to_host(staging.data(), *flat.node()->storage, ...);
backend.copy_from_host(*node_->storage, node_->storage_offset, staging.data(), nbytes);
```

Every assignment is a **device to host to device round trip**, for data that never needed
to leave the GPU.

### Measurements

Same arithmetic, 512x512 f32, minimum over 20 iterations, profiling on throughout:

```
                                submissions    wall       gpu
p.assign_(p.detach() - g*lr)        3.0      0.921 ms   0.211 ms
(p.detach() - g*lr).realize()       1.0      0.289 ms   0.211 ms
```

**The GPU time is identical.** The extra 0.63 ms is entirely transfer and submission --
about three times the cost of the arithmetic it exists to store.

That accounts exactly for the optimiser's measured 32 submissions per CIFAR-100 step:
8 parameters x 3 for `assign_`, plus 8 momentum `realize()` calls. The whole stage is
3.30 ms of wall for 0.31 ms of GPU (`gpu/wall` = 0.09), and 1.19 MB of parameters makes
a 2.39 MB round trip through host memory every step.

**This is not an optimiser problem.** `nn.BatchNorm2d` updates its running statistics
with `assign_` too, so it is paid on the FORWARD pass of any model that uses one:

```
BatchNorm2d, one layer, training mode    9.0 submissions   1.650 ms
BatchNorm2d, one layer, eval mode        1.0 submissions   0.467 ms
```

Eight extra submissions and 1.18 ms for two running-statistic updates. A network with
fifty BatchNorm layers pays that fifty times per forward pass.

### What that makes the task

**Differently shaped.** Batching the `realize()` calls addresses 8 of the 32 submissions
and none of the round trip. Worse, it *cannot* be done while `assign_` stays eager: a
download forces a wait, so any graph containing an assignment is cut at that point no
matter how the realize calls are arranged. The ordering is forced -- `assign_` must
become part of the graph before batching is possible at all.

---

## 2. Prior art

Read from the clones in `third_party/reference/`. Nothing here was run; these are
code-reading findings, and the file and symbol names are given so they can be checked.

### llama.cpp / ggml-vulkan — the closest peer, same API

Two findings, and the second contradicts the naive form of #19.

**Same-device copies never touch the host.** `ggml_vk_buffer_copy` (ggml-vulkan.cpp)
branches on `src->device == dst->device`. Same device is `vkCmdCopyBuffer`, recorded into
a command buffer. The host staging path exists only for the genuine cross-device case:

```cpp
static void ggml_vk_buffer_copy(vk_buffer& dst, ..., vk_buffer& src, ...) {
    if (src->device == dst->device) {
        ...
        ggml_vk_buffer_copy_async(subctx, dst, dst_offset, src, src_offset, size);
        ...
    } else {
        // Copy device to device  [via sync_staging]
    }
}
```

vkML's `assign_` takes the host path even though it has already *checked* that source and
destination share a device.

**A graph is deliberately NOT one submission.** `ggml_backend_vk_graph_compute` batches
nodes to a work target instead:

```cpp
bool submit = (submitted_nodes >= ctx->device->max_nodes_per_submit) ||
              (flops_per_submit != 0 && batch_flops >= flops_per_submit) ||
              (i + ctx->num_additional_fused_ops >= last_node) ||
              (almost_ready && !ctx->almost_ready_fence_pending);
```

The comment gives the reason: *"Submit after enough work has accumulated, to overlap CPU
cmdbuffer generation with GPU execution."* Three details are worth copying the thinking
from, not the code:

* **The first submissions are deliberately small.** `if (submit_count < 3)
  flops_per_submit *= 2;` — start the GPU early, then amortise. Latency first, throughput
  after.
* **The cap is device-aware.** Weaker AMD parts get a much lower `flops_cap`, with the
  comment *"On weaker AMD GPUs larger submissions can hit a driver timeout"*. One
  enormous submission is a correctness risk, not just a tuning choice.
* **There is a node cap as well as a work cap**, for graphs with many cheap nodes.

### tinygrad — the closest architecture, and the cleanest answer

**Assign is a graph node, not an eager copy** (`tinygrad/tensor.py`):

```python
assign = self.uop.after(self.uop.store(x.uop))
```

A STORE with an ordering edge. It schedules and fuses like any other operation.

**The optimiser is split into build and run** (`tinygrad/nn/optim.py`):

```python
def step(self):
    Tensor.realize(*self.schedule_step())
```

`schedule_step()` builds the update for *every* parameter and returns the tensors to
realise; `step()` realises them in one call. `Tensor.realize(self, *lst)` is variadic by
design.

**Horizontal fusion is a separate, opt-in lever** — `self.fused` concatenates every
parameter into one flat tensor, runs the update once, and slices the results back out.
The comment is explicit that this is a distinct mechanism from the batching above.

So tinygrad has two levers, and keeps them separate: *graph batching* (always on) and
*horizontal fusion across parameters* (opt-in, for when per-kernel overhead dominates).

### CUTLASS, rocBLAS, Tensile, CLBlast — a different problem

These are kernel libraries, not runtimes. They do not schedule submissions at all; they
are called by something that does. Checked rather than assumed: none of them contains a
submission-batching policy.

They do show the *same underlying pattern* one level down. All four ship batched entry
points — `CUTLASS/examples/24_gemm_grouped`, `rocblas_*_strided_batched`,
`CLBlast/src/routines/levelx/xgemmbatched` — because per-launch overhead dominates when
each problem is small. CUTLASS's grouped GEMM passes pointers and per-problem sizes in
GPU memory so that *differently shaped* problems still share one launch.

That is the same idea as tinygrad's `fused` optimiser: when you have many small
independent operations, put them in one launch. It is the right precedent for a possible
later stage, and the wrong one for the problem in front of us, which is a host round trip.

---

## 3. What vkML already has

`realize()` in `src/dispatch/executor.cpp` **already takes several roots**:

```cpp
void realize(std::span<const NodePtr> roots) {
    const std::vector<Node*> order = topological_order(roots);
    ...
    backend.compute(order);
}
```

One `backend.compute()`, and `VulkanBackend::compute()` records one command buffer and
makes one submission. **I verified this by reading, not by running** -- the span overload
has no Python binding, so nothing currently calls it with more than one root.

So the batching primitive exists at the executor layer and is unreachable from where it
is needed. That matches step 4 of the workflow: the abstraction can carry this, and what
is missing is a lazy assign and a binding.

---

## 4. Options

### A. Make `assign_` a device-side copy, keep it eager

Replace the host round trip with a device-to-device copy, as ggml does.

| | |
|---|---|
| **Benefit** | Removes the 2.39 MB round trip. Drops `assign_` from 3 submissions to about 2 |
| **Cost** | Needs a `copy_device_to_device` on the `Backend` interface — a public interface change, and the CPU backend needs it too |
| **Worthwhile when** | You want most of the win for a small, contained change |
| **Not worthwhile when** | You want one submission per step — this cannot get there, because an eager assign still cuts the graph |

### B. Make assign a graph node, and batch realize (tinygrad's model)

Add an `Assign` op whose destination is an existing buffer. `assign_` records it; nothing
executes until something is realised. Expose the multi-root `realize()` to Python, and
give `Optimizer` a `schedule_step()`/`step()` split.

| | |
|---|---|
| **Benefit** | A whole optimiser step becomes one submission. Fixes BatchNorm's forward path at the same time, because it fixes `assign_` for every caller |
| **Cost** | The largest change here. A node that writes to a buffer another node reads is a real aliasing hazard, and `bind_storage()` currently allocates for every computed node rather than binding to an existing one. Both need care |
| **Worthwhile when** | Submission overhead is a material share of the step — which is now measured at 27.6% |
| **Not worthwhile when** | The graph machinery cannot express the write ordering safely. **This needs verification before committing** |

### C. Horizontal fusion across parameters (tinygrad `fused`, CUTLASS grouped)

Concatenate parameters, run one update kernel, slice back.

| | |
|---|---|
| **Benefit** | Fewest dispatches — one update kernel instead of one per parameter |
| **Cost** | Does nothing about the round trip, so on its own it fixes nothing here. Concatenation itself costs bandwidth unless parameters are allocated contiguously from the start, which is an allocator change |
| **Worthwhile when** | After A and B, if per-dispatch cost still dominates for models with many small parameters |
| **Not worthwhile when** | Now. The measurement says the cost is transfer and submission, not dispatch count |

---

## 5. Recommendation

**A first, then B. Not C.**

A is a small, contained change that removes the largest single cost and is independently
valuable. It also de-risks B: with the copy already device-side, making assign lazy is a
scheduling change rather than a scheduling *and* transfer change, and the two can be
measured separately.

B is where the "one submission per step" goal is actually reachable, and it fixes
BatchNorm's forward path as a side effect — which A does not, and which #19 never
mentioned because #19 was scoped to the optimiser.

C is deferred with a revisit trigger: **after A and B, if dispatch count rather than
submission count dominates a measured step** (P7).

**One thing not to copy from #19's wording: "a step is one submission" is not the goal.**
ggml-vulkan submits several times per graph on purpose, ramps the first few submissions up
in size, and caps them per device because a single huge submission can trip a driver
timeout. The goal is *few* submissions with the GPU kept busy, not *one*.

### Open question, to settle before B

Whether vkML's executor can express "this node writes to a buffer an earlier node reads"
safely. `Recorder::dispatch()` emits a global barrier between every pair of dispatches, so
ordering within a submission is conservative and probably sufficient — but
`bind_storage()` allocates fresh storage for every computed node and would need to bind an
Assign node to its destination's existing storage instead. **I have read both and have not
tested either.** This is the first thing to verify when B starts.

---

## 6. Consequences

* Task #19's title is wrong and should be rewritten around `assign_`, not the realize calls.
* `Tensor::assign_` changes behaviour under both options. Under A it stops moving data
  through host memory; under B it stops executing at call time. **B changes observable
  timing semantics of a public API** and needs saying in the release notes.
* `python/vkml/optim.py`'s module docstring attributes the cost to "how often Python
  forces materialisation". That is half right and should be corrected: the round trip is
  the larger half, and it is not about frequency.
* A test asserting a submission BUDGET for an optimiser step, in the style of
  `test_backward_emits_no_degenerate_reductions`, is the right gate for both stages.

---

## 7. Stage A, as built and measured

`Backend` gained one method:

```cpp
virtual void copy_device_to_device(Storage& dst, int64_t dst_offset, const Storage& src,
                                   int64_t src_offset, size_t nbytes) = 0;
```

Pure virtual rather than defaulted. A default staging through the host would be
correct and quietly slow, which is the exact defect this removes; a new backend
should have to answer the question rather than inherit a silent answer. CPU
implements it with `memmove`, Vulkan by reusing the `Recorder::copy` that staging
already used, with no chunking -- there is no staging capacity to bound it.

`assign_` stays **eager**. Nothing about when work happens changed, only where the
bytes travel.

**The overlap case is the one that still goes through the host.** `vkCmdCopyBuffer`
requires disjoint regions when source and destination are the same buffer, and
`t[0:5].assign_(t[2:7])` reaches exactly that. `storages_overlap()` detects it and
falls back to the previous path, so the semantics are unchanged rather than
narrowed.

### Method

Both arms measured with the same script on the same machine, the A arm being the
frozen unmodified build restored with `git stash` and rebuilt (rule 7). Minimum
across runs (rule 2), validation off (rule 5), warm (rule 6), profiling on in both
arms (rule 4).

**A measurement hazard worth recording.** Part-way through, the stage split
reported the step at 28.50 ms against a 11.99 ms baseline -- an apparent 2.4x
regression in `forward` and `backward`, which this change cannot touch. Re-running
gave 11.20 ms and 26.13 ms for identical builds. The cause was the GPU sitting at
a low DPM state:

```
/sys/class/drm/card1/device/pp_dpm_sclk
0: 200Mhz    1: 400Mhz *    2: 1500Mhz
```

400 MHz of a possible 1500. Every wall and GPU figure on this machine is therefore
bimodal, and a single run of either arm can land in the wrong mode. The minimum
over several **process** runs is what selects the high-clock samples; a minimum
within one process is not enough, because the clock state persists for the whole
process. Submission COUNTS are unaffected, which is why they are the primary
evidence below.

### Results

Per-operation, minimum of 3 process runs x 30 reps:

```
                        before          after
assign_ 512x512      3 subs 0.704 ms   2 subs 0.234 ms    3.0x
SGD step, 4 params  20 subs 2.046 ms  16 subs 1.276 ms    1.6x
Adam step           24 subs 3.379 ms  20 subs 2.301 ms    1.5x
AdamW step          24 subs 3.424 ms  20 subs 2.569 ms    1.3x
BatchNorm2d train    9 subs 1.256 ms   7 subs 1.142 ms    1.1x
BatchNorm2d eval     1 sub  0.316 ms   1 sub  0.325 ms    unchanged
```

Exactly **one submission per assignment** disappears, everywhere -- the download
half of the round trip. The upload becomes the device copy, which still costs a
submission because stage A is eager. That is the ceiling for this stage, and it is
why stage B exists.

Adam and AdamW were never measured before; they cost more than SGD because they
realise `m` and `v` per parameter as well as assigning.

**BatchNorm barely moves in wall time (1.256 -> 1.142) despite losing two
submissions.** Its running statistics are 64 floats, so the transfer was never the
cost there -- the submissions were, and one submission is roughly 0.07 ms. This is
worth stating because the ADR predicted BatchNorm would benefit, and the honest
answer is: in submissions yes, in time barely. A network with fifty such layers
would save a hundred submissions, and whether that is visible needs measuring on
such a network rather than inferring from this one.

Whole CIFAR-100 step, minimum of 6 process runs x 25 reps:

```
              before                       after
forward     1 sub   2.43 ms            1 sub   2.36 ms
backward   11 subs  5.65 ms           11 subs  6.03 ms
optimiser  32 subs  2.74 ms           24 subs  1.73 ms    1.6x
step               11.01 ms                   10.23 ms
```

The optimiser stage is 37% faster and the whole step 7%. `forward` and `backward`
differ only by run-to-run noise, which is the expected result: neither calls
`assign_` in this model.

**GPU time is unchanged at 7.33 ms in both arms**, which is the check that matters.
Stage A moves no arithmetic; if the GPU total had moved, something would be wrong.

---

## 8. Stage B0 -- the assumptions, tested

4 and 5 named two assumptions, both read and neither run. They are now run
(`tests/cpp/test_aliasing.cpp`, plus API-level cases in
`tests/python/test_vulkan_kernels.py`). **One of them is false.**

### Assumption 1 -- an Assign node can bind to its destination. FALSE.

`is_realized()` is *defined* as `storage != nullptr` (`graph/node.h`), and
`topological_order()` treats a realised node as a leaf:

```cpp
// A realised node is a leaf for scheduling purposes: its value
// already exists, so nothing behind it needs recomputing.
if (node->is_realized()) { done.insert(node); stack.pop_back(); continue; }
```

So binding an Assign node to its destination's storage up front makes the
scheduler **skip it**. It would look computed before it had run. Verified:

```
order size while unbound   3
order size once bound      0
```

**Consequence for stage B.** The predicate conflates two facts that coincide
for every node today -- *has storage* and *has been computed*. Assign is the
first node for which they differ, so stage B must separate them before it can
schedule anything. The smallest form is an explicit flag or a distinct
`is_computed()`; that is a design decision, not an implementation detail, and
it belongs in stage B's own commit rather than being smuggled in.

### Assumption 2 -- the global barrier orders WAR. SUPPORTED BY READING, NOT BY TEST.

The barrier is one `VkMemoryBarrier2`:

```
srcStageMask = dstStageMask = COMPUTE_SHADER | TRANSFER
srcAccessMask = SHADER_WRITE | TRANSFER_WRITE
dstAccessMask = SHADER_READ | SHADER_WRITE | TRANSFER_READ | TRANSFER_WRITE
```

RAW and WAW need availability and visibility, and the access masks supply both.
WAR needs only an **execution** dependency -- the earlier access only reads, so
there is nothing to make available -- and the stage masks supply that. On the
spec's rules the barrier is sufficient for all three.

**The tests cannot confirm it, and saying so is the point of this section.**
Behavioural tests were written for WAR across independent dispatches, RAW within
a graph, a dispatch writing the buffer its own input aliases, and determinism
over 20 repeats, on both backends. They pass. They **also pass with
`rec.barrier()` commented out of `VulkanBackend::compute()`** -- measured, not
assumed. A test that passes in both arms distinguishes nothing.

Vulkan's synchronization validation is the instrument built for this, and it is
blind here. It discovers shader accesses through **descriptor bindings**, and
vkML has none: `grep vkCmdBindDescriptorSets src/` returns nothing, because
buffers reach shaders as `bufferDeviceAddress` values in push constants. The
layer sees dispatches that touch no resources and reports no hazards whether or
not any exist. Enabling it changed nothing in either arm, which is why the
negative control came back empty.

**This is a standing limitation of the project, not of this task.** vkML's
central design choice -- no descriptors, addresses in push constants -- costs it
the ability to machine-check its own synchronisation. Worth knowing before the
next person reaches for the validation layer expecting an answer.

### What this changes about stage B

* The storage-binding question has a concrete answer and a concrete blocker.
  Stage B starts by separating "bound" from "computed", not by adding an OpKind.
* The synchronisation question does not block stage B: the barrier is
  conservative and the analysis says it covers WAR. But it cannot be *verified*
  by test on this design, so stage B should keep the barrier exactly as it is
  and treat any future attempt to make it selective as needing a different kind
  of evidence.
* The tests are worth keeping as value-regression guards. They are labelled in
  the file with what they do and do not detect, so nobody later mistakes a green
  run for proof.

---

## 9. Stage B, part way: three findings that reshape the rest

Multi-root realize is built and measured (4 tensors: 4 submissions -> 1). Before
adding the Assign node, three assumptions surfaced. Two were validated by
experiment, one by reading the type system. None of them was anticipated in 4.

### Finding 1 -- `detach()` forces realization, so it CUTS the graph. Measured.

```
building a*2.0 lazily              0 submissions
calling .detach() on it            1 submission
building then realizing explicitly 1 submission
```

`autograd.cpp` `detach()` realizes its source when that source is not yet
computed -- it has to, because it shares the source's *buffer*, and an unbound
node has none to share.

**Every optimiser calls `.detach()` on its intermediates** (`_velocity`, `_m`,
`_v`), so each parameter's update is cut at that point no matter how the realize
calls are arranged. A prototype SGD that builds all updates lazily and calls
`V.realize()` once still measured:

```
SGD today              16 submissions   1.307 ms
SGD batched realize    13 submissions   1.174 ms
```

13, not the 1 + N = 9 predicted. Parameters agree to 1e-6 over 30 steps, so this
is a cost finding, not a correctness one. (An earlier run of this prototype
reported the parameters DISAGREEING; that was an unseeded numpy RNG in the
harness giving the two arms different training data, not a library defect.)

**This caps what any batching can achieve until `detach()` stops forcing
evaluation**, and that is a change to autograd, not to the optimiser.

### Finding 2 -- nothing would hold the Assign node. Verified.

If `assign_` builds an Assign node and returns, no live reference points at it,
so it is never scheduled and the assignment silently never happens.

The fix is the one tinygrad uses: re-point the tensor at the Assign node in
place. That works here -- verified that `Module.parameters()` returns the same
Python object every call, and the same object as the module attribute, so a node
swap is visible to the model. But it leads directly to:

### Finding 3 -- an Assign chain would accumulate across steps. Read, not run.

`Node::src` is `std::array<std::shared_ptr<Node>, kMaxSrc>`. If step N's Assign
holds step N-1's Assign as a source, each step's nodes keep the previous step's
alive, transitively, for the whole run.

`kFlagComputed` stops the SCHEDULER walking back, but it does not drop the
ownership edge, and nothing else does either: there is no site in `dispatch/`,
`graph/` or `tensor.cpp` that clears `src` after realization. So the chain is an
ownership leak, growing by one graph per step. Note `~Node` already tears down
iteratively *because* long chains occur -- the machinery for deep chains exists,
which is not the same as preventing this one.

tinygrad handles exactly this in `Tensor.assign`, with a branch on the uop's
`base` and `has_buffer_identity()`. Whatever vkML does needs an equivalent, and
it is not a detail of the Assign node -- it is most of the work.

**I have not run this one.** It follows from the ownership type and the absence
of any cleanup site; confirming it needs the Assign node to exist.

### What this makes the remaining task

The Assign node is still the right destination, but it is no longer the next
step, and it is bigger than 4 estimated. Two things now sit in front of it:

1. **`detach()` must stop forcing evaluation** for batching to be worth anything.
   That is an autograd change with its own correctness surface.
2. **A collapse rule** so an assigned tensor does not retain its history.

Both are ADR-sized. Recorded here rather than absorbed into an implementation
commit, because the estimate in 4 -- "the largest change here" -- understated it
in a specific way worth naming: the cost is not in the Assign node, it is in the
two invariants around it.

---

## 10. The batched optimiser, as built and measured (2026-08-02)

9 concluded that the Assign node was "no longer the next step". It was right
about that and wrong about what *was*: the batching multi-root realize already
makes possible had not been claimed, and it does not need the Assign node,
`detach()` to stop forcing, or a collapse rule. It needs the passes ordered so
that **nothing is detached before the batched realise**.

`Optimizer.step()` is now three passes, in the base class, for all four
optimisers:

```
pass 1   build EVERY parameter's state update lazily, realise them together
pass 2   build EVERY parameter's new value from that state, realise together
pass 3   assign
```

### Results

CIFAR-100 CNN, 8 parameters, optimiser phase only so it dominates the measured
window (rule 1b). Minimum of 40 warm steps within a process, and the minimum
across four process runs per arm; the A arm is the frozen unmodified build
restored with `git stash` (rule 7).

```
                    submissions          ms
                  before  after    before  after   speedup
  SGD               16      9       1.097  0.675    1.63x
  SGD momentum      24     10       1.659  1.134    1.46x
  SGD nesterov      24     10       1.862  1.147    1.62x
  RMSProp           24     10       2.058  1.255    1.64x
  RMSProp cent+mom  40     10       3.356  1.790    1.87x
  Adam              32     10       3.032  1.719    1.76x
  AdamW             32     10       3.199  1.911    1.67x
```

**Every arm's parameters are bit-identical to the baseline's** after 60 training
steps, checked as a float64 sum over every parameter. This is a scheduling
change and the check is what makes that a statement rather than an intention.

The submission count is now `2 + N` and no longer varies with the optimiser:
RMSProp centered-with-momentum has three state tensors per parameter and Adam
two, and both realise them in one submission. That is the multi-root realize
sharing a common subgraph — the dependency between RMSProp's momentum buffer
and its running average costs a dispatch, not a submission.

Whole CIFAR-100 step, best of 5 rounds of 20 steps:

```
                     before    after
  submissions/step     39        25
  step wall         13.57 ms  12.09 ms
  host and driver    42.0%     35.7%
  GPU / wall          0.58      0.64
```

### The finding worth keeping: fewer submissions is not the same as faster

An intermediate arm batched **only** pass 1: 24 → 17 submissions, and
1.836 → 2.123 ms. **Seven fewer submissions and slower than doing nothing.**
Pass 2 is where most of the submissions were, so removing a few from pass 1
while leaving pass 2 per-parameter merely made one big submission the GPU
finishes before the host can queue the next.

This is the counter-example to treating submission count as the objective, and
it is measured rather than argued. §4a of `EXTENSIBILITY-ROADMAP.md` should be
read with it in hand.

### What is now the next step, and what is not

The remaining `N` is `assign_`, still eager, one submission per parameter. That
is what the Assign node removes, and 9's two blockers still stand in front of
it. Nothing here made them easier or harder.

**Stage C (horizontal fusion) remains deferred**, and this sharpens its revisit
trigger: with the constant term at 2 and the per-parameter term at 1, a model
with hundreds of parameters pays almost all of its optimiser cost in `assign_`.
Stage B's Assign node addresses that directly; concatenating parameters does
not, and would still be the wrong lever.

**Regression-tested.** `tests/python/test_invariants.py::
test_an_optimizer_step_stays_within_its_submission_budget` pins `2 + N` for all
seven configurations, in a subprocess because the suite's eager fixture would
make the batching unobservable. Verified by restoring the per-parameter realise:
all seven turn red, and green again when reverted.

---

## 11. Backward had the same defect, in C++ (2026-08-02)

10 batched the optimiser. Attributing the step again immediately showed
`backward()` doing the same thing one layer down: **11 submissions per CIFAR
step**, five of them carrying a single dispatch.

Two causes, both the same shape as 10's and neither previously recorded.

### 11a. The leaf-deposit loop realised one gradient at a time

`backward()` ends by depositing each leaf's accumulated gradient, and called
`total.realize()` inside the loop. Every leaf is independent, so that is one
submission per parameter for work that could share one.

Built first, realised together, exactly as `Optimizer.step()` does. The realise
is **not** an optimisation and is not optional: it cuts the graph so that step
N's gradients do not keep step N's forward alive into step N+1. Doing it in one
call preserves that property exactly.

### 11b. Two backward rules ignored eager mode

`ops.cpp`'s `finish()` realises **only when `eager()`**, and every backward rule
built through it — except two. `MaxPool2d` and `Slice` construct their node by
hand, because their adjoints need a kernel rather than a composition, and both
called `realize()` **unconditionally**.

In eager mode that is indistinguishable from correct: everything realises
anyway. In lazy mode — which is what both examples train under — each one cut
the graph at the point it appeared. A CNN with three pooled blocks paid it three
times per backward pass.

> **The whole validation suite runs eager.** `conftest.py`'s autouse fixture
> forces it, so a failure names the operator that produced it. That is the right
> default and it meant 1,456 tests could not tell the two versions apart.

### Results

CIFAR-100 CNN, one step, submissions counted per phase (exact, clock-independent):

```
                   before    after 11a   after 11b
  backward           11          4           1
  step total         39         25          25
```

Whole step, best of 5 rounds of 20 steps:

```
                     before 10    after 10    after 11
  submissions/step      39           25          25
  step wall          13.57 ms     12.09 ms    11.71 ms
  host and driver     42.0%        35.7%       33.7%
  GPU / wall           0.58         0.64        0.66
```

The step-total count does not move at 11b because the submissions it removes
were inside `backward`, which had already been reduced to 4 by 11a — the two
overlap, and 11b's value is the wall time and the correctness of the rule rather
than a further count.

### Regression-tested, and the gap that let this exist

`tests/python/test_invariants.py::test_lazy_execution_gives_the_same_gradients_as_eager`
compares gradients **bit for bit** between the two modes, over `max_pool2d`,
`slice`, a conv chain and a reduction, **on both backends**. Bit-identity rather
than a tolerance because determinism is the project's hard invariant and
scheduling is not arithmetic.

The CPU arm is not a formality: autograd sits above the backend and does not
know which one it is driving, so a CPU-only build is a configuration where
nothing else covers lazy autograd, and three CI jobs build exactly that.

Verified by making the lazy path diverge on purpose: the `slice` case turns red
and reverting turns it green.

**What let the defect exist:** a rule that must hold for every backward rule was
enforced by a helper the two exceptional rules did not use, and nothing checked
the property directly. The test now checks the property.

---

## 12. Stage B's premise, re-checked — and most of it taken without stage B (2026-08-02)

10 and 11 left the optimiser at `2 + N` submissions, the `N` being one
`assign_` per parameter. 4 named the Assign node as the way to remove it, and 9
put two ADR-sized changes in front of that. **Both of those claims were re-run
before starting, and one of them is no longer true.**

### 9's finding 1 does not block anything any more

> *"`detach()` must stop forcing evaluation for batching to be worth anything."*

Measured today:

```
  detach on an UNREALISED node    1 submission
  detach on a REALISED node       0 submissions
```

The forcing is real and unchanged. What changed is that §10 **ordered around**
it — state is detached in pass 2, *after* the batched realise, so every
`detach()` in every optimiser now sees a realised node and costs nothing. The
finding was correct when written and is no longer a blocker. It stays worth
doing on its own merits; it is not on the path to anything.

### The remaining `N` needed a batched copy, not a graph node

`assign_` did not need to become lazy. It needed to stop being **one submission
per call**, and only the backend knows what a submission is. So
`Backend::copy_device_to_device` now takes a **span** of ranges, and
`vkml::assign(dst, src)` is the batched sibling of
`realize(std::span<const NodePtr>)` — same shape, same reason.

| | |
|---|---|
| **Benefit** | Removes the whole per-parameter term. No new OpKind, no aliasing analysis, no ownership rule, no change to when work happens |
| **Cost** | One more method on the `Backend` interface, and a second public entry point that does what `assign_` does |
| **Worthwhile when** | The cost is the submission rather than the scheduling — which is what §11's attribution says it is |
| **Not worthwhile when** | The assignments need to *participate in* a graph rather than merely happen together. That is still stage B, and this does not reach it |

The single-tensor `assign_` is now a call into the span form with a span of
one, so there is one implementation and the two cannot disagree. The span form
is the primitive **because only it can amortise a submission**: making the
single form the primitive and the batch a loop over it is precisely the
arrangement that cost 8 submissions per step.

Mixed batches work. An assignment needing the host-staged path — the
overlapping `t[0:5].assign_(t[2:7])` case — takes it for itself while the rest
still share a submission.

### Results

Assignment alone, the CIFAR CNN's eight parameters (1.14 MiB), minimum of 200
warm repeats:

```
                  submissions        ms
  one at a time        8         0.4994
  batched              1         0.0957 - 0.2208
```

The implied cost of a copy submission is **40–80 µs**, which agrees with the
~80 µs marginal host cost measured independently by regressing wall time on the
number of trivial realises.

Optimiser phase, minimum of 40 warm steps, and the **parameters are again
bit-identical** across all seven configurations:

```
                    submissions              ms
                  §11    §12        §11     §12
  SGD               9      2       0.675   0.552
  SGD momentum     10      3       1.134   1.109
  SGD nesterov     10      3       1.147   1.254
  RMSProp          10      3       1.255   1.241
  RMSProp cent+mom 10      3       1.790   1.963
  Adam             10      3       1.719   1.645
  AdamW            10      3       1.911   1.820
```

**The wall column does not move, and that is the honest reading.** The
optimiser phase is dominated by the two batched realises — GPU work plus the
wait for it — so a 0.3–0.4 ms saving sits inside a measurement whose run-to-run
spread on this machine is larger than that. The submission count is exact and
clock-independent, which is why it is the primary evidence and why the gate is
written against it.

The saving is visible where it is not buried:

```
  submissions per CIFAR step       15  ->  8
       upload 2 · backward 1 · optimiser 3 · item 2
  step wall                     11.71 ms  ->  10.07 ms
  host and driver                  33.7%  ->  24.0%
  GPU / wall                        0.66  ->  0.76
```

### What is left, and what stage B is now for

Eight submissions per step: 2 uploads, 1 backward, 3 optimiser, 2 for
`.item()`. **The optimiser is no longer the largest item.** Stage B's remaining
value is no longer the optimiser at all — it is `nn.BatchNorm2d`, which calls
`assign_` on the FORWARD pass of every layer and cannot batch across layers the
way an optimiser batches across parameters, because each layer's assignment is
separated by the next layer's arithmetic. That is the case only a graph node
fixes, and it should be the argument for stage B when it is written.

Stage C (horizontal fusion) stays deferred and its trigger is unchanged.

**Regression-tested.** The optimiser budget is now a **constant 3, independent
of the parameter count**, over eight parameters and all seven configurations —
a budget of the previous `2 + N` form would have passed every intermediate
version and still scaled with the model.
`test_batched_assign_is_one_submission` pins the primitive, and four more cover
correctness against the one-at-a-time path, all-or-nothing validation, length
mismatch and the overlapping fallback inside a batch, **on both backends**.
Verified by breaking two things: restoring one submission per copy turns eight
tests red, and validating lazily instead of up front turns the
all-or-nothing test red.
