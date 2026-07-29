# ADR 0006 — Lazy assign, and how work is batched into submissions

**Status:** accepted; **stage A implemented and measured (7 below)**, stages B and C
still proposed
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
