# ADR 0012 — Asynchronous submission, and the four hazards it opens

**Status:** accepted, implemented and measured.
**Date:** 2026-08-02
**Covers:** `Recorder`'s command-buffer ring, `VulkanBackend::compute`'s submission, the
deferred-free queue in `vk::Allocator`, and the invariant `dispatch/executor.h` states.
**Hardware:** AMD RX 5600M (RDNA1), RADV. Every measurement below was run on it, with the
validation layers **off** (`MEASUREMENT-AUDIT.md` §6c — they are on by default and cost 45%
of an MNIST step).

---

## 1. The measurement this exists for

An MNIST MLP step at batch 64 costs **1189 µs** against an arithmetic floor of 5.7 µs. It is
not a kernel problem: sweeping a dependent chain of trivial kernels and regressing wall time
against the node count separates the two costs cleanly, because the fixed cost cancels in the
slope.

```
  per submission (fixed)   66.32 us wall,  3.31 us GPU
  per node (marginal)       4.83 us wall,  5.40 us GPU
```

A step makes **seven** submissions, so ~460 µs of it is submission overhead. Splitting that
66 µs at the point `submit()` returns, with no timestamps involved:

```
  host per dispatch (record + barrier)     0.24 us
  host per submission (vkQueueSubmit2)    ~16.5 us
  blocking on completion                  ~40.5 us   of which ~8 us is real GPU work
```

**Two thirds of a submission is the host asleep in `vkWaitSemaphores`.** An idle
`synchronize()` costs 1.95 µs, so the wait *path* is not the cost — the wake-up is. The GPU
finishes, raises an interrupt, and the scheduler puts the thread back on a core; that round
trip is ~32 µs and belongs to the OS, not to vkML or to Vulkan.

`Recorder` holds **one** command buffer, so `begin()` must wait for the previous submission
before it can reset it. Host and GPU therefore never overlap, by construction.

## 2. The alternatives, prototyped before anything was rewritten

Raw Vulkan, one trivial dispatch per submission, minimum of three independent process
launches. The point of prototyping first was to learn the ceiling before paying for the
design:

| model | µs/submission | against today |
|---|---|---|
| one command buffer, submit + wait (today) | 59.3 | — |
| ring of 2, no wait | 36.3 | 1.6× |
| **ring of 4, no wait** | **17.7** | **3.4×** |
| ring of 8, no wait | 17.4 | 3.4× |
| record once, re-submit the same buffer | 15.2 | 3.9× |

**A ring of 2 is not enough.** The host comes back round to a buffer still in flight and
blocks anyway; the depth has to exceed the number of submissions the GPU can be behind. Four
saturates and eight adds nothing, so **four** is the choice and it is measured rather than
picked.

**Replay is rejected for now, with its number.** Re-recording a one-dispatch buffer costs
~2 µs against a ~17 µs submission, so replaying buys 1.14× on top of the ring. It would cost
fixed buffer addresses across steps and an invalidation rule for when the graph changes —
real complexity, for a saving that only becomes interesting once submissions are batched
enough that recording is a meaningful share of one. Revisit when a submission carries enough
dispatches that its record time approaches 17 µs; at 0.24 µs each that is ~70 dispatches.

## 3. The decision

1. `Recorder` owns a **ring of four command buffers**, each with its own timeline value and
   its own query range. `begin()` waits only for **that slot's** previous submission.
2. `VulkanBackend::compute()` **submits and returns**. It does not wait.
3. Anything that reads device memory on the host waits first: `copy_to_host`, and therefore
   `.item()` and `.numpy()`. `synchronize()` waits for everything.
4. `vk::Allocator::free` **defers** while submissions are outstanding.

The invariant that changes is stated plainly, because `dispatch/executor.h` documented the old
one: **`realize()` no longer implies the results exist in memory.** It implies they have been
*ordered*. The only thing that implies existence is a host read or `synchronize()`.

## 4. The four hazards, and why each is closed

vkML's synchronisation **cannot be machine-checked** — it addresses buffers through
`bufferDeviceAddress`, so the validation layer sees no resource access and reports nothing
(ADR 0006 §8). Every argument here is therefore made on the specification and held by
construction, which is the standard this ADR has to meet rather than a caveat on it.

### 4a. Within a submission — unchanged

A global memory barrier follows every dispatch. Conservative and already in place.

### 4b. Between submissions on one queue

Two dispatches in different submissions may touch the same memory, and nothing in the *old*
design had to think about it because the host waited in between.

The guarantee relied on is **submission order**: batches submitted to one queue are ordered,
and a pipeline barrier's first synchronisation scope includes every command submitted earlier
in submission order — *including commands in earlier batches* — while its second scope
includes everything submitted later. So one barrier makes every prior write visible to every
subsequent read on that queue.

vkML's compute path already ends each submission with a barrier, because the loop emits
`dispatch; barrier;` per node. **That is not relied upon.** `copy_device_to_device` ends with
copies and no trailing barrier, so the property held by accident of one code path rather than
by design. `Recorder::begin()` therefore emits a **leading barrier** whenever a previous
submission is still outstanding.

The trade is explicit: **one extra barrier per submission, ~2.4 µs**, against ~42 µs saved by
not blocking. It buys the property locally, in the one function that knows whether anything is
in flight, rather than as a claim about every caller's last recorded command.

### 4c. Host reads

Handled at the two functions that perform them. `StagingBuffer::download` already waits; it
now waits for **everything outstanding** rather than only for its own copy, because the data
it is about to read may be produced by a submission that has not been waited on.

### 4b bis. And it cannot be verified on this machine

Stated plainly because it is the weakest link in the chain, not a footnote.

The leading barrier was **removed on purpose** and the whole suite re-run in lazy mode: **1550
tests passed.** A probe built specifically to provoke the hazard — realise a long chain, drop
its output while in flight, reallocate the same size, read it back — found **zero corrupted
elements in 20 attempts**, with the barrier removed *and* the retirement queue removed.

The reason is that RADV serialises submissions to one queue in practice, so nothing here
overlaps and no test can tell correct synchronisation from none. That is **driver behaviour,
not a Vulkan guarantee**, and CLAUDE.md's standing rule applies: *this machine's limits are not
the contract*. The barrier stays because the specification requires it, and the first thing
able to falsify any of this is a driver that overlaps submissions.

What *is* verified, with a control that fails:
`tests/cpp/test_vulkan_device.cpp` submits three times round the ring without waiting; removing
the per-slot query partitioning makes it fail with `VK_ERROR_DEVICE_LOST`, because a command
buffer then resets query slots another in-flight submission is still writing.

### 4d. Memory reuse — the one that was not obvious

Under the synchronous model a `Storage` could free its allocation the instant its last
reference died, because nothing was ever in flight. Asynchronously, this sequence is a
GPU-side use-after-free:

```
  realise a graph writing tensor T        submitted, still running
  Python drops T                          Storage dies -> Allocator::free
  allocate a new tensor U                 the free list hands back T's range
  realise a graph writing U               U's dispatch and T's dispatch share memory
```

Nothing would fault. The GPU would write two tensors into one range and the results would be
wrong intermittently and unreproducibly — the worst failure mode this project can have, and
the exact reason bit-reproducibility is its acceptance criterion.

`Allocator::free` therefore pushes onto a **retirement queue** stamped with the timeline value
at the moment of the free, and a block returns to the free list only once the GPU has passed
it. Stamping with "now" is conservative and correct without tracking which submissions read
which allocation: any submission that could reference this memory was submitted at or before
the free, so waiting for the current value covers all of them.

Retirement is drained on every allocate, on every free, on `synchronize()` and whenever the
stats are read, using `vkGetSemaphoreCounterValue` — a poll, never a block, measured at
**1.11 µs** against an allocate-and-free cycle of 0.06 µs. Memory is returned no later than the
next allocation after the GPU catches up, and the queue cannot grow without bound while work is
being submitted.

**Honest status of this mechanism.** Given §4b's leading barrier, submission N+1's work cannot
start until N's has finished, so the reuse race is *already* closed by the barrier and the
retirement queue is redundant **today**. It is kept, rather than deleted as unearned
complexity, for one specific reason: §6 names selective barriers as the next optimisation, and
whoever does that work will be reasoning about *data* dependencies between nodes. They would
have no reason to think about *allocation* identity, and the failure they would reintroduce is
silent, unreproducible, and destroys bit-reproducibility — the project's defining guarantee.
The cost of keeping it is ~45 lines and a 1.11 µs poll; the cost of the alternative is a class
of bug this project cannot detect. `test_deferred_frees_are_all_returned` guards it, and fails
when the drain is disabled.

## 5. What it costs

| | |
|---|---|
| **Benefit** | Measured below: 3.4× per submission, 1.35× on an MNIST step |
| **Cost** | Four command buffers instead of one; a retirement queue; one extra barrier per submission (~2.4 µs); peak memory rises because frees lag the GPU |
| **Worthwhile when** | Per-step GPU time is under ~100 µs, which is where the constant dominates — every small model, every inference step, every workload at low batch |
| **Not worthwhile when** | One submission carries enough arithmetic to hide the round trip. At CIFAR-100's 8 ms step this is under 1% and the complexity would not be earned |

### Measured, after

`python bench/latency_bench.py`, validation off, minimum of 40:

```
                                 before      after
  per submission (fixed)       66.32 us   19.52 us     3.4x
  per node (marginal, wall)     4.83 us    2.40 us
  per node (marginal, GPU)      5.40 us    5.35 us     unchanged, as expected

  MNIST MLP 784-128-10, b=64     1189 us     883 us     1.35x
```

The projection made **before** the work was ~340 µs of 1189, "a 1.4× step". It came out at
306 µs and 1.35×.

Two things worth reading off the table. The per-node GPU cost did not move, which is right —
nothing about the barriers between nodes changed, and a figure that had moved would mean the
measurement was wrong. And in the sweep, `GPU/wall` now **exceeds 1** from eight nodes upward
(1.75 at thirty-two): the host returns before the GPU has finished, which under the old model
was arithmetically impossible. That ratio is the clearest single check that the change is real.

The failure mode this introduces, stated so it is recognisable: an error inside a shader is no
longer attributable to the `realize()` that produced it, because that call returned before the
work ran. `VKML_EAGER=1` remains the answer and now has a second reason to exist — and the
executor makes it a genuine one by synchronising after each eager realise.

**A consequence for every profiling consumer**: `vulkan_last_profile()` was implicitly valid
after `realize()`, because `realize()` waited. It is not any more. Reading it without waiting
returns the *previous* submission's intervals. This bit `bench/latency_bench.py` during this
very change and understated the GPU column by 3×; the docstring on `measure_gpu` now says why
the `synchronize()` there is load-bearing rather than cosmetic.

## 6. What is deliberately not done

- **Multiple queues.** One queue keeps submission order as the whole ordering argument. A
  transfer queue would need real semaphore dependencies and is a separate decision.
- **Thread safety.** `Recorder` was not thread-safe and still is not. The ring makes the
  *device* concurrent, not the recorder. DataLoader prefetch (#22) still needs the threading
  contract recorded in `python/vkml/data.py`.
- **Per-buffer hazard tracking.** The global barrier stays. llama.cpp's range-overlap approach
  (`overlaps_unsynced` in `ggml-vulkan.cpp`) would let most barriers be elided and is worth
  ~2.4 µs per node — a separate ADR, and it needs aliasing information the executor does not
  yet produce.
- **Command-buffer replay.** §2 rejects it with its number.
