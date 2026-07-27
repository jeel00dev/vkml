# ADR 0002 — Introduce a minimal Allocator abstraction now

**Status:** accepted
**Date:** 2026-07-26
**Covers:** review question 2.

---

## Context

`Storage` currently owns `{void* data, size_t nbytes, Device device, Deleter deleter}`,
and `make_cpu_storage()` is a free function that calls `std::aligned_alloc`.

The question is whether an explicit `Allocator` interface should exist before the CPU
backend is written, given the allocators the project will eventually want: CPU, Vulkan
device memory, pinned/staging host memory, arena, pool, and debug.

## Analysis

The existing deleter-closure design is genuinely flexible — any allocation strategy can
already produce a `Storage` by capturing its own context in the deleter. So the question
is not "is it possible without an interface" but "does an interface pay for itself".

**Argument against (premature abstraction):** M0 needs exactly one allocator. An interface
with one implementation is a liability, and the deleter closure already covers the
mechanism.

**Arguments for, which win:**

1. **Retrofitting touches every allocation site.** Allocation is called from tensor
   creation, every op that produces a new buffer, autograd's gradient buffers, and the
   optimizer. Introducing the seam after 30 kernels exist means editing all of them.

2. **M1 needs a *policy* object, not just a deleter.** A Vulkan allocator has to
   suballocate `VkDeviceMemory` in large blocks, keep a free list per memory type, and
   respect `maxMemoryAllocationSize` (measured at 4 GiB − 4 B on this GPU,
   ARCHITECTURE.md §1.1). That is stateful and long-lived. A deleter closure can free, but
   there is nowhere for the allocation *policy* to live. The interface is where it goes.

3. **The staging path in M1 needs two allocators on one device.** This GPU exposes only
   256 MiB of host-visible device-local memory (no resizable BAR), so uploads must go
   through host-coherent staging. That means a Vulkan `Device` owns a *device* allocator
   and a *staging* allocator simultaneously. Selecting between them is naturally
   "which allocator do I ask", and awkward any other way.

4. **It makes the allocation count testable per-allocator**, which the leak tests in
   ARCHITECTURE.md §7.4 tier 8 want.

**Where the line is drawn.** The interface stays minimal — allocate, name, device. It
deliberately does *not* model alignment policies, memory kinds, streams, or async
free lists. Those are speculative, and each can be added to a concrete allocator without
touching the interface. This mirrors ggml's `ggml_backend_buffer_type`, which is likewise
a thin "ask this thing for memory" seam rather than a general allocator framework — and
which exists in ggml for precisely reason 3, so that host-pinned memory usable by a GPU
can be requested independently of who computes on it.

**What is deliberately *not* added now:**

- **`DebugAllocator`** (poisoning, guard regions). ASan already covers heap overflow and
  use-after-free better than a hand-rolled version would, and the asan preset is wired up
  and green. Revisit only if a class of bug appears that ASan cannot see — most likely
  candidate is logical overrun *within* a correctly-sized buffer caused by bad stride
  arithmetic, which guard regions would not catch either.
- **Pool / arena allocators.** The graph memory planner at M5 solves the same problem
  better for the case that matters (intermediate activations), by assigning offsets inside
  one large buffer. A general pool would duplicate that.

## Decision

Add `vkml::Allocator` in `core` with a single concrete implementation, `CpuAllocator`,
plus a process-wide `cpu_allocator()` accessor.

`Storage` is **unchanged**. Its deleter-closure constructor remains public and is the path
for *foreign* memory — importing a NumPy array or a DLPack capsule without copying, where
the deallocation is someone else's business. Allocators are a factory for owned memory;
they are not the only way a `Storage` can come to exist.

`make_cpu_storage()` is kept as a thin delegate to `cpu_allocator()`, so no existing call
site or test changes.

## Consequences

- One virtual call per allocation. Irrelevant: allocations are per-tensor, not per-element.
- The seam exists before the kernels do, so M1 adds `VulkanAllocator` and
  `VulkanStagingAllocator` without editing anything above `backend/`.
- `Allocator` lives in `core` (layer 1), so `backend/vulkan` (layer 4) can implement it
  without inverting the layering — the same dependency inversion that lets `Storage` hold
  a deleter it knows nothing about.
