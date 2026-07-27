# CPU backend — baseline performance

**Date:** 2026-07-26
**Build:** `relwithdebinfo` (`-O2 -g`), GCC 14.2.1, single-threaded
**Machine:** Ryzen 5 4600H, Void Linux
**Reproduce:** `cmake --preset relwithdebinfo && ./build/relwithdebinfo/bin/vkml_bench`
**Machine-readable:** `vkml_bench --json`

This is the reference point every Vulkan kernel in M1 will be measured against.
Its purpose is *not* to look good — the CPU backend is the correctness oracle
and is deliberately unoptimised (ARCHITECTURE.md, M0 scope). It exists so that
M1 has a number to beat and, more importantly, a shape to compare against.

Methodology: warm-up is untimed, iteration count is auto-tuned to ~0.2 s, five
repeats, **minimum** reported (least polluted by scheduler noise on a
deterministic single-threaded workload). Mean is printed alongside so a wide
spread is visible.

---

## Results

```
category       benchmark                                         min           mean       throughput
----------------------------------------------------------------------------------------------------
memory         alloc 1 KiB                                   50.0 ns        51.5 ns                -
memory         alloc 1024 KiB                                89.3 ns        90.3 ns                -
memory         alloc 65536 KiB                              11.20 us       11.43 us                -
transfer       upload 1 MiB                                 23.22 us       23.64 us       45.17 GB/s
transfer       download 1 MiB                               23.38 us       23.43 us       44.85 GB/s
transfer       upload 16 MiB                                 1.40 ms        1.43 ms       11.99 GB/s
transfer       download 16 MiB                               1.44 ms        1.47 ms       11.66 GB/s
graph          build chain depth 10                          1.05 us        1.06 us                -
graph          build chain depth 100                        11.52 us       11.66 us                -
graph          build chain depth 1000                      116.75 us      117.05 us                -
graph          topo order depth 100                          6.93 us        6.99 us                -
graph          topo order depth 1000                        59.43 us       59.72 us                -
dispatch       realize 1 node (1x1)                         359.3 ns       360.6 ns                -
dispatch       realize 10 nodes (1x1)                        2.57 us        2.64 us                -
kernel         add 256x256                                  21.05 us       21.38 us  3112.81 Melem/s
kernel         relu 256x256                                 25.81 us       26.45 us  2539.17 Melem/s
kernel         exp 256x256                                 167.03 us      168.75 us   392.36 Melem/s
kernel         sum 256x256                                 507.53 us      508.39 us   129.13 Melem/s
kernel         softmax 256x256                               1.05 ms        1.06 ms    62.57 Melem/s
kernel         contiguous(T) 256x256                         1.02 ms        1.02 ms    64.38 Melem/s
kernel         add 1024x1024                               813.61 us      836.83 us  1288.80 Melem/s
kernel         relu 1024x1024                              705.67 us      731.78 us  1485.92 Melem/s
kernel         exp 1024x1024                                 2.68 ms        2.72 ms   390.64 Melem/s
kernel         sum 1024x1024                                 8.40 ms        8.44 ms   124.80 Melem/s
kernel         softmax 1024x1024                            16.84 ms       16.94 ms    62.26 Melem/s
kernel         contiguous(T) 1024x1024                      20.43 ms       20.97 ms    51.33 Melem/s
kernel         relu 1024x1024 strided                       21.90 ms       22.13 ms    47.87 Melem/s
matmul         sgemm 64^3                                  163.75 us      166.59 us     3.20 GFLOP/s
matmul         sgemm 128^3                                   2.08 ms        2.09 ms     2.01 GFLOP/s
matmul         sgemm 256^3                                  16.53 ms       16.56 ms     2.03 GFLOP/s
training       MLP 784-128-10 fwd (batch 64)                 7.55 ms        7.59 ms                -
training       MLP 784-128-10 fwd+bwd (batch 64)            14.63 ms       15.17 ms                -
```

---

## What these numbers say

### 1. The strided path costs 31x

`relu 1024x1024` runs at 1486 Melem/s contiguous and **47.9 Melem/s strided** —
a 31x gap. That is the price of the general `linear_to_offset` walk, which does
an integer division and multiply *per axis, per element*, against a flat
`o[i] = f(x[i])` loop on the fast path.

This is the single most important number for M1, for two reasons:

- It confirms the fast-path/slow-path split in `iterate.h` earns its keep, and
  that llama.cpp compiling separate `_aligned` shader variants is not
  over-engineering.
- The Vulkan kernels will face the *same* division, and a GPU integer divide is
  proportionally more expensive than a CPU one. Any elementwise shader should
  therefore ship a contiguous variant from the start rather than as an
  optimisation later.

`contiguous(T)` at 51 Melem/s is the same effect measured directly.

### 2. Reductions are the weakest kernel

`sum` at ~125 Melem/s and `softmax` at ~62 Melem/s are far below `add` at 1289.
Two causes, both known and both deliberate for M0:

- `reduce_generic` recomputes `linear_to_offset` per element with no contiguous
  fast path — the same 31x effect as above, applied to every reduction.
- pairwise summation recurses to blocks of 32, which is a correctness
  requirement (a sequential sum misses the 1e-5 gate by ~49x at K=4096; see
  `src/backend/cpu/reduce.h`) but costs call overhead.

Not worth fixing on CPU. Worth knowing before writing the Vulkan reduction,
where the tree structure is mandatory anyway because the target GPU has no
global float atomics.

### 3. MatMul is ~2 GFLOP/s, and that is the headline for M1

The naive triple loop with a pairwise inner reduction sustains **2.0 GFLOP/s**.

For contrast, measured on the *target GPU* through ggml's tuned Vulkan GEMM
(ARCHITECTURE.md §1.2): **4290 GFLOP/s fp32**. That is a **~2100x** gap. Even a
mediocre first Vulkan GEMM should be 100x faster than this, which is exactly why
ARCHITECTURE.md §8 puts GEMM at M3 with a "≥3 TFLOPS or stop and profile" gate
rather than treating it as ordinary work.

The 64³ case is faster per-FLOP (3.2 vs 2.0 GFLOP/s) because the whole working
set is 48 KiB and fits in L1/L2; beyond that it is bandwidth-bound.

### 4. Graph and dispatch overhead match the ADR-0001 measurements

- Graph construction: **~117 ns/node** (116.75 us / 1000)
- Topological order: **~59 ns/node**
- Dispatch (schedule + bind + kernel call, 1x1 tensor): **~360 ns/node**

ADR-0001 predicted ~410 ns/node build and ~460 ns/node traverse for the
shared_ptr model against ~20 ns for an arena. These numbers are lower because
this build is `-O2` rather than the ADR's benchmark configuration, but the
*ratio* that drove the decision is unchanged, and the conclusion stands: at
M0-M4 graph sizes this is well under 1 % of step time, and the fix is the M5
lowering rather than a change to the ownership model.

The ~360 ns dispatch figure is the one to watch in M1. On Vulkan it becomes
command-buffer recording plus a queue submit, and if per-node overhead does not
drop sharply once batched into one command buffer, that is a runtime bug rather
than a kernel one.

### 5. Transfers: 45 GB/s at 1 MiB, 12 GB/s at 16 MiB

The drop is cache residency — 1 MiB fits in L2/L3, 16 MiB does not, so the
larger case measures main-memory bandwidth.

Both are *host memcpy*, so they are an upper bound rather than a prediction.
On Vulkan these become staging copies across PCIe, bounded by roughly 8-16 GB/s
on this machine's link, and every transfer must go through a staging buffer
because the GPU exposes only 256 MiB of host-visible device-local memory
(measured, ARCHITECTURE.md §1.1). Expect Vulkan upload to land *below* the
16 MiB figure here, not above it.

### 6. Allocation is not a bottleneck

50 ns for 1 KiB, 89 ns for 1 MiB — glibc's allocator serving from its bins. The
11.2 us for 64 MiB is `mmap` plus first-touch page faults.

This matters as a contrast: a Vulkan `vkAllocateMemory` is on the order of tens
of microseconds and has a hard cap on the number of live allocations, which is
precisely why the M1 allocator must suballocate from large blocks rather than
allocating per tensor.

---

## Carried forward to M1

| Observation | Consequence for M1 |
|---|---|
| Strided path 31x slower | Ship contiguous variants of elementwise shaders from the start |
| Reductions weakest kernel | Tree reduction is mandatory anyway (no global float atomics); budget time for it |
| MatMul 2 GFLOP/s vs 4290 on GPU | Do not write GEMM until the runtime is proven — ARCHITECTURE.md §8, M3 |
| Dispatch ~360 ns/node | Per-node cost must fall once batched into one command buffer; if not, that is a runtime bug |
| Host transfer 12-45 GB/s | Vulkan staging will be slower; measure, do not assume |
| `vkAllocateMemory` is ~100x a malloc | Suballocate; never one allocation per tensor |

## Known non-goals

The CPU backend is frozen except for bug fixes. None of the above will be
optimised: multithreading, SIMD intrinsics, blocked GEMM and a contiguous
reduction fast path are all deliberately absent. Its job is to be obviously
correct and to keep being the oracle.
