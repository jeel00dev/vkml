# vkml — Architecture Study & Recommendation

**Status:** design proposal, pre-implementation. Nothing has been built yet.
**Date:** 2026-07-26
**Target machine:** AMD Radeon RX 5600M (Navi 10 / RDNA1), RADV / Mesa 26.1.5, Void Linux.

This document is the deliverable for stages 1–6 of the agreed development strategy:
explore the reference projects, explain their architecture, compare design choices,
recommend an architecture, explain the trade-offs, and produce a roadmap.

Everything numeric in §1 was **measured on this machine**, not quoted from spec sheets.

---

## 0. Executive summary — the recommendation in one page

Build **vkml** as a *deferred-execution graph framework with an eager-feeling Python API*,
not as an eager op-by-op dispatcher.

Six decisions carry the whole design:

| # | Decision | Source of the idea | Why |
|---|---|---|---|
| 1 | **Backward rules build graph nodes, never call kernels** | ggml `ggml_compute_backward`, tinygrad | Cuts the kernel count from ~120 to ~64. Backward reuses forward kernels. Higher-order derivatives and checkpointing fall out free. |
| 2 | **Offline graph memory planner** (simulate refcounts → assign offsets in one buffer) | ggml `ggml-alloc.c` | On 5.75 GB this is the difference between training and OOM. Zero allocation per training step. |
| 3 | **Optimizer step is a graph node** | ggml `ggml_opt_step_adamw` | Whole fwd+bwd+update becomes one Vulkan submit, zero CPU↔GPU sync per step. Critical: this box has only 256 MiB of host-visible VRAM. |
| 4 | **Buffer Device Address instead of descriptor sets** | *not* from the references — enabled by targeting one modern device | Deletes descriptor pools/sets/updates entirely (~500 lines of the most error-prone Vulkan code, plus per-dispatch CPU cost). Verified available. |
| 5 | **No global float atomics anywhere; deterministic tree reductions only** | forced by hardware, wanted anyway | `shaderBufferFloat32AtomicAdd = false` on this GPU (measured). Determinism also makes the PyTorch-parity suite reproducible instead of flaky, and keeps fp32 error inside the 1e-5 tolerance. |
| 6 | **CPU backend is written first and is the correctness oracle** | ggml `test-backend-ops` | Every Vulkan kernel is validated against CPU, and CPU is validated against PyTorch. Two cheap comparisons instead of one expensive one. |

The API stays PyTorch-shaped. The *execution model* is closer to `tinygrad` + `ggml` than
to PyTorch, and that is deliberate: PyTorch's eager model is affordable because it is
backed by cuBLAS/cuDNN and a caching allocator on 24 GB cards. Neither applies here.

---

## 1. Hardware ground truth (measured on this machine)

Everything in this section came from `vulkaninfo`, `ggml`'s device probe, and a
micro-benchmark I built against the local `libggml-vulkan.so`.

### 1.1 Device capabilities

| Property | Value | Consequence for the design |
|---|---|---|
| Device | RX 5600M, `RADV NAVI10`, RDNA1, `deviceID 0x731f` | llama.cpp has an explicit `AMD_RDNA1` tuning path — we should too |
| Vulkan | instance 1.4.341, device 1.4.354 | We may use Vulkan 1.3 core features freely |
| Device-local VRAM | **5.75 GiB** (heap 0) | Total budget for weights + grads + optimizer state + activations |
| Host-visible device-local | **256 MiB** (heap 2) | **No resizable BAR.** Staging-buffer uploads are mandatory; cannot map VRAM |
| Host-visible host-coherent | 3.57 GiB (heap 1) | Staging pool lives here |
| Max single allocation | 4 GiB − 4 B | Suballocate; never one buffer for everything |
| `maxComputeSharedMemorySize` | 65536 (64 KB LDS) | GEMM block tiles up to ~48 KB are viable; >32 KB costs occupancy |
| `maxComputeWorkGroupInvocations` | 1024 | Workgroups of 256 are the sweet spot |
| Subgroup size | default **32**, controllable 32–64 | RDNA1 wave32/wave64 selectable per pipeline |
| `subgroupSizeControl` / `computeFullSubgroups` | true / true | We can pin wave size per shader — llama.cpp does exactly this for RDNA1 |
| fp16 arithmetic / 16-bit storage | **true / true** | fp16 supported end to end |
| bf16 | **false** | bf16 is off the table on this GPU |
| **Cooperative matrix (tensor cores)** | **absent** | Hand-tiled GEMM is the only option. No `VK_KHR_cooperative_matrix` |
| `shaderBufferFloat32AtomicAdd` | **false** | **Global float atomicAdd is unavailable** — drives the whole backward design |
| `shaderSharedFloat32AtomicAdd` | **true** | LDS float accumulation *is* available (intra-workgroup only) |
| `bufferDeviceAddress` | **true** | Descriptor-set-free compute is possible |
| `scalarBlockLayout` | true | C-struct-identical layouts in SSBOs; no std140 padding games |
| `timelineSemaphore`, `synchronization2`, `maintenance4` | true | Modern, simple sync; spec-constant workgroup sizes |
| `maxPushConstantsSize` | 256 bytes | Sets the shape/stride descriptor budget — see §5.3 |
| `timestampPeriod` | 10 ns | GPU profiling via query pools is accurate enough |

Toolchain present: `glslc` (shaderc 2026.2, glslang 16.3, SPIRV-Tools 2026.1), GCC 14.2.1,
CMake 4.2.2, Python 3.14.6, PyTorch 2.12.1 (CPU), NumPy 2.4.4, pytest 9.0.3.
Missing and needed: `ninja`, and a binding library (`pybind11` 3.0.4 or `nanobind` 2.13.0 —
both support Python 3.14).

### 1.2 Measured performance ceiling

Benchmarked through ggml's Vulkan backend, which is a mature hand-tiled GEMM — so these
numbers are **the bar a good implementation should approach**, not a hardware peak.

```
-- square GEMM --                              -- small N (memory bound) --
f32  512³     0.580 ms    463 GFLOPS           f32  M=K=4096, N=1    185 GB/s
f32  1024³    0.675 ms   3183 GFLOPS           f32  M=K=4096, N=4    165 GB/s
f32  2048³    4.043 ms   4249 GFLOPS           f32  M=K=4096, N=16   102 GB/s
f32  4096³   32.066 ms   4286 GFLOPS           f16  M=K=4096, N=1    130 GB/s
f16  1024³    0.515 ms   4170 GFLOPS
f16  2048³    3.235 ms   5311 GFLOPS
f16  4096³   23.954 ms   5738 GFLOPS
```

Theoretical peak: 2304 ALUs × 2 flop × ~1.265 GHz boost ≈ **5.8 TFLOPS fp32**;
192-bit GDDR6 @ 12 Gbps ≈ **288 GB/s**.

Three conclusions that shape the roadmap:

1. **A well-tuned Vulkan GEMM reaches ~74 % of fp32 peak on this GPU.** That is the target
   for M3. A naive one-thread-per-output-element kernel will land near 5 %; the gap is
   entirely tiling and is why §5.4 specifies the tile structure up front.

2. **fp16 buys only ~1.34×, not 2×.** So mixed precision on this machine is primarily a
   *memory capacity* win (half the bytes for weights and activations in a 5.75 GB budget),
   not a throughput win. That demotes it in the roadmap from "performance feature" to
   "capacity feature" — worth doing, but after correctness, and justified by VRAM not FLOPS.

3. **Small-batch work is bandwidth-bound and tops out near 185 GB/s (~64 % of peak BW).**
   Training a small chess-eval net at batch 1–16 will be bandwidth-limited, not FLOP-limited.
   Fusion (see §4.2) matters more than GEMM tuning for that workload.

> One caution about reading ggml's own benchmarks: `test-backend-ops` reports 3.6 GFLOPS for
> `f32×f32, m=4096, n=1, k=14336`. That is **not** a hardware limit — my own benchmark of the
> same op at `k=4096` gets 185 GB/s, near bandwidth. It is an unoptimised shape in llama.cpp's
> dispatch (LLM inference never uses f32 weights). Do not infer a hardware ceiling from it.

---

## 2. What the reference projects actually do

### 2.1 ggml — the primary model

**Tensor representation** (`ggml/include/ggml.h:657`). A flat POD struct, 4 dimensions:

```c
struct ggml_tensor {
    enum ggml_type type;
    struct ggml_backend_buffer * buffer;
    int64_t ne[4];          // element counts
    size_t  nb[4];          // byte strides — nb[0] = type size, nb[i] = nb[i-1]*ne[i-1]
    enum ggml_op op;
    int32_t op_params[16];  // inline op attributes, no heap allocation
    int32_t flags;          // INPUT | OUTPUT | PARAM | LOSS | COMPUTE
    struct ggml_tensor * src[10];
    struct ggml_tensor * view_src; size_t view_offs;
    void * data;
    char name[64];
    void * extra;
};
```

Why it looks like this:

- **`ne` + `nb` (separate strides in *bytes*)** means transpose/permute/reshape/slice are
  metadata-only. No copy. Every kernel that honours `nb` handles arbitrary views for free.
- **`op_params` as a fixed inline array** means a tensor node is a fixed-size POD. That in
  turn means the whole graph can live in a bump-allocated arena with no per-node `malloc`.
- **`view_src` / `view_offs`** exist purely so the *allocator* knows a buffer is still live
  when only a view of it is referenced. It is a lifetime-tracking field, not a layout field.
- **`flags`** carry training semantics (`PARAM`, `LOSS`) directly on the tensor, which is how
  the backward builder knows where to stop.

**Graph** (`ggml/src/ggml-impl.h:329`). `ggml_cgraph` is an array of node pointers in
topological order plus a hash set of visited tensors, `grads[]`, `grad_accs[]`, and
`use_counts[]`. It is built by `ggml_build_forward_expand(gf, result)` doing a DFS from the
output. The graph is a *plan*, built fresh each iteration but structurally identical each time.

**Autograd** (`ggml/src/ggml.c`, `ggml_compute_backward`). This is the single most important
idea in the codebase for us. It is a `switch` over `tensor->op` where each case **builds more
forward nodes**:

```c
case GGML_OP_MUL: {
    if (src0_needs_grads) ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, src1));
    if (src1_needs_grads) {
        struct ggml_tensor * tmp = ggml_mul(ctx, src0, grad);
        if (!ggml_are_same_shape(src0, src1)) tmp = ggml_repeat_back(ctx, tmp, src1);
        ggml_add_or_set(ctx, cgraph, isrc1, tmp);
    }
} break;
```

No `mul_backward` kernel exists. The backward pass is *the same executor running more of the
same ops*. Broadcast backward is `repeat_back`, i.e. a sum. Everything downstream — the
allocator, the scheduler, the backend — treats forward and backward nodes identically.

**Training** (`ggml/src/ggml-opt.cpp`). Three graphs: `gf` (forward), `gb_grad`
(forward+backward), `gb_opt` (forward+backward+optimizer). The Adam update is
`ggml_opt_step_adamw(ctx, param, grad, m, v, adamw_params)` — **a node in the graph**, with
`m` and `v` as persistent tensors. The consequence is that an entire training step is one
graph submission with no host round-trip.

**Memory** — two distinct layers, and conflating them is a common mistake:

- `ggml_dyn_tallocr` (`ggml-alloc.c:120`): an *offset* allocator. Sorted free-block array
  (max 256 blocks), best-fit search skipping the last block, then a last-block pass with a
  "reuse factor" heuristic, coalescing on free. It allocates *addresses*, not memory.
- `ggml_gallocr` (`ggml-alloc.c:488`): walks the graph once counting `n_children` and
  `n_views` per tensor, then simulates execution — allocating a node's output before it runs
  and freeing a source when its refcount hits zero. The peak offset observed becomes the size
  of **one** real buffer. `ggml_gallocr_alloc_graph` then just re-stamps the cached offsets if
  the graph topology is unchanged.

Plus an in-place optimisation (`ggml-alloc.c:631`): if an op is in `ggml_op_can_inplace()`,
a parent has exactly one child and no views, and layouts match, the output reuses the
parent's address outright.

This works **only because ggml's graphs are static**. Inference re-runs the same graph per
token; training re-runs the same graph per step. That assumption holds for us too.

**Backend abstraction** (`ggml/src/ggml-backend-impl.h`). Four nested vtable structs:
`ggml_backend_reg` (enumerate devices) → `ggml_backend_device` (properties, `supports_op`,
buffer types) → `ggml_backend` (a stream: `graph_compute`, async copies, events) →
`ggml_backend_buffer_type` / `ggml_backend_buffer` (allocation and tensor I/O).

The separation of `buffer_type` from `backend` is subtle and correct: it lets a tensor's
*storage* be described independently of *who computes on it* (pinned host memory usable by
the GPU, unified memory, etc.).

`ggml_backend_sched` (`ggml-backend.cpp:774`) assigns each node a backend, cuts the graph into
`ggml_backend_sched_split`s at backend boundaries, and inserts copies for split inputs. This
is how ops a backend doesn't implement transparently fall back to CPU.

### 2.2 llama.cpp's Vulkan backend — the implementation reference

16,944 lines in one file, ~180 GLSL shaders. What matters:

**Shader variant generation** (`vulkan-shaders/vulkan-shaders-gen.cpp`, 1205 lines). A
build-time C++ program that invokes `glslc` with different `-D` defines to produce thousands
of SPIR-V blobs (per dtype × per tile size × aligned/unaligned × coopmat/scalar).

**GEMM structure** (`vulkan-shaders/mul_mm.comp`). Classic three-level tiling, parameterised
by *specialisation constants* rather than defines where possible:

```glsl
layout (constant_id = 1) const uint BM = 64;   // block tile M   (LDS)
layout (constant_id = 2) const uint BN = 64;   // block tile N   (LDS)
layout (constant_id = 3) const uint BK = 16;   // block tile K
layout (constant_id = 4) const uint WM = 32;   // warp tile M
layout (constant_id = 5) const uint WN = 32;   // warp tile N
layout (constant_id = 7) const uint TM = 4;    // thread tile M  (registers)
layout (constant_id = 8) const uint TN = 2;    // thread tile N  (registers)
shared FLOAT_TYPEV2 buf_a[BM * SHMEM_STRIDE];  // SHMEM_STRIDE = BK/2 + 1  ← bank-conflict pad
shared FLOAT_TYPEV2 buf_b[BN * SHMEM_STRIDE];
```

Global → LDS → registers, with three tile sizes (`s`/`m`/`l`) selected at runtime by matrix
shape, and `align`ed variants that skip bounds checks. The `+1` on `SHMEM_STRIDE` is LDS
bank-conflict padding.

**RDNA1 is a first-class tuning target** (`ggml-vulkan.cpp:3190`) — directly relevant:

```cpp
static const std::unordered_map<std::string, uint32_t> rdna1_pipelines = {
    {"soft_max", 64}, {"im2col", 64}, {"argmax", 64}, {"mul_mat_vec", 64},
    {"mul_mat_vec_f16", 32}, {"mul_mat_vec_f32_f16", 32}
};
static constexpr uint32_t RDNA_DEFAULT_SUBGROUP_SIZE = 32;
```

Per-shader subgroup size, wave32 default, wave64 for reductions. Detection keys off
`wavefrontsPerSimd == 20` from `VK_AMD_shader_core_properties`. **This map is empirical tuning
data for our exact GPU** and is worth treating as a starting point rather than rediscovering.

**Synchronisation** (`ggml-vulkan.cpp:2852`). One global `vkCmdPipelineBarrier` with
`shaderRead|shaderWrite|transferRead|transferWrite` on both sides between dependent dispatches.
No per-buffer barriers. Blunt, correct, and cheap enough — a lesson in not over-engineering.

**Descriptor sets** (`ggml-vulkan.cpp:2360`). Pools of 128 sets, grown 1.5×, handed out by a
monotonic index that resets per graph. This is ~200 lines of machinery that **we can delete
entirely** by using buffer device address (§5.3).

**Submission batching** (`ggml-vulkan.cpp:14509`). Submit every 100 nodes *or* every ~100 MB
of estimated matmul traffic, whichever comes first, with the threshold adapted from the
previous graph's total. Balances GPU idle time against submission overhead.

### 2.3 stable-diffusion.cpp — the layer library and the VRAM problem

Runs SDXL (6.5 GB of weights) on this 6 GB card. Its notes record ~5 s/step at 1024², with
"RADV spills slightly to system RAM". Two things to take:

**`GGMLBlock`** (`src/core/ggml_extend.hpp:3211`) is `nn.Module` rebuilt on ggml:

```cpp
class GGMLBlock {
protected:
    std::unordered_map<std::string, std::shared_ptr<GGMLBlock>> blocks;
    std::unordered_map<std::string, ggml_tensor*> params;
    void init_blocks(ggml_context*, const String2TensorStorage&, const std::string prefix);
    virtual void init_params(ggml_context*, const String2TensorStorage&, const std::string prefix) {}
public:
    void init(ggml_context* ctx, const String2TensorStorage& m, std::string prefix);
    size_t get_params_num(); size_t get_params_mem_size();
};
```

Named children + named params + recursive init with a dotted prefix, so parameter names match
`state_dict` keys. `Linear`, `Conv2d`, `LayerNorm`, `GroupNorm`, `RMSNorm`, `MultiheadAttention`
derive from it. This is the right shape for our `nn` layer.

**Params/compute split** (`GGMLRunner`, `ggml_extend.hpp:1731`): `params_ctx` holds persistent
weights; `compute_ctx` + `compute_allocr` (a `ggml_gallocr`) holds the per-forward graph.
This separation is exactly what we need for training and is worth copying verbatim in spirit.

**`ggml_graph_cut`** (`src/core/ggml_graph_cut.cpp`, 1013 lines) cuts a graph into `Segment`s
with `SegmentResidency::{STREAMED, RESIDENT}` and per-segment `compute_buffer_size`, so weights
can be streamed in and out when the model exceeds VRAM. Relevant much later, if ever; noted
as prior art for the "model doesn't fit" case.

**What not to copy:** `GGMLRunner` is a ~500-line class inside a 4123-line header, carrying
graph-cut plans, weight adapters, multi-device scheduling, and caching in one object. The
ideas are good; the packaging is accreted. Take the ideas, not the file.

### 2.4 tinygrad — the API and autograd model

Not present on this machine; assessed from knowledge of its design.

The relevant lesson is its **lazy + realize** model. Tensor ops build a `LazyBuffer` DAG;
nothing executes until `.numpy()`, `.item()`, or an explicit `.realize()`. The scheduler then
sees a whole subgraph at once and can fuse elementwise chains into single kernels. `TinyJit`
captures a repeated function (a training step) and replays it as a fixed kernel sequence,
skipping all Python overhead on subsequent calls.

Its autograd is small because — as in ggml — backward functions are written in terms of
forward tensor ops, not kernels. Its optimizers (`SGD`, `Adam`) are ~20 lines each of tensor
arithmetic, and because they are tensor ops they get captured into the JIT alongside the model.

The API lesson: **users experience it as eager.** `x = a + b` returns a Tensor immediately.
Laziness is an implementation detail that only surfaces at realize boundaries. That is
precisely the trick that lets us have a PyTorch-shaped API on a graph engine.

### 2.5 PyTorch — the user experience only

Studied at the layer level, as agreed:

- **`Tensor` / `TensorImpl`** — a handle holding a `Storage` (refcounted bytes) plus
  `sizes`, `strides`, `storage_offset`, `dtype`, `device`. Views share storage. Our tensor
  should be this, plus ggml's `nb`-in-bytes convention.
- **Dispatcher** — `DispatchKeySet` bitsets route `aten::add` to CPU/CUDA/Autograd/Autocast
  implementations, with boxed fallback kernels and per-key fallthrough. This is built to
  support ~20 backends × ~2000 ops × autograd/autocast/functorch layering. **For two backends
  and ~64 ops this is enormous over-engineering** — a `std::array<OpImpl, N_OPS>` per backend
  is the right size.
- **Autograd** — a tape of `Node`s recorded during forward, each with `apply()` calling
  dedicated `*_backward` ATen kernels. Powerful for dynamic control flow; the cost is roughly
  doubling the kernel count. We take the *interface* (`.backward()`, `.grad`, `requires_grad`,
  `no_grad()`) and reject the *implementation*.
- **`nn.Module`** — `_parameters`, `_buffers`, `_modules` ordered dicts; recursive
  `named_parameters()`, `state_dict()`, `.to(device)`. Copy this wholesale; it is the API
  users know.

---

## 3. Design comparison — the five real forks

### Fork 1: execution model

| | Eager (PyTorch) | Static graph (ggml) | Lazy + realize (tinygrad) |
|---|---|---|---|
| Kernel launch | immediate, per op | batched, whole graph | batched at realize points |
| Memory planning | caching allocator, runtime | offline, provably minimal | scheduler-driven |
| Fusion | none (needs `torch.compile`) | manual op design | automatic elementwise fusion |
| Debugging | trivial | hard (nothing runs until compute) | medium |
| Dynamic control flow | native | needs graph rebuild | needs rebuild / JIT invalidation |
| Python overhead per step | high | near zero | near zero once JIT'd |
| Fits 5.75 GB? | poorly | very well | well |

**Recommendation: lazy + realize, with an eager debug mode.**

Rationale. The 5.75 GB budget makes offline memory planning close to mandatory — ggml's
approach yields the minimum possible peak for a given graph, and a caching allocator cannot
match it. But a pure static-graph C API (ggml's) is hostile to the PyTorch-like ergonomics we
want. tinygrad demonstrates the synthesis: build lazily, realize at observation points, and
let an explicit `vkml.compile()` capture the training step.

Trade-off accepted: a bug in a kernel surfaces at the realize point, not at the op. **Mitigation
is a hard requirement, not a nice-to-have:** `VKML_EAGER=1` / `vkml.set_eager(True)` forces a
realize after every op. Same numerics, ~10× slower, and it makes op-level PyTorch comparison
straightforward. The validation suite in §7 runs in eager mode by default.

### Fork 2: autograd

| | Kernel-based (PyTorch) | Graph-based (ggml, tinygrad) |
|---|---|---|
| New kernels for backward | ~1 per op | ~4 total (see below) |
| Higher-order derivatives | needs double-backward kernels | free — differentiate the backward graph |
| Gradient checkpointing | custom autograd Function | re-emit the subgraph |
| Peak perf | slightly higher (fused backward kernels) | slightly lower |
| Total code | large | small |

**Recommendation: graph-based, unconditionally.** This is the highest-leverage decision in the
document.

Concretely, for the full long-term feature list I count **~64 forward ops**. With graph-based
autograd, exactly **four** ops need a genuinely new kernel for their backward:

1. `scatter_add` — embedding backward, `index_add`
2. `col2im` — conv2d input/weight gradient
3. `max_pool2d_backward` — scatter through argmax indices
4. `where_backward` for the mask-select case (arguably just `mul` by a mask — may be zero)

Everything else — every elementwise op, every reduction, matmul, softmax, layernorm,
attention — is expressible as a composition of forward ops. `d(matmul(A,B))/dA = grad @ Bᵀ`
is a `matmul` of a transposed view: no new kernel, no copy.

Trade-off accepted: a fused backward kernel would beat the composed version by maybe 10–20 %
on some ops. That is the right thing to give up for a ~60-kernel reduction in scope, and
individual ops can be fused later as a pure optimisation without any API change.

### Fork 3: the backend interface

ggml's four-level split (`reg` → `device` → `backend` → `buffer_type`/`buffer`) exists to
support 20 backends, dynamic `.so` loading, multi-GPU, and pipeline parallelism.

**Recommendation: keep three of the four levels, drop `reg`.**

```
Device   — enumeration, properties, supports_op(), creates Buffers and Streams
Buffer   — device memory + upload/download/copy
Stream   — records and submits work; owns the command buffer
```

Reasoning: `buffer_type` vs `backend` separation earns its keep even at two backends (host
staging buffers are host-allocated but device-visible). `reg` earns nothing until we ship
loadable backend plugins, which is not a goal. Keep `supports_op(node) -> bool` from day one
— it is what makes CPU fallback for unimplemented ops work, and retrofitting it is painful.

Trade-off accepted: adding CUDA/Metal later means adding a `Device` implementation, which is
the intended extension point. Adding *dynamically loaded* backends later would require
introducing `reg` — a contained, one-file change.

### Fork 4: tensor dimensionality

ggml fixes `MAX_DIMS = 4`. PyTorch is unbounded (heap `SmallVector`).

**Recommendation: fixed `MAX_DIMS = 4`.** Justification is concrete: the stated feature list —
CNNs `[N,C,H,W]`, transformers `[B,H,S,D]`, RNNs `[T,B,F]`, chess nets — is entirely 4D or
less. And the push-constant budget makes it a real constraint, not a preference:

| MAX_DIMS | bytes for 3 tensors' `ne`+`nb` (u32) | remaining of 256 B |
|---|---|---|
| 4 | 96 | 160 |
| 6 | 144 | 112 |
| 8 | 192 | 64 |

At 4 dims we have comfortable room for offsets and op params in push constants. At 8 we would
have to move shape metadata into a uniform buffer, adding an indirection to every kernel.

Trade-off accepted: `Conv3d` and 5D video tensors are excluded. Neither is on the feature list.
The escape hatch — moving the descriptor block to a UBO — is a localised change if it is ever
needed, and I'd rather pay it then than tax every kernel now.

### Fork 5: gradient accumulation without float atomics

Forced by `shaderBufferFloat32AtomicAdd = false`.

| Approach | Deterministic | Works here | Speed |
|---|---|---|---|
| Global `atomicAdd(float)` | no | **no** | fastest where available |
| `atomicCompSwap` loop on `uint` | no | yes | slow under contention |
| LDS accumulate + one global write per workgroup | within workgroup | yes | fast |
| Two-pass: partials buffer → reduction kernel | **yes** | yes | fast, costs one extra pass |

**Recommendation: LDS accumulation inside a workgroup, deterministic two-pass across
workgroups. No global float atomics anywhere in the codebase.**

This is a case where the hardware limitation pushes us toward the design we should want anyway.
Determinism means: same inputs → bit-identical outputs, every run. That makes the entire
PyTorch-parity suite reproducible rather than intermittently flaky, and it makes "did my change
alter numerics?" answerable by a hash comparison.

There is a second, quantitative reason. The tolerance target is `atol=rtol=1e-5` for fp32.
For a length-K dot product:

- sequential summation: relative error grows as `K · ε` → at K=4096, `4096 × 1.19e-7 ≈ 4.9e-4`
  — **fails the tolerance by 50×**
- pairwise/tree summation: grows as `log₂(K) · ε` → at K=4096, `12 × 1.19e-7 ≈ 1.4e-6`
  — **passes with 7× margin**

So the 1e-5 target is achievable, but only with tree reductions and fp32 accumulators. This is
a design constraint derived from the test plan, which is the right direction for it to flow.

---

## 4. Recommended architecture

### 4.1 Layering

```
python/vkml/                 Python package — nn.Module, optim, DataLoader, dtypes
   │                         Pure Python where it isn't hot. Mirrors torch's API surface.
   ▼  nanobind
src/api/                     C++ public headers — Tensor, Device, dtype. Stable ABI surface.
   ▼
src/autograd/                Backward rule table: op → λ(node, grad) building graph nodes
   ▼
src/graph/                   Node, Graph, topological build, structural hashing, lowering
   ▼
src/core/                    Tensor handle, Storage, shape/stride algebra, broadcasting
   ▼
src/plan/                    Memory planner (ggml gallocr port) + execution plan cache
   ▼
src/dispatch/                op × device → impl table; CPU-fallback graph splitting
   ▼
src/backend/api/             Device / Buffer / Stream interfaces (abstract)
   ├─ src/backend/cpu/       Reference implementation — the correctness oracle
   └─ src/backend/vulkan/    Instance, device, suballocator, pipeline cache, recorder
        ▼
shaders/                     GLSL compute → SPIR-V at build time, embedded as uint32_t[]
```

Dependency rule, enforced by directory-level include checks in CI: **each layer may include
only from layers strictly below it.** `backend/vulkan` must not know what an `nn.Linear` is;
`autograd` must not know Vulkan exists.

### 4.2 Core data structures

```cpp
// src/core/tensor.h  — a handle, not the data. Cheap to copy.
class Tensor {
    std::shared_ptr<Node>    node_;      // producing graph node (null for leaves)
    std::shared_ptr<Storage> storage_;   // null until realized
    Shape                    shape_;     // ne[4] + nb[4], strides in bytes (ggml convention)
    DType                    dtype_;     // F32 | F16 | I32 | I64 | BOOL
    Device                   device_;
    size_t                   offset_ = 0;
    bool                     requires_grad_ = false;
};
```

`Shape` carries `ne[4]` and `nb[4]` with `nb` in **bytes**, following ggml. Bytes rather than
elements because it makes mixed-dtype views and sub-byte padding expressible without special
cases, and because it is what the shaders need anyway.

```cpp
// src/graph/node.h  — POD-ish, arena-allocated
struct Node {
    OpKind        op;
    Shape         shape;
    DType         dtype;
    Device        device;
    std::array<Node*, 4> src;      // 4 is enough for every op in scope (matmul+bias+mask)
    OpParams      params;          // 64-byte inline union, no heap
    Node*         view_src;        // lifetime anchor for the planner — ggml's idea
    size_t        view_offs;
    uint32_t      flags;           // PARAM | OUTPUT | LOSS | NO_GRAD
    Storage*      storage;         // assigned by the planner
    size_t        storage_offset;
};
```

`src[4]` rather than ggml's `src[10]`: the widest op in scope is fused attention
(q, k, v, mask). Sizing to actual need keeps the node small and cache-friendly.

### 4.3 The training step, end to end

This is the flow that everything else serves:

```python
model = vkml.nn.Sequential(vkml.nn.Linear(128,256), vkml.nn.ReLU(), vkml.nn.Linear(256,10))
opt   = vkml.optim.Adam(model.parameters(), lr=1e-3)

@vkml.compile                          # optional; without it, same result, more CPU overhead
def step(x, y):
    loss = model(x).cross_entropy(y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss
```

What happens on the **first** call:

1. Python builds a lazy graph: forward nodes, then `backward()` appends backward nodes via the
   rule table, then `opt.step()` appends `adam_step` nodes. One graph, ~3× the forward size.
2. `plan/` runs the refcount simulation, computes offsets, and reports peak activation bytes.
3. `dispatch/` assigns each node a device and splits at boundaries (normally: no splits).
4. `backend/vulkan` records **one command buffer** for the entire step: dispatch, barrier,
   dispatch, barrier, …, and caches it keyed by graph structural hash.

On **subsequent** calls: re-submit the cached command buffer with new input data. Zero
allocation, zero descriptor updates, zero Python graph building, one queue submit, one
timeline-semaphore wait. The only host↔device traffic is the input batch in and the scalar
loss out.

That last property is why decision #3 matters so much on this machine: with only 256 MiB of
host-visible VRAM, every avoided synchronisation is real time saved.

---

## 5. Vulkan backend design

### 5.1 Memory: two levels, matching two lifetimes

```
Level 1 — DeviceAllocator   : suballocates VkDeviceMemory in 256 MB blocks, free-list,
                              separate pools per memory type. ~400 lines. (VMA's job,
                              done ourselves because we need only ~10 % of VMA.)
Level 2 — GraphPlanner      : assigns byte offsets inside one large activation buffer
                              for every intermediate tensor. Port of ggml_gallocr.
```

Lifetimes map cleanly onto the split:

| Class | Lifetime | Allocation |
|---|---|---|
| Parameters | whole training run | Level 1, persistent |
| Gradients | whole training run | Level 1, persistent |
| Optimizer state (`m`, `v`) | whole training run | Level 1, persistent |
| Activations / intermediates | one step | Level 2, planned offsets |
| Staging | transient | host-visible ring buffer |

**Worked budget for a model with P parameters, fp32, Adam:**

```
params 4P + grads 4P + Adam m,v 8P              = 16P bytes resident
activations                                      = planner output, batch-dependent
```

Against 5.75 GiB, reserving ~0.5 GiB for driver/fragmentation headroom:

| P | resident (16P) | left for activations |
|---|---|---|
| 10 M | 160 MB | ~5.1 GB |
| 50 M | 800 MB | ~4.4 GB |
| 100 M | 1.6 GB | ~3.6 GB |
| 250 M | 4.0 GB | ~1.2 GB |
| 350 M | 5.6 GB | **does not fit** |

So ~250 M parameters is the practical fp32+Adam ceiling, and ~150 M is comfortable. Storing
params and optimizer state in fp16 (with an fp32 master copy only where needed) moves the
ceiling, which is the real argument for mixed precision here — capacity, per §1.2.

**Staging**: because host-visible device-local memory is only 256 MiB, uploads go
host-coherent staging → `vkCmdCopyBuffer` → device-local. A persistent 64 MB ring buffer with
timeline-semaphore-tracked regions avoids per-upload allocation.

### 5.2 Sync

`timelineSemaphore` and `synchronization2` are both available, so: one timeline semaphore per
stream, monotonically incremented per submit. `wait(value)` replaces fence pools entirely.

Between dispatches, follow llama.cpp: **one global memory barrier**, not per-buffer barriers.
The refinement worth adding — which llama.cpp mostly doesn't do — is to *skip* the barrier when
consecutive nodes provably don't alias, which the planner already knows because it assigned the
offsets. Cheap to compute, and it lets independent elementwise chains overlap.

### 5.3 No descriptor sets

Because `bufferDeviceAddress` is available, kernels take raw 64-bit pointers:

```glsl
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

layout(buffer_reference, scalar) buffer F32Buf { float v[]; };

layout(push_constant, scalar) uniform PC {
    F32Buf a, b, d;          // 24 bytes  — device addresses, no descriptors
    uvec4  ne_a, nb_a;       // 32
    uvec4  ne_b, nb_b;       // 32
    uvec4  ne_d, nb_d;       // 32
    uint   n_elem;           //  4
    uint   op_params[8];     // 32
} p;                          // 156 of 256 bytes
```

What this deletes relative to ggml-vulkan: `VkDescriptorPool` management, pool growth,
`VkDescriptorSetLayout` per parameter count, `vkUpdateDescriptorSets` on every dispatch,
per-graph set-index reset, and the `MAX_PARAMETER_COUNT` ceiling. Roughly 200–500 lines of
the most bug-prone code in a Vulkan backend, plus measurable per-dispatch CPU cost.

`scalarBlockLayout` is what makes the struct above lay out identically in C++ and GLSL, so the
push-constant struct can be shared via a single header included by both.

Trade-off accepted: this makes the backend require Vulkan 1.2+ with `bufferDeviceAddress`.
That excludes some older/mobile GPUs. Given the project exists specifically to target this
machine, that is the right trade — and a descriptor-set path can be added later behind the same
internal interface if portability ever matters.

### 5.4 Kernels

**Build**: one `.comp` per op family. CMake invokes `glslc` at build time; SPIR-V is embedded
as `uint32_t[]` via generated headers, so there are no runtime file dependencies. A
`VkPipelineCache` is persisted to disk so the driver's SPIR-V→ISA compile is paid once.

**Variants**: prefer `layout(constant_id = N)` specialisation constants (tile sizes, workgroup
dims via `maintenance4`'s `local_size_id`, flags) over `#define` variants, because
specialisation constants mean **one SPIR-V blob, many pipelines** rather than N blobs. Use
`#define` only when the *type* must change (f32 vs f16 buffers). This is where I would
deliberately diverge from llama.cpp: their generator produces thousands of blobs largely for
quantisation types we don't have, and it is a significant build-time cost.

**Subgroup size**: pin per pipeline with
`VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`, seeded from llama.cpp's RDNA1 table
(§2.2): wave64 for reductions/softmax, wave32 default.

**GEMM plan** (M3). Target ≥3 TFLOPS fp32 at 1024³, stretch 4 TFLOPS, against the measured
4.29 TFLOPS reference:

```
workgroup 256 threads (8 waves of 32)
block tile   BM=64,  BN=64,  BK=16    LDS: (64+64)*16*4 B = 8 KB, double-buffered = 16 KB
thread tile  TM=4,   TN=4               → 16 accumulators in registers per thread
LDS padding  stride = BK + 1            → avoid 32-way bank conflicts
vectorized global loads via vec4
```

Three size classes (`s`/`m`/`l`) selected by shape, and an `aligned` variant with bounds checks
compiled out, exactly as llama.cpp does. Expect the naive version to hit ~5 % of peak and the
tiled version ~60–75 %; budget real time for this milestone.

**Reductions**: subgroup-shuffle within a wave → LDS across waves → deterministic two-pass
across workgroups. Never a global float atomic (§3, Fork 5).

**RNG**: counter-based Philox (as in sd.cpp's `rng_philox.hpp`), so `rand(seed, offset)` is a
pure function — reproducible, parallel, no state. Note explicitly: this will **not** match
PyTorch's RNG bit-for-bit, and it should not try to (§7.2).

---

## 6. Op inventory

~64 forward ops cover the entire long-term feature list.

| Group | Ops | Count |
|---|---|---|
| Creation | `zeros ones full arange randn rand` | 6 |
| View (zero-copy) | `reshape view permute transpose slice expand squeeze unsqueeze` | 8 |
| Movement (copy) | `contiguous cat stack pad index_select scatter_add` | 6 |
| Binary EW | `add sub mul div pow maximum minimum eq lt gt` | 10 |
| Unary EW | `neg exp log sqrt rsqrt recip sin cos tanh sigmoid relu gelu silu abs sign clamp` | 16 |
| Reductions | `sum mean max min prod argmax argmin` (arbitrary axes) | 7 |
| Matmul | `matmul bmm` | 2 |
| Composite | `softmax log_softmax layernorm rmsnorm batchnorm` | 5 |
| Conv/pool | `im2col col2im conv2d maxpool2d avgpool2d` | 5 |
| Loss | `mse_loss cross_entropy` | 2 |
| Optimizer (as nodes) | `sgd_step adam_step` | 2 |
| Misc | `where masked_fill dropout triu tril` | 5 |

RNN/LSTM/GRU and MultiHeadAttention are **compositions**, not ops — they add zero kernels.

Backward adds **4** new kernels total (§3, Fork 2). For comparison, a kernel-based autograd
design over the same op set would need ~60.

---

## 7. Validation architecture

Correctness over performance, PyTorch as ground truth. Building the harness is part of M0, not
an afterthought.

### 7.1 Three comparison axes

```
vkml-vulkan  ──compare──  vkml-cpu  ──compare──  PyTorch (CPU, torch 2.12.1)
      └──────────────── compare ────────────────────────┘
```

- **vkml-cpu vs PyTorch** catches algorithm/semantics errors (wrong formula, wrong axis,
  wrong broadcast rule) in an environment where debugging is easy.
- **vkml-vulkan vs vkml-cpu** catches kernel errors (indexing, races, tiling) with an oracle
  that shares our exact semantics, so a mismatch is unambiguously a kernel bug.
- **vkml-vulkan vs PyTorch** is the end-to-end gate.

This is ggml's `test-backend-ops` pattern (compare every backend against CPU) plus a
PyTorch tier on top.

### 7.2 On "the same random seed"

The spec says to generate identical random inputs using the same seed. Read literally that
would require bit-reproducing PyTorch's Mersenne Twister and Philox streams, which is a large
amount of work that validates nothing about our operators.

**Recommendation:** generate each test's inputs *once* (NumPy or torch), then feed the same
bytes to both frameworks. For training-parity tests, initialise in PyTorch and copy the weights
into vkml. RNG parity is then tested separately and only for distributional properties (mean,
variance, χ² uniformity, independence across offsets) rather than exact values.

This is a deliberate reinterpretation of the requirement rather than a relaxation of it — the
intent (identical inputs) is fully preserved; only the mechanism changes.

### 7.3 Tolerances, decided in advance

Per the rule that a mismatch must never be fixed by loosening tolerance, the expected error is
derived up front from the reduction length (see §3, Fork 5):

| Op class | atol | rtol | Justification |
|---|---|---|---|
| Elementwise fp32 | 1e-6 | 1e-6 | single op, ~1 ulp |
| Reductions K ≤ 1024 | 1e-5 | 1e-5 | tree sum, `log₂(K)·ε ≈ 1.2e-6` |
| Matmul K ≤ 4096 | 1e-5 | 1e-5 | tree sum, `12·ε ≈ 1.4e-6`, 7× margin |
| Matmul K > 4096 | 1e-4 | 1e-4 | derived from `log₂(K)·ε`; recompute per shape |
| fp16 storage, fp32 accum | 1e-3 | 1e-3 | fp16 ε ≈ 9.8e-4 |
| `exp`/`log`/`tanh`/`gelu` | 1e-5 | 1e-5 | transcendental ULP differences vs libm |

Any failure is investigated as a bug first. If a tolerance genuinely needs to change, the
change must come with the error analysis that justifies it, recorded in the test.

**NaN follows PyTorch, on both backends.** This was unwritten until issue #27,
which is why it had drifted: `relu(nan)` returned 0 while `maximum(x, 0)` and
`clamp_min(x, 0)` — the same function spelled differently — returned NaN, and the
Vulkan `amax`/`amin` reductions dropped NaN where the CPU propagated it. A tolerance
cannot express any of this: NaN is not *far from* a number, it is a different kind
of answer.

The rule is that torch is the reference, because a user should get the same answer
from vkml as from torch, and the same answer from either backend. Where torch and
numpy disagree — `sign(nan)` is +0.0 in torch and NaN in numpy — torch wins.

One mechanism explains every case, and is worth stating once: **every comparison
against NaN is false.** So `x > 0 ? x : 0` falls through to 0 and destroys a NaN,
while `x <= 0 ? 0 : x` falls through to `x` and keeps it. The two are identical on
numbers. Choosing between them is not a matter of style, and the same choice
appears in `relu`'s gradient, whose mask is `x <= 0` for exactly this reason.
Comparison alone cannot make a min/max reduction propagate NaN at all, which is
why both backends test for it explicitly.

Pinned by `tests/python/test_nan_semantics.py`, which checks every claim against
torch on both backends.

**f32 → f16 narrowing rounds to nearest, ties to even, on every backend and every driver.**
This was implicit and therefore untrue for a while. SPIR-V leaves `OpFConvert`'s rounding mode
implementation-defined, so `float16_t(x)` in a shader meant whatever the driver chose: RADV
rounds to nearest even, AMD's Windows compiler rounds toward zero, and the same program
produced different f16 values on the two. A tolerance cannot paper over that — the two answers
are both "correct" to 1 ulp, and the determinism guarantee in §7.4 tier 7 is exact-match. The
shaders therefore narrow in the integer domain (`f32_to_f16_bits` in `shaders/common.glsl`),
where nothing is implementation-defined, and `tests/python/test_f16.py` checks it on the tie
midpoints, which is where the rounding modes disagree and nowhere else.

### 7.4 Test tiers

1. **Op unit tests** — every op, every dtype, contiguous + strided + broadcast + edge shapes
   (empty, size-1, non-power-of-2). Both backends, both against torch.
2. **Autograd tests** — analytical gradient vs central finite differences
   (`(f(x+h)-f(x-h))/2h`, `h=1e-3` in fp64 on CPU) *and* vs `torch.autograd.grad`. Every input
   and every parameter checked.
3. **Layer tests** — each `nn` layer forward + backward + parameter grads vs the torch
   equivalent, with weights copied over.
4. **Optimizer tests** — 100 steps on a fixed problem; compare parameter trajectories, not
   just the endpoint. Catches drift that a single step hides.
5. **Model parity** — MNIST MLP → MNIST CNN → tiny GPT → chess eval net. Identical init,
   data order, hyperparameters. Compare the loss curve step by step.
6. **Property-based** (Hypothesis) — random shapes/strides/dtypes for shape algebra and
   broadcasting, where hand-written cases miss combinations.
7. **Regression** — golden hashes of outputs for a fixed seed. Because §3 Fork 5 makes
   everything deterministic, this is exact-match, not tolerance-based.
8. **Resource** — every allocation freed at exit (leak counter in `DeviceAllocator`), peak
   VRAM within 10 % of the planner's prediction, no `VK_ERROR_DEVICE_LOST` under a stress
   sweep of large tensors.
9. **Benchmarks** — GFLOPS/GB-s per op tracked over time against the §1.2 reference numbers,
   so a performance regression is visible.

Validation layers 1–4 run under `VKML_EAGER=1` so failures point at the offending op.

CI runs Khronos validation layers (`VK_LAYER_KHRONOS_validation`, present on this machine) on
every Vulkan test; any validation error fails the build.

---

## 8. Roadmap

Each milestone has an exit gate. Nothing proceeds until its gate is green. Effort estimates are
rough dev-days for one person and should be treated as relative sizing.

### M0 — Foundation and the oracle (no Vulkan) · ~10–15 d
Repo, CMake, `Tensor`/`Shape`/`Storage`/`Node`/`Graph`, broadcasting and shape algebra, CPU
backend for ~30 ops, graph-based autograd with the rule table, nanobind bindings, `nn.Module`
+ `Linear`/`ReLU`/`Sequential`, `SGD`, and the full pytest harness from §7.
> **Gate:** MNIST MLP trains on CPU; loss curve matches PyTorch within 1e-5 for 100 steps;
> finite-difference gradient check passes for all 30 ops.

*Why first:* it fixes the whole API surface and the oracle before any GPU debugging exists to
confuse matters. Every later milestone is validated against what M0 produces.

### M1 — Vulkan bring-up · ~8–12 d
Instance/device selection, `DeviceAllocator`, staging ring, timeline sync, SPIR-V build+embed,
pipeline cache, command recording, and exactly one op (`add`) end to end.
> **Gate:** `vkml.randn(1024,1024,device='vulkan')` round-trips; `add` matches CPU exactly;
> 10,000 alloc/free cycles leak nothing; validation layers clean.

### M2 — Elementwise and reductions · ~8–12 d
All 26 elementwise ops, all 7 reductions with arbitrary axes, strided/broadcast paths, tree
reduction with subgroup shuffles.
> **Gate:** tier-1 tests green for every op on Vulkan; reduction error inside the §7.3 bound
> for K up to 1e6.

### M3 — GEMM · ~10–15 d
Naive → tiled → tuned, three size classes, aligned variants, fp32 then fp16.
> **Gate:** correct for all shapes including non-multiples of tile size; **≥3 TFLOPS fp32 at
> 1024³** (reference: 4.29 TFLOPS). Below 2 TFLOPS, stop and profile rather than proceed.

### M4 — Training on GPU · ~6–10 d
Backward on Vulkan, `adam_step` as a graph node, `nn.Linear`, `cross_entropy`.
> **Gate:** MNIST MLP trains on Vulkan; loss curve matches the M0 CPU run and PyTorch within
> 1e-4 over 100 steps; parameter trajectories compared, not just final loss.

### M5 — Memory planner and step capture · ~8–12 d
Port `ggml_gallocr`, in-place reuse, `vkml.compile()` command-buffer capture, barrier elision
using planner alias information.
> **Gate:** zero allocations per step after warm-up; measured peak VRAM within 10 % of the
> planner's prediction; ≥2× step-time improvement on the M4 benchmark.

### M6 — CNNs · ~10–14 d
`im2col`/`col2im`, `conv2d` (im2col+GEMM first, direct later), pooling, `BatchNorm`.
> **Gate:** MNIST CNN parity with PyTorch; conv backward matches finite differences.

### M7 — Transformers · ~10–14 d
`softmax`/`log_softmax`, `LayerNorm`, `Embedding` (+ `scatter_add` backward),
`MultiHeadAttention` as a composition, causal masking.
> **Gate:** a small GPT trains; loss matches PyTorch within 1e-4 for 100 steps.

### M8 — Recurrent · ~6–8 d
`RNN`/`LSTM`/`GRU` as compositions; `tanh`/`sigmoid` gate fusion if profiling justifies it.
> **Gate:** parity against `torch.nn.LSTM` forward, backward, and parameter grads.

### M9 — Production concerns · ~10–15 d
fp16 mixed precision (motivated by capacity, §1.2), model serialization, `DataLoader`,
profiler with timestamp queries, benchmark suite, docs.
> **Gate:** the chess eval net trains end to end; a documented benchmark table exists.

**Critical path:** M0 → M1 → M2 → M3 → M4 is the spine; everything after M4 is largely
parallelisable. M3 is the highest-risk milestone and the one most likely to overrun.

---

## 9. Explicit non-goals

Recorded so they don't creep in:

- Quantisation (int8/int4/k-quants). Training doesn't need it; it is most of ggml's complexity.
- Distributed / multi-GPU training.
- bf16 — unsupported by this GPU.
- Dynamic shapes without recompilation. Changing batch size re-plans the graph; that is fine.
- A general graph compiler with algebraic rewrites. Fusion, if added, will be a small
  peephole pass over elementwise chains, not a compiler.
- PyTorch-bit-exact RNG (§7.2).
- Loadable backend plugins (no `reg` layer, §3 Fork 3).

---

## 10. Open questions for you

1. **Python bindings — `nanobind` or `pybind11`?** I lean nanobind: ~4× smaller binaries,
   faster compiles, better `dlpack`/ndarray support, and a cleaner story on Python 3.14. The
   cost is a smaller ecosystem and less Stack Overflow coverage. pybind11 3.0.4 is the
   conservative choice and also supports 3.14.

2. **First real model to target after MNIST?** The chess eval net is the most personally useful
   and is small enough to fit comfortably; a tiny GPT exercises more of the op surface. This
   changes whether M7 or M9 comes first.

3. **Does the naive-first GEMM path in M3 need to be timeboxed?** I'd suggest a hard cap —
   if the tiled kernel isn't at 2 TFLOPS within the estimate, fall back to calling ggml's
   Vulkan GEMM temporarily so M4+ isn't blocked, and return to it later. Happy to leave it
   uncapped if you'd rather do it properly first.

4. **Should `MAX_DIMS = 4` be revisited?** I've argued for 4 in §3 Fork 4 based on the stated
   feature list and the push-constant budget. If Conv3d or video models are plausibly in
   scope, say so now — it's much cheaper to decide before the kernels exist.

---

## Appendix A — Idea provenance

Per the coding rules, every borrowed idea with its source and justification.

| Idea | From | Why it exists there | Why it applies here |
|---|---|---|---|
| `ne[]`/`nb[]` byte strides | ggml | zero-copy views without a layout system | same; also what shaders need |
| `view_src` lifetime anchor | ggml | allocator must not free a viewed buffer | our planner has the identical problem |
| Backward builds graph nodes | ggml, tinygrad | avoids a second kernel set | cuts ~60 kernels from scope |
| Refcount-simulating offset planner | ggml `gallocr` | minimal peak memory for static graphs | 5.75 GB makes this decisive |
| In-place reuse for eligible ops | ggml `ggml_op_can_inplace` | halves buffers for elementwise chains | same |
| Optimizer step as a graph node | ggml `ggml-opt` | one submit per training step | 256 MiB BAR makes sync expensive |
| `buffer_type` ≠ `backend` | ggml | host-pinned memory usable by GPU | staging buffers need exactly this |
| `supports_op()` + graph splitting | ggml `sched` | transparent CPU fallback | we won't have every op on GPU for a long time |
| 3-level GEMM tiling, LDS pad | llama.cpp `mul_mm.comp` | ~74 % of peak without matrix cores | no cooperative matrix on RDNA1 |
| Specialisation constants for tiles | llama.cpp | one source, many pipelines | fewer SPIR-V blobs than `#define` variants |
| Per-pipeline subgroup size | llama.cpp `rdna1_pipelines` | RDNA1 wants wave64 for reductions | measured tuning data for our exact GPU |
| One global barrier between dispatches | llama.cpp | correct and cheap | don't over-engineer sync |
| Adaptive submission batching | llama.cpp | balances GPU idle vs submit overhead | same |
| Named children + params, dotted prefix | sd.cpp `GGMLBlock`, PyTorch | `state_dict` compatibility | users expect `nn.Module` |
| `params_ctx` / `compute_ctx` split | sd.cpp `GGMLRunner` | persistent vs per-graph lifetime | maps onto our two allocator levels |
| Counter-based Philox RNG | sd.cpp `rng_philox.hpp` | stateless, parallel, reproducible | GPU dropout/init need exactly this |
| Lazy build + realize points | tinygrad | fusion and planning need whole subgraphs | lets a graph engine feel eager |
| JIT capture of a repeated step | tinygrad `TinyJit` | removes Python overhead | becomes `vkml.compile()` |
| `.backward()` / `.grad` / `no_grad()` | PyTorch | the API users already know | UX parity is a stated goal |
| Backend-vs-CPU op test matrix | ggml `test-backend-ops` | unambiguous kernel-bug attribution | our §7.1 middle tier |
| Buffer device address, no descriptors | *none of them* | — | they support old GPUs; we don't have to |

## Appendix B — Reproducing the measurements

```bash
# device capabilities
vulkaninfo --json=0 -o dev0.json     # then read capabilities.device.{features,properties}

# GEMM benchmark (source in scratchpad; links against the local llama.cpp build)
g++ -O2 -std=c++17 gemm_bench.cpp -o gemm_bench \
    -I/home/jeel/Projects/llama.cpp/ggml/include \
    -L/home/jeel/Projects/llama.cpp/build/bin -lggml -lggml-base \
    -Wl,-rpath,/home/jeel/Projects/llama.cpp/build/bin
GGML_VK_VISIBLE_DEVICES=0 ./gemm_bench

# ggml's own op benchmarks, for comparison
/home/jeel/Projects/llama.cpp/build/bin/test-backend-ops perf -o MUL_MAT -b Vulkan0
```
