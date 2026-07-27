# ADR 0001 — Graph node ownership, and when an explicit IR arrives

**Status:** accepted
**Date:** 2026-07-26
**Supersedes:** the `std::array<Node*, 4> src` sketch in ARCHITECTURE.md §4.2
**Covers:** review questions 1 (ownership) and 3 (intermediate representation) — they
turn out to have a single answer, so they are decided together.

---

## Context

Graph nodes are created from Python, one op at a time, lazily. Nothing in the C++ core
has a natural scope that owns them. The current implementation uses
`std::shared_ptr<Node>` for `src`, `view_src` and `grad`.

Before ~5,000 lines of backend, autograd and binding code come to depend on it, the
question is whether that is the right model.

## Measurements

Everything below was measured on this machine (`docs/adr/bench/ownership_bench.cpp`,
GCC 14.2 `-O2`), not estimated. Option B here is a faithful arena implementation:
`std::vector<ArenaNode>` with `uint32` handles and the same traversal algorithm.

```
sizeof(vkml::Node)        = 320 bytes      sizeof(Shape)            = 80 bytes
sizeof(ArenaNode)         = 224 bytes      sizeof(OpParams)         = 64 bytes
sizeof(shared_ptr<Node>)  = 16 bytes

nodes         A build(ms)    B build(ms)     A trav(ms)     B trav(ms)
500                0.2651         0.0100         0.2226         0.0033
5000               2.0534         0.0997         2.2997         0.0361
50000             23.7838         1.6508        24.5079         0.9149

context: one relu over 64x128 f32 (8192 elements) = 0.00101 ms of actual compute
```

**The arena is 20x faster to build and up to 64x faster to traverse.** That is a much
larger gap than intuition suggested, and it deserves to be taken seriously rather than
explained away.

The number that decides the matter, though, is the last line. Per-node *compute* is
~1.0 µs for a modest elementwise op; per-node Option-A *overhead* is ~0.41 µs build +
~0.46 µs traverse. So the honest framing is:

| graph size | realistic model | Option A overhead | compute | overhead share |
|---|---|---|---|---|
| ~40 nodes | MNIST MLP step (M0–M4) | ~0.035 ms | ~5 ms (matmul-dominated) | **<1 %** |
| ~1,000 nodes | small transformer step (M7) | ~0.9 ms | ~50 ms | ~2 % |
| ~5,000 nodes | LSTM unrolled 100 steps (M8) | ~4.4 ms | ~20 ms | **~20 %** |

Option A is free at the scales M0 reaches, and becomes genuinely expensive only for deep
unrolled graphs — which is exactly where `vkml.compile()` (ARCHITECTURE.md §4.3) removes
per-step graph construction altogether.

## Options considered

### Option A — `std::shared_ptr<Node>` (current)

- **Construction:** one `make_shared` per node (object + control block in one allocation),
  plus an atomic increment per source edge. 410 ns/node measured.
- **Destruction:** O(n), needs the iterative worklist already implemented in `~Node`
  to avoid recursive teardown blowing the stack on deep chains.
- **Cache locality:** poor. 320-byte nodes scattered across the heap; a 5,000-node graph
  spans ~1.6 MB, so traversal misses L2 constantly. This is most of the 64x traversal gap.
- **Heap allocations:** one per node.
- **Fragmentation:** real over a long run — thousands of same-sized blocks allocated and
  freed per step. glibc's malloc handles this size class well (tcache/fastbins), so it is
  a minor concern rather than a serious one.
- **Thread safety:** refcounts are atomic, so tensors can be handed between threads safely.
  Node *contents* are unprotected, which is fine because they are immutable after
  construction (see "Guardrails" below).
- **Python interop:** the decisive advantage. Python holds Tensors for arbitrary,
  unpredictable lifetimes; `shared_ptr` makes that correct with zero bookkeeping.
- **Autograd / lazy execution:** both work directly. A backward node holding its forward
  activations alive is exactly the required semantics, and it happens automatically.
- **Optimization passes:** rewriting means building new nodes and dropping old ones —
  natural, and the old ones free themselves.
- **Vulkan / distributed:** neutral. Neither touches node ownership.

### Option B — arena owned by a `Graph`, `uint32` handles

- **Construction / destruction / locality:** far better, per the table above. Destruction
  is O(1): drop the whole vector.
- **Heap allocations:** amortised to near zero.
- **Fatal problem — ownership.** *Who owns the Graph, and when does it die?* In a Python
  session:
  ```python
  x = vkml.randn(3, 3)
  y = x + 1
  del x            # y still needs x's node
  ```
  A single process-wide arena grows without bound, which fails the "long-running training
  jobs" criterion outright. A per-step arena dangles the moment a user retains a tensor
  across steps — and retaining tensors across steps is normal (parameters, loss history,
  metrics).
- To fix that you must know which nodes are still reachable, which means refcounting or
  tracing GC. **At that point Option B has become Option A with an extra indirection and
  hand-rolled lifetime bookkeeping.**
- Evidence from mature projects points the same way: ggml uses an arena (`ggml_context`)
  and gets away with it because the *application* owns the loop — build graph, compute,
  reset context, no user-held tensors. tinygrad and PyTorch, both of which expose tensors
  to arbitrary Python code, both use refcounting. The dividing line is not performance,
  it is who controls object lifetime.

### Option C — intrusive refcounting (`intrusive_ptr<Node>`)

This is PyTorch's model (`c10::intrusive_ptr` over `TensorImpl`).

- Pointers shrink 16 → 8 bytes, taking `Node` from 320 to roughly 256 bytes: ~20 % fewer
  cache lines touched during traversal.
- The refcount could be made **non-atomic**, since graph construction happens under the
  Python GIL. That turns a ~20-cycle `lock xadd` into ~1 cycle.
- But the measured build cost is dominated by *allocation and zero-initialisation of 320
  bytes*, not by the atomics. Removing the atomics and shrinking the pointers might buy
  1.5–2x. It does not approach the arena's 20x.
- Cost: a hand-written refcounting base class is a well-known source of subtle bugs
  (self-assignment, exception safety during construction, aliasing). PyTorch's took years
  to settle.
- **Verdict: not worth it.** It pays a real maintenance cost for a fraction of the
  available win, and the available win does not matter at M0–M4 scale.

### Option D — hybrid: `shared_ptr` for the authored graph, arena for the lowered graph

Keep Option A as the *user-facing, lazily-built* graph, where lifetimes are unpredictable
and safety matters. Then have `vkml.compile()` **lower** it once into a flat, arena-backed,
index-based executable form that the memory planner, scheduler and executor consume:

```
Python  ──builds──▶  Node graph (shared_ptr, safe, arbitrary lifetimes)
                          │
                          │  lower()  — once per distinct graph structure
                          ▼
                     ExecGraph (std::vector<ExecNode>, uint32 indices, no refcounts)
                          │
                          ├──▶ memory planner (M5)
                          ├──▶ optimization passes (later)
                          └──▶ CPU / Vulkan executor
```

This captures the arena's advantages precisely where they are measurable — traversal,
planning and execution, which the benchmark shows is *half* the total cost — while leaving
lifetime management to `shared_ptr`, where it is correct for free.

It also answers review question 3: **the lowered form is the IR.** There is no need for a
separate IR layer; `ExecGraph` is it.

## Decision

**Keep Option A now. Commit to Option D — the lowered arena IR — at M5, alongside the
memory planner that first needs it.**

Rationale:

1. At M0–M4 graph sizes, Option A's overhead is under 1 % of step time. Optimising it now
   would be premature by any definition.
2. Every alternative that is faster either breaks Python lifetime semantics (B) or buys a
   fraction of the win for a large maintenance cost (C).
3. The lowering in D is **purely additive**. It requires no change to `Node`, no change to
   any layer above, and no rework of what is already built — it is a new consumer of the
   existing graph, not a replacement for it.
4. M5 is where the requirement actually appears: the memory planner needs a stable node
   ordering with side-tables for offsets, which is a flat array, not a pointer graph.

This is an explicit bet that big-graph workloads (M7 transformer, M8 RNN) arrive *after*
`compile()` exists. If that ordering slips, revisit — the measurements above are the data
needed to do so, and the benchmark is kept in `docs/adr/bench/` for exactly that purpose.

## Guardrails adopted now

Small, cheap commitments that keep the M5 lowering from becoming a refactor:

1. **`Node` must not appear in the public Tensor API.** `include/vkml/api/tensor.h` exposes
   no `Node`, `NodePtr`, or graph header. Swapping the internal representation then cannot
   break the ABI or the bindings.
2. **Nodes are immutable after construction**, except for the realisation fields
   (`storage`, `storage_offset`). Documented on the type. This is what makes it safe for
   two Tensors to share a node, for passes to assume a node never changes underneath them,
   and for the lowering to be a pure function of the graph.
3. **Pass results live in side-tables keyed by position in the topological order**, never
   as mutations of `Node`. `compute_use_counts` already follows this shape.

## Rejected micro-optimisations (recorded so they are not rediscovered)

- **Pooling `Node` allocations.** Would recover maybe 2–3x of build cost. No consumer needs
  it; revisit only if profiling at M7 says so.
- **Replacing `std::unordered_set` in `topological_order`.** ggml uses a custom
  open-addressing set with a bitset (`ggml_hash_set`) precisely because the standard
  container is slow here, and it is likely a large share of the measured 0.46 µs/node
  traversal cost. Deliberately not done: the lowering in D removes repeated traversal
  entirely, which is a better fix than making the traversal faster.
- **Shrinking `OpParams` from 64 to 32 bytes.** Every current param struct fits in 32, and
  `Conv2dParams` will need 28. Saves 10 % of `Node`. Not worth the churn while 64 matches
  ggml's precedent and leaves headroom.
