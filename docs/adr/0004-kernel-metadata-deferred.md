# ADR 0004 — Operator metadata: deferred

**Status:** deferred, with trigger conditions
**Date:** 2026-07-26

---

## Context

A table of per-operator properties was considered:

- `deterministic`
- `supports_views` / `supports_broadcasting`
- `supports_inplace`
- `associative_reduction`

## Analysis

Taking each in turn against what the codebase actually needs *today*:

**`deterministic`** — every operator in vkml is deterministic, by construction.
The CPU backend is single-threaded with fixed reduction order, and the Vulkan
backend is committed to deterministic tree reductions because the target GPU has
no global float atomics anyway (ARCHITECTURE.md §3, Fork 5). A flag whose value
is `true` for every entry carries no information. It becomes useful only if a
non-deterministic fast path is ever added, which is not planned.

**`supports_views` / `supports_broadcasting`** — these are not per-operator
properties in this design. Every kernel handles arbitrary strides through the
same `iterate.h` helpers, and broadcasting is resolved *before* kernels run, by
inserting stride-0 `Broadcast` view nodes during graph construction. There is no
kernel that could answer "no" to either question, so the fields would encode a
distinction the architecture does not make.

**`supports_inplace`** — this one is real but premature. ggml has exactly this
(`ggml_op_can_inplace`) and uses it in its allocator: if an op can write its
output over an input, and that input has one consumer and no live views, the
allocator reuses the buffer. vkml has no memory planner yet — the M0 executor
allocates every intermediate up front. The flag has no consumer until M5.

**`associative_reduction`** — would let a pass reorder or split reductions.
There is no such pass, and reordering a float reduction changes its numerics,
which conflicts with the determinism guarantee that the golden-hash regression
tests rely on. Any future use needs a numerics decision first, not a flag.

## Decision

**Do not add an operator metadata system now.** Add fields individually, when a
consumer exists, to a table introduced at that point.

The first one will almost certainly be `can_inplace`, arriving with the memory
planner at M5, ported from `ggml_op_can_inplace` — that is the exact use case
the idea exists to serve, and it is the one place the information pays for
itself.

## Why this is not just laziness

The cost of adding a metadata table later is genuinely low: it is a
`std::array<OpTraits, kNumOps>` beside the existing `kOpNames` array in
`src/graph/op.cpp`, with the same `static_assert` keeping it exhaustive. Nothing
about the current design obstructs it, and no code needs restructuring to
accommodate it.

The cost of adding it *now* is not zero: four fields × 76 operators is 304
assertions about behaviour that nothing verifies, which will drift out of date
precisely because nothing reads them. An unverified, unread table is worse than
no table.

## Trigger conditions

Introduce `OpTraits` when any of these becomes true:

1. The memory planner needs `can_inplace` (expected at M5).
2. A second backend disagrees with the CPU backend about which ops it supports
   in a way that `Backend::supports()` cannot express per-node.
3. A fusion pass needs to know which ops are elementwise.

Until then, `Backend::supports(const Node&)` answers the only question anything
actually asks, and answers it per-node — which is strictly more precise than a
static per-op table, since it can also consider dtype and rank.
