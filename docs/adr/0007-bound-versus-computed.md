# ADR 0007 — "Bound" and "computed" are different states

**Status:** accepted
**Date:** 2026-07-29
**Covers:** the first commit of stage B in
`docs/adr/0006-lazy-assign-and-submission-batching.md`, whose B0 experiment made
this necessary.
**Supersedes:** `Node::is_realized()`, which meant both at once.

---

## 1. Why this exists

`is_realized()` was defined as `storage != nullptr`. That was correct for every
node vkML had, because binding storage and computing into it always happened
together: the executor allocated a buffer and then immediately ran the kernel
that filled it.

Stage B introduces the first node for which they come apart. An Assign node
writes into its **destination's** buffer, so it is bound before it runs. B0
measured what that does today:

```
topological_order size, node unbound     3
topological_order size, node bound       0
```

The scheduler treats a node with storage as a leaf whose value already exists,
so an Assign node would be **silently skipped** -- never executed, no error.

One predicate answering two questions is the defect. This splits it.

---

## 2. The two states

### Bound -- `Node::is_bound()`

**Meaning:** this node has memory to write into. `storage != nullptr`.

Says nothing about the contents of that memory.

**Set by:**

| Where | When |
|---|---|
| `executor.cpp` `bind_storage()` | during `realize()`, before `compute()` |
| `tensor.cpp` `make_leaf()` | at construction, for a leaf that arrives with host data |
| `autograd.cpp` `detach()` | when sharing a buffer with the source |

**Never cleared.** A node keeps its storage for its lifetime.

### Computed -- `Node::is_computed()`

**Meaning:** the memory this node is bound to holds this node's value. Carried
as `kFlagComputed` in the existing `flags` word -- no new field, and no change
to `Node`'s size.

**Set by:**

| Where | When |
|---|---|
| `executor.cpp` `realize()` | **after** `backend.compute(order)` returns, for every node in the order |
| `tensor.cpp` `make_leaf()` | at construction -- an `Input` leaf has no rule to evaluate, so memory is all it needs |
| `autograd.cpp` `detach()` | after the shared value exists |

**Never cleared today.** Values are immutable once produced; nothing recomputes
a node in place. Stage B does not change that -- an Assign writes a *different*
node's buffer, it does not invalidate its own.

### The ordering that matters

```
   unbound, uncomputed          a freshly built graph node
        |  bind_storage()
   bound, uncomputed            <-- the state that did not exist before,
        |  compute()                and the one Assign lives in
   bound, computed              a value that can be read
```

The middle state is the whole point. Before this split it was unrepresentable.

---

## 3. Which decisions key on which

Every call site was classified rather than mechanically renamed, because the
two meanings genuinely divide:

| Site | Predicate | Why |
|---|---|---|
| `graph.cpp` `topological_order()` | **computed** | Stops the traversal. A node whose value exists needs no recomputation; a node merely holding a buffer does |
| `graph.cpp` `to_dot()` shading | **computed** | Shades what the scheduler treats as a leaf, so the picture and the plan agree |
| `executor.cpp` `bind_storage()` early return | **bound** | Its only job is allocating memory; it must not re-bind what already has some |
| `executor.cpp` view-base assertion | **bound** | A view needs its base's *address*, not its value |
| `cpu_backend.cpp` node assertion | **bound** | "Somewhere to write" is what a kernel needs before it runs |
| `cpu_backend.cpp` source assertion | **bound** | See below -- this one is subtle |
| `vulkan_backend.cpp` node assertion | **bound** | Same as the CPU backend |
| `autograd.cpp` `detach()` | **computed** | Detaching shares a *value*; a buffer not yet written is not one |

**The source assertion is the subtle one, and it is deliberately `is_bound()`.**
A source's value is usually produced by an earlier node *in the same batch*, and
`kFlagComputed` is only set once the whole batch returns. Asserting
`is_computed()` there would fire on every ordinary chain. What is locally
checkable is that the memory exists; that it holds the right value is the
executor's ordering guarantee, not something a backend can assert mid-batch.

---

## 4. Why they must stay independent

The temptation, once stage B is done, will be to notice that the two flags move
together for almost every node and fold them back. Three reasons not to:

1. **Assign is not the last such node.** Any operation writing into memory it
   does not own has the same shape: in-place elementwise ops, a memory planner
   that reuses buffers, gradient accumulation into an existing `.grad`. Each
   needs "bound but not yet computed" to exist.
2. **The failure mode is silence.** A skipped node produces no error, no
   exception and no wrong-looking value -- just a stale buffer. B0 found this
   before implementation; the same bug found after would present as an optimiser
   that quietly does not update some parameters.
3. **A memory planner needs both, separately.** Reusing a dead node's buffer
   means binding a new node to old memory: bound early, computed later. That is
   the same state, and `docs/ARCHITECTURE.md` already anticipates the planner.

**Binding this decision:** re-merging the predicates requires revisiting this
ADR, not just a refactor that makes the tests pass.

---

## 5. Cost

| | |
|---|---|
| **Benefit** | Makes the state Assign needs representable, and turns a silent skip into an impossible one. No size cost -- `kFlagComputed` uses the existing `flags` word |
| **Cost** | Two predicates where there was one, so every future site has to choose. That choice is a real judgement, as the source assertion in 3 shows |
| **Worthwhile when** | Any node can hold memory it has not written -- which is now |
| **Not worthwhile when** | Binding and computing are genuinely simultaneous everywhere. That was true before stage B and is why one predicate was right until now |

## 6. Verification

Behaviour is unchanged: 101 C++ cases and 1225 Python tests pass, and no
production code path produces a bound-uncomputed node yet.

The property is pinned in two places, both of which fail if the predicates are
re-merged:

* `test_graph.cpp`, "computed nodes terminate the traversal" -- with the
  subcase "storage ALONE does not terminate it".
* `test_aliasing.cpp`, "binding storage does not make a node look computed",
  which is the B0 experiment inverted: it used to assert the schedule went
  empty, and now asserts it does not.
