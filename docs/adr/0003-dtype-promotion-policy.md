# ADR 0003 — dtype promotion policy

**Status:** accepted (strict now, documented target for later)
**Date:** 2026-07-26

---

## Context

vkml currently **refuses** binary operations between tensors of different
dtypes:

```python
a = vkml.tensor([1.0, 2.0])          # float32
b = a.to(vkml.int32)
a + b                                 # TypeError: vkml does not promote implicitly
```

PyTorch instead promotes, via a lattice that resolves `float32 + int64` to
`float32`, `float16 + float32` to `float32`, and so on, with separate rules for
Python scalars, 0-dim tensors and named "categories".

Implementing that lattice is not needed to train anything, so it is out of M0
scope. But the *decision* cannot be deferred: promotion changes the observable
result type of every binary op, so retrofitting it later is an API break unless
the eventual behaviour is fixed now and the current behaviour is a strict
subset of it.

## Decision

**Remain strict through M1. Adopt a reduced, explicitly-specified subset of
PyTorch's promotion rules afterwards — never a different one.**

The current strictness is deliberately chosen so that it is *forward
compatible*: every program vkml accepts today produces an identical result
under the future rules, because today it only accepts operands that already
share a dtype. Adding promotion can therefore only turn errors into successes,
never change an existing answer. That property is what makes deferral safe, and
it is the reason strictness was chosen over "promote to the wider operand"
guesswork.

### Target behaviour (to implement, not yet implemented)

1. **Tensor–tensor promotion follows a total order.**

   ```
   bool  <  int32  <  int64  <  float16  <  float32
   ```

   The result takes the greater of the two operand dtypes. This is PyTorch's
   behaviour restricted to the five dtypes vkml has, and it agrees with PyTorch
   on every pair in that set.

   Note `int64 < float16` is deliberate and matches PyTorch: mixing an integer
   with any floating type yields the floating type, even though float16 cannot
   represent every int64. Diverging here to be "safer" would silently disagree
   with the reference implementation, which is worse.

2. **Python scalars do not promote the tensor.** `int_tensor * 2.0` yields the
   *tensor's* dtype, not float32 — PyTorch's "wrapped number" rule. vkml already
   behaves this way (scalars adopt the peer's dtype in `scalar_like`), so this
   part needs no change.

3. **Comparison operators always yield `bool`.** Already true.

4. **Reductions preserve the input dtype**, except `mean`, which requires a
   floating input. PyTorch upcasts integer `mean` to float32; vkml will raise
   instead, because an implicit result-type change on a reduction is exactly the
   kind of surprise this project is trying to avoid, and `x.to(float32).mean()`
   is one call.

5. **In-place operations never promote.** `assign_` already requires an exact
   dtype match and will keep doing so; a promoting in-place op would silently
   reinterpret the destination buffer.

6. **`where(cond, a, b)`** promotes `a` and `b` against each other; `cond` must
   be `bool`.

### Not adopted from PyTorch

- **Category-based promotion** and the 0-dim-tensor special case. PyTorch treats
  a 0-dim tensor differently from a 1-element 1-dim tensor during promotion.
  This is a well-known source of confusion, and vkml will treat rank-0 tensors
  as ordinary tensors.
- **Complex and bfloat16 dtypes.** Neither exists here; bf16 is unsupported by
  the target GPU (measured, ARCHITECTURE.md §1.1).
- **`torch.set_default_dtype`.** Global mutable state affecting numerics.
  Creation functions take an explicit dtype with a float32 default.

## Consequences

- Until this is implemented, mixed-dtype code must insert `.to()` explicitly.
  The error message already says so.
- When it is implemented, the change is purely additive; no existing test should
  need to change. `test_dtype_promotion_is_strict` will be replaced by a table
  test comparing every dtype pair against PyTorch.
- The promotion table must be validated against PyTorch pair-by-pair, in the
  same harness as everything else, before it is enabled.

## Where the strictness lives

`check_same_dtype` in `src/api/ops.cpp` — a single function, which is where the
promotion lattice will be inserted.
