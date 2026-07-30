# ADR 0009 — Operand metadata moves out of push constants

**Status:** accepted; **partially superseded** — the extents-once repack in §2 is
implemented for `where` and `softmax`, which now fit the guarantee. Only `cat`
still needs the device-buffer decision in §3, and it is deferred until its cost
is demonstrated.
**Date:** 2026-07-30
**Covers:** issue #2 (Windows: push constants exceed the 128-byte guaranteed
minimum), and the shape of the per-dispatch metadata path in general.

---

## 1. The defect

Vulkan guarantees `maxPushConstantsSize >= 128`. Nothing more. This project's
development machine reports 256, so the guaranteed minimum was never exercised,
and the budget was written down as the number that machine happened to report:

```cpp
static_assert(sizeof(WherePush) <= 256, "where push constants exceed the device budget");
```

`256` is not a budget. It is one device's limit, asserted as if it were the
contract. Three blocks are over the real minimum:

| Block | Size | Over by | Now |
|---|---|---|---|
| `WherePush` | 168 | 40 | **120 — repacked, fits** |
| `SoftmaxPush` | 152 | 24 | **120 — repacked, fits** |
| `CatPush` | 144 | 16 | still over; extents genuinely differ |

On a device that reports 128 — AMD's Windows driver on the reporting machine —
those pipelines cannot be created. 104 tests fail and MNIST cannot train.

The size is dominated by `Operand`, which is 32 bytes (`uvec4 ne` + `uvec4 nb`)
and appears once per tensor. `where` has four.

## 2. What was measured before choosing

Two cheap facts, both from live dispatches with the probe proven non-vacuous
(rule 10 — the first version of this probe printed nothing at all because
pytest captures stderr, and a `if (true)` control was what exposed it):

| Question | Result |
|---|---|
| Do `where`'s four operands carry the same extents? | **Yes, 101/101.** Inputs arrive already expanded to the output shape; broadcasting is carried entirely by zero strides |
| Do `cat`'s? | **No, 22/22 differ.** It concatenates along an axis, so the inputs genuinely have different extents there |

`softmax` was not measured at the time, and issue #24 was right that the
omission changes this section's conclusion. Measured now, the same way:

| Question | Result |
|---|---|
| Do `softmax`'s four operands share extents pairwise? | **Yes, 160/160.** `in_kept.ne == out_kept.ne` and `in_axis.ne == out_axis.ne` on every live dispatch across the whole Python suite |

That puts `SoftmaxPush` at **120 bytes** under the repack — inside the
guarantee:

```
today                                152 B      extents once per split   120 B
  2x uint64 src, dst          16              2x uint64 + 2x uint32     24
  2x uint32 n_out, n_axis      8              2x ne (kept, axis)        32
  4x GpuOperand              128              4x nb                     64
```

So the obvious repack — store the extents once per dispatch rather than once per
operand — fixes `where` (168 → 120) **and `softmax` (152 → 120)**, and does not
generalise only to `cat`. That asymmetry is
the argument against per-op packing: it works, but every future kernel has to
re-derive its own budget and discover for itself which trick applies.

**Both repacks are now implemented**, and the sharing they rely on is
STRUCTURAL rather than observed — which is what made them safe to do:

- `where`: `ops.cpp` broadcasts `cond`, `a` and `b` to the output shape at the
  single construction site, and the output is `Shape::contiguous(dims)`, so
  broadcasting is carried entirely by zero strides.
- `softmax`: the node is built with `Shape::contiguous(a.shape())`, so input and
  output have the same dims and `split_for_reduce` partitions them by the same
  axis. Only the strides can differ, which they do whenever the input is a
  transposed or broadcast view — the case the per-operand stride blocks exist
  for.

`strides_sharing_extents()` re-checks it per dispatch anyway, because a
guarantee nothing verifies decays: a later change to broadcasting would not
break a test, it would silently index with the wrong extents.

## 3. Decision

**Move the per-operand extent/stride array into a device buffer and pass its
address in push constants.**

```
WherePush today                     168 B      WherePush after         ~40 B
  4x uint64 addresses                32          4x uint64 addresses     32
  uint32 n                            4          uint32 n                 4
  4x Operand (uvec4 ne + uvec4 nb)  128          uint64 operands ->       8
```

Every op gets the same shape, permanently under the guaranteed minimum, with
room for rank > 4 later — which the current 4-dim limit forecloses today and
which would otherwise make this budget worse, not better.

**Benefit.** One mechanism for every kernel. The budget question stops recurring
per-op, and `VKML_MAX_DIMS` stops being load-bearing for portability.

**Cost.** An extra indirection per operand access, and the metadata is read from
memory rather than arriving in registers. The values are uniform across a
workgroup, so a compiler should keep them in scalar registers after one load,
but *should* is not a measurement.

**Worthwhile when** the indirection does not show on small kernels — the ones
whose runtime is dominated by launch and metadata rather than by arithmetic.

**Not worthwhile when** it does. If a small elementwise dispatch measurably
regresses, per-op repacking (§2) is the fallback for the three offenders, and
the general problem returns the next time an op needs five operands.

## 4. What implementation must handle

Recorded because none of it is visible from the decision, and two of the three
are easy to get silently wrong.

1. **The metadata is host-written and device-read, every dispatch.** That is the
   opposite of `splitk_workspace`, the existing workspace idiom, which is
   GPU-written and never touched by the host.

2. **The allocator has no memory kind for this.** `MemoryKind` offers
   `DeviceLocal` (not host-mappable on this card) and `HostStaging` (system
   memory, mappable). Neither is right: staging puts the read across PCIe for
   every workgroup, and device-local cannot be written without a staged copy and
   a barrier per dispatch, which would serialise exactly what ADR 0006's
   submission batching exists to keep parallel. Both devices on the development
   machine report a host-visible **and** device-local heap (256 MiB on NAVI10,
   2776 MiB on RENOIR) — that is the memory type this wants, and it needs a
   third `MemoryKind` with a documented fallback for devices that expose none.

3. **A ring buffer, not a single block.** Metadata for a dispatch must stay
   valid until that dispatch completes. Since submissions are batched, several
   are in flight at once, so the region must be sized to the in-flight window and
   reclaimed against the existing timeline semaphore. Overwriting in place would
   produce wrong results only under load, which is the worst way to find out.

## 4a. A second instance of the same root cause, found while gating this

The defect in §1 is not really "push constants are too big". It is **limits
asserted against the development machine instead of against the guarantee**, and
once the assertions were rewritten against the floor, a second instance turned
up immediately.

Every pipeline vkML creates requests a workgroup of **256 invocations**
(`KernelConfig::workgroup_size` defaults to it, and measurement across the whole
Python suite found no kernel using anything else). The Vulkan specification's
Required Limits table puts the minimum `maxComputeWorkGroupInvocations` at
**128** — cited from the spec, not measured here, and worth confirming before
acting on it.

If that floor is right, the blast radius is larger than §1: a conformant device
reporting the minimum cannot create **any** vkML pipeline, not merely the three
oversized ones. It has not been reported because it has not been met — both
development GPUs report 1024, and the Windows device in issue #2 got far enough
to fail on push constants, which means it too reports more than 128.

Measured shared memory is fine: the largest request is 8192 bytes against a
guaranteed 16384.

**Not fixed here, deliberately.** Halving the workgroup changes occupancy and
the GEMM tile geometry is tuned around the current width, so this is a
performance decision with a benchmark attached, not a portability patch. Recorded
rather than silently carried (P7).

## 5. Verification this must carry

- The static asserts change from `<= 256` to the guaranteed minimum, so the
  build fails on a machine that cannot see the problem. That is the regression
  gate: the original defect was undetectable locally, and after this it is a
  compile error.
- A test that a dispatch's metadata survives a full in-flight window, which is
  the one failure mode §4.3 introduces and the one no existing test would catch.
- The small-kernel benchmark that decides §3, run to the rules in
  docs/MEASUREMENT-AUDIT.md: minimum across process runs, warm pipelines,
  validation off, frozen baseline arm.
- MNIST and CIFAR-100 end to end, since every kernel's metadata path changes.
