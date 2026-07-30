# ADR 0009 — Operand metadata moves out of push constants

**Status:** accepted; **not yet implemented**
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

| Block | Size | Over by |
|---|---|---|
| `WherePush` | 168 | 40 |
| `SoftmaxPush` | 152 | 24 |
| `CatPush` | 144 | 16 |

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

So the obvious repack — store the extents once per dispatch rather than once per
operand — fixes `where` (168 → 120) and does not generalise. That asymmetry is
the argument against per-op packing: it works, but every future kernel has to
re-derive its own budget and discover for itself which trick applies.

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
