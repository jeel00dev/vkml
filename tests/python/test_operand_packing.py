"""The shared-extent push-constant packing, and whether its operands stay distinct.

WHY THIS FILE EXISTS
--------------------
`where` and `softmax` store the operand EXTENTS once for several operands rather
than once each, because four 32-byte operand blocks put their push constants at
168 and 152 bytes -- over the 128 Vulkan guarantees, so those pipelines cannot be
created on a device reporting the minimum (issue #2, docs/adr/0009 §2).

That packing rests on the operands genuinely sharing extents, which they do by
construction: `ops.cpp` broadcasts every `where` input to the output shape, and
`softmax` is shape-preserving. The host re-checks it per dispatch anyway.

What the host CANNOT check is that the shader's push-constant block still
mirrors the C++ struct. Nothing declares that correspondence in one place -- it
is two declarations that must agree -- and getting it wrong reorders the stride
blocks, which silently indexes the wrong operand.

**Measured gap this file closes.** Swapping `a_nb` and `b_nb` in `where.comp`
without touching the host was caught by exactly ONE test in the whole suite,
`test_attention_agrees_across_backends`, and only because attention happens to
feed `where` operands with different strides. Every dedicated `where` test passed
the swap, because they use same-shaped contiguous inputs whose stride blocks are
identical -- so permuting them is invisible.

The tests below give every operand a DIFFERENT stride pattern, which is what
makes a permutation detectable at all.
"""

from __future__ import annotations

import numpy as np
import pytest

import vkml as V
from vkvalidate import gpu_device, vulkan_ready

pytestmark = pytest.mark.skipif(not vulkan_ready(), reason="no Vulkan device")

RTOL = 1e-5
ATOL = 1e-5


def _both(fn, *arrays):
    """Run on Vulkan and on the CPU oracle, returning both results."""
    gpu = fn(*[V.tensor(a, device=gpu_device()) for a in arrays]).numpy()
    cpu = fn(*[V.tensor(a, device=V.cpu) for a in arrays]).numpy()
    return gpu, cpu


def test_where_operands_are_not_interchangeable():
    """Each of cond, a and b gets a distinct stride pattern.

    This is the test that fails when the shader's stride blocks are permuted
    relative to the host struct. Same-shaped contiguous inputs cannot detect
    that, because their stride blocks are byte-identical -- so the shapes here
    are chosen to broadcast along DIFFERENT axes, giving each operand a stride
    layout no other operand shares.
    """
    rng = np.random.default_rng(4)
    cond = (rng.random((8, 16, 1)) > 0.5).astype(np.bool_)   # broadcasts on axis 2
    a = rng.standard_normal((8, 1, 4)).astype(np.float32)     # broadcasts on axis 1
    b = rng.standard_normal((1, 16, 4)).astype(np.float32)    # broadcasts on axis 0

    gpu, cpu = _both(lambda c, x, y: V.where(c, x, y), cond, a, b)

    assert gpu.shape == (8, 16, 4)
    np.testing.assert_allclose(gpu, cpu, rtol=RTOL, atol=ATOL)


def test_where_distinguishes_its_two_value_operands():
    """a and b specifically, which a symmetric test cannot separate.

    If `a_nb` and `b_nb` are swapped, `where` reads each value operand with the
    other's strides. Making the two disagree in BOTH shape and content is what
    turns that into a wrong answer rather than a coincidence.
    """
    rng = np.random.default_rng(5)
    cond = (rng.random((6, 10)) > 0.5).astype(np.bool_)
    a = np.full((6, 1), 1.0, dtype=np.float32)      # constant along axis 1
    b = np.arange(10, dtype=np.float32)[None, :]    # varies along axis 1

    gpu, cpu = _both(lambda c, x, y: V.where(c, x, y), cond, a, b)

    np.testing.assert_allclose(gpu, cpu, rtol=RTOL, atol=ATOL)
    # Independently of the oracle: every selected value must come from the
    # operand the condition names.
    expected = np.where(cond, a, b)
    np.testing.assert_allclose(gpu, expected, rtol=RTOL, atol=ATOL)


def test_where_still_correct_when_nothing_is_broadcast():
    """The contiguous fast path, where the shader skips index arithmetic entirely.

    Kept alongside the strided cases because the CONTIGUOUS specialisation
    constant selects a different code path, and a packing change touches the
    struct both paths read.
    """
    rng = np.random.default_rng(6)
    cond = (rng.random((32, 32)) > 0.5).astype(np.bool_)
    a = rng.standard_normal((32, 32)).astype(np.float32)
    b = rng.standard_normal((32, 32)).astype(np.float32)

    gpu, cpu = _both(lambda c, x, y: V.where(c, x, y), cond, a, b)
    np.testing.assert_allclose(gpu, cpu, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("op", ["softmax", "log_softmax"])
def test_softmax_input_and_output_strides_stay_separate(axis, op):
    """A transposed input is the case the two stride blocks per split exist for.

    softmax's output is freshly allocated and contiguous, while the input may be
    a view -- so `in_*_nb` and `out_*_nb` genuinely differ, and swapping them
    reads the input with the output's layout. A contiguous input cannot detect
    that, because then the two blocks are equal.
    """
    rng = np.random.default_rng(7)
    x = rng.standard_normal((24, 40)).astype(np.float32)

    fn = getattr(V, op)
    gpu, cpu = _both(lambda t: fn(t.T, axis), x)

    np.testing.assert_allclose(gpu, cpu, rtol=RTOL, atol=ATOL)


def test_softmax_rows_sum_to_one_on_a_strided_input():
    """An oracle-free check, so a shared misunderstanding cannot pass both sides.

    Every softmax row sums to 1 by definition, whatever the input's layout. If
    the packing indexed the wrong extents the rows would mix elements from
    neighbouring rows, and the sums would drift off 1 even though the GPU and
    CPU might still agree with each other.
    """
    rng = np.random.default_rng(8)
    x = rng.standard_normal((16, 24)).astype(np.float32)

    got = V.softmax(V.tensor(x, device=gpu_device()).T, 1).numpy()

    np.testing.assert_allclose(got.sum(axis=1), np.ones(24), rtol=1e-6, atol=1e-6)


def test_the_packed_extents_really_are_shared():
    """The host refuses a dispatch whose operands do not share extents.

    The packing is only sound while that holds. It holds by construction, so
    this cannot be provoked through the public API -- which is the point: the
    check exists for a FUTURE change to broadcasting or shape inference, and
    what is asserted here is that ordinary use does not trip it.
    """
    rng = np.random.default_rng(9)
    shapes = [((5,), (5,), (5,)), ((3, 7), (3, 1), (1, 7)), ((2, 3, 4), (1, 3, 1), (2, 1, 4))]

    for cshape, ashape, bshape in shapes:
        cond = (rng.random(cshape) > 0.5).astype(np.bool_)
        a = rng.standard_normal(ashape).astype(np.float32)
        b = rng.standard_normal(bshape).astype(np.float32)
        gpu, cpu = _both(lambda c, x, y: V.where(c, x, y), cond, a, b)
        np.testing.assert_allclose(gpu, cpu, rtol=RTOL, atol=ATOL)
