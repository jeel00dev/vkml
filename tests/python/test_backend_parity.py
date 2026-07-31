"""The correctness chain's precondition: whatever Vulkan computes, the CPU can too.

`ARCHITECTURE.md` §7 makes correctness a chain — CPU against PyTorch for
semantics, then Vulkan against the CPU for kernel bugs, "an oracle that shares
our exact semantics, so a mismatch is unambiguously a kernel bug". That second
link only exists if the CPU backend supports at least everything the Vulkan one
claims. **CPU support must be a superset of Vulkan support.**

Nothing enforced that, and it broke: widening the Vulkan `supports()` gates for
f16 across the movement and indexing family made the GPU accept `triu`, `tril`,
`scatter_add`, `im2col`, `col2im` and `max_pool2d` in f16 while the CPU kernels
still raised `DTypeError`. Every one of those had a GPU result no oracle could
check, and the whole suite stayed green, because every existing test drives the
two backends *separately* and none compares what they will accept.

So this file sweeps the surface and asserts the invariant directly. It is
deliberately about CAPABILITY rather than values -- what each backend agrees to
run -- which is the thing the per-operator tests cannot see.
"""

from __future__ import annotations

import numpy as np
import pytest

import vkml as V
from vkvalidate import gpu_device, vulkan_ready

pytestmark = pytest.mark.skipif(not vulkan_ready(), reason="no Vulkan device available")

# Ops whose GPU and CPU results differ by rounding rather than by construction:
# transcendentals, and anything that sums. They get the tolerance the suite
# already derives for them rather than exact equality.
INEXACT = {"sum", "mean", "exp", "gelu", "softmax", "log_softmax", "pow", "matmul",
           "col2im", "scatter_add"}


def _operands(dtype, device, strided, seed=1):
    rng = np.random.default_rng(seed)
    make = lambda shape, s: V.tensor(  # noqa: E731
        (rng.standard_normal(shape) * 0.5 + 1.5).astype(dtype), device=device)
    a, b, image = make((4, 4), 1), make((4, 4), 2), make((1, 1, 4, 4), 3)
    if strided:
        a, b, image = a.transpose(0, 1), b.transpose(0, 1), image.transpose(2, 3)
    index = V.tensor(np.array([0, 1, 1, 0], dtype=np.int64), device=device)
    return a, b, image, index


def build_cases(dtype, device, strided):
    """One entry per operator, over operands resident on `device`."""
    a, b, im, ix = _operands(dtype, device, strided)
    cases = [
        ("add", lambda: a + b), ("mul", lambda: a * b), ("relu", lambda: V.relu(a)),
        ("exp", lambda: V.exp(a)), ("gelu", lambda: V.gelu(a)),
        ("clamp", lambda: V.clamp(a, 0.0, 1.0)), ("greater", lambda: a > b),
        ("where", lambda: V.where(a > b, a, b)),
        ("sum", lambda: V.sum(a, 0)), ("mean", lambda: V.mean(a, 0)),
        ("amax", lambda: V.amax(a, 0)), ("prod", lambda: V.prod(a, 0)),
        ("argmax", lambda: V.argmax(a, 0)), ("softmax", lambda: V.softmax(a, 1)),
        ("log_softmax", lambda: V.log_softmax(a, 1)), ("matmul", lambda: V.matmul(a, b)),
        ("cat", lambda: V.cat([a, b], 0)),
        ("triu", lambda: V.triu(a)), ("tril", lambda: V.tril(a)),
        ("index_select", lambda: V.index_select(a, 0, ix)),
        ("scatter_add", lambda: V.scatter_add(a, 0, ix, 2)),
        ("im2col", lambda: V.im2col(im, (2, 2))),
        ("max_pool2d", lambda: V.max_pool2d(im, (2, 2))),
        ("contiguous", lambda: a.transpose(0, 1).contiguous()),
    ]
    return cases


def run(case):
    """(ok, result). `ok` is False when the backend declined the operator."""
    try:
        return True, case()
    except (V.NotImplementedError_, V.DTypeError):
        return False, None


CONDITIONS = [
    pytest.param(np.float32, False, id="f32-contiguous"),
    pytest.param(np.float32, True, id="f32-strided"),
    pytest.param(np.float16, False, id="f16-contiguous"),
    pytest.param(np.float16, True, id="f16-strided"),
]


@pytest.mark.parametrize("dtype,strided", CONDITIONS)
def test_cpu_supports_everything_vulkan_does(dtype, strided):
    """The invariant. A GPU capability with no CPU counterpart has no oracle."""
    gpu_cases = build_cases(dtype, gpu_device(), strided)
    cpu_cases = build_cases(dtype, V.cpu, strided)

    orphans = []
    for (name, on_gpu), (_, on_cpu) in zip(gpu_cases, cpu_cases):
        gpu_ok, _ = run(on_gpu)
        cpu_ok, _ = run(on_cpu)
        if gpu_ok and not cpu_ok:
            orphans.append(name)

    assert not orphans, (
        f"Vulkan accepts {orphans} but the CPU does not, so nothing can verify them. "
        "Either implement the CPU kernel or narrow VulkanBackend::supports()."
    )


@pytest.mark.parametrize("dtype,strided", CONDITIONS)
def test_vulkan_results_match_the_cpu_oracle(dtype, strided):
    """The chain's second link, swept rather than spot-checked."""
    tol = 1e-3 if dtype is np.float16 else 1e-5
    gpu_cases = build_cases(dtype, gpu_device(), strided)
    cpu_cases = build_cases(dtype, V.cpu, strided)

    for (name, on_gpu), (_, on_cpu) in zip(gpu_cases, cpu_cases):
        gpu_ok, gpu_out = run(on_gpu)
        if not gpu_ok:
            continue                       # declined; the test above owns that case
        cpu_ok, cpu_out = run(on_cpu)
        assert cpu_ok, f"{name}: covered by test_cpu_supports_everything_vulkan_does"

        got, want = gpu_out.numpy(), cpu_out.numpy()
        assert got.dtype == want.dtype, f"{name}: dtype differs, {got.dtype} vs {want.dtype}"
        if name in INEXACT:
            np.testing.assert_allclose(got.astype(np.float64), want.astype(np.float64),
                                       atol=tol, rtol=tol, err_msg=name)
        else:
            np.testing.assert_array_equal(got, want, err_msg=name)


def test_prod_folds_in_index_order():
    """Why prod has no Vulkan kernel, as an executable claim.

    A product's fold order is not a rounding detail: it decides when the fold
    overflows. Alternating 1e20 and 1e-20 cancels pair by pair in index order
    and reaches inf as soon as the large values are grouped -- which is exactly
    what reduce.comp's lane-striding would do, since lane 0 takes every WG-th
    element.

    So a GPU prod on that structure would return inf where this returns 1.0 --
    a different answer, not a tolerance miss. Pinned here so that anyone adding
    one has to confront the ordering first.
    """
    x = np.empty(512, dtype=np.float32)
    x[0::2], x[1::2] = 1e20, 1e-20

    assert V.prod(V.tensor(x), 0).numpy().item() == 1.0

    # The same values, grouped as a strided fold would group them. The overflow
    # is the point, so it is silenced rather than allowed to look like a fault.
    with np.errstate(over="ignore"):
        grouped = np.float32(1.0)
        for v in x[0::256]:
            grouped *= v
    assert np.isinf(grouped), "the demonstration no longer demonstrates anything"


def test_the_declined_set_is_known():
    """Pins exactly what Vulkan still refuses, so a change to it is deliberate.

    `prod` has no Vulkan kernel at all -- a product's fold order decides when it
    overflows, so a lane-strided GPU fold would give a different answer, not a
    rounding difference (test_prod_folds_in_index_order). That is a recorded,
    binding decision (VERIFICATION-AUDIT.md sec4), not a gap awaiting work.

    `max_pool2d` used to appear here for strided inputs too. Its shader now maps
    each image position through an Operand, so it accepts what the CPU accepts
    and has left the set -- which is exactly what this test is for: implementing
    something makes it fail, and widening the set is then a deliberate edit
    rather than a silent drift.
    """
    declined = {}
    for dtype, strided in ((np.float32, False), (np.float32, True),
                           (np.float16, False), (np.float16, True)):
        key = f"{'f16' if dtype is np.float16 else 'f32'}-{'strided' if strided else 'contiguous'}"
        declined[key] = sorted(
            name for name, case in build_cases(dtype, gpu_device(), strided)
            if not run(case)[0])

    assert declined == {
        "f32-contiguous": ["prod"],
        "f32-strided": ["prod"],
        "f16-contiguous": ["prod"],
        "f16-strided": ["prod"],
    }, f"the set of operators Vulkan declines changed: {declined}"


@pytest.mark.parametrize("dtype,name", [(np.float32, "f32"), (np.float16, "f16")])
def test_max_pool2d_addresses_planes_by_dtype_not_by_four_bytes(dtype, name):
    """More than one plane, in both dtypes. Single-plane images cannot catch this.

    `max_pool2d.comp` located each plane with `plane * H * W * 4`, a hardcoded
    four-byte element. For f16 that advances twice as far as it should, so plane
    0 was correct and every later plane read unwritten memory as zeros — silent
    wrong numbers on a configuration the declined-set test lists as SUPPORTED.

    Every existing max_pool2d test used a (1, 1, 4, 4) image. With `plane == 0`
    the stride is multiplied by zero, so the wrong constant could not show. The
    shapes here have several planes precisely so that it can, and both dtypes
    because f32 is right either way.

    The backward pass had the same expression for the original input it reloads
    to recover the argmax, so it is checked too.
    """
    rng = np.random.default_rng(11)
    image = rng.random((2, 3, 8, 8)).astype(dtype)

    cpu = V.max_pool2d(V.tensor(image, device=V.cpu), (2, 2)).numpy().astype(np.float32)
    gpu = V.max_pool2d(V.tensor(image, device=gpu_device()), (2, 2)).numpy().astype(np.float32)

    assert np.array_equal(gpu, cpu), (
        f"max_pool2d {name}: the backends disagree on a multi-plane image; "
        f"plane 0 max diff {np.abs(gpu[0] - cpu[0]).max()}, "
        f"plane 1 max diff {np.abs(gpu[1] - cpu[1]).max()}"
    )
    assert (gpu != 0).any(), "every output is zero, which is the shape of the plane-stride bug"


@pytest.mark.parametrize("dtype,name", [(np.float32, "f32"), (np.float16, "f16")])
def test_max_pool2d_backward_addresses_planes_by_dtype(dtype, name):
    """The adjoint reloads the original input to recover each window's argmax.

    It carried the same hardcoded element size, so a gradient could be routed
    from a plane read at the wrong address — which lands it on the wrong input
    element rather than merely producing a wrong value.
    """
    rng = np.random.default_rng(12)
    image = rng.random((2, 3, 8, 8)).astype(dtype)

    grads = []
    for device in (V.cpu, gpu_device()):
        t = V.tensor(image, device=device, requires_grad=True)
        V.sum(V.max_pool2d(t, (2, 2)), [0, 1, 2, 3]).backward()
        grads.append(t.grad.numpy().astype(np.float32))

    assert np.array_equal(grads[0], grads[1]), (
        f"max_pool2d backward {name}: the gradients differ across backends"
    )


@pytest.mark.parametrize("dtype,name", [(np.float32, "f32"), (np.float16, "f16")])
@pytest.mark.parametrize("axes,layout", [((2, 3), "HW-swapped"), ((0, 1), "NC-swapped")])
def test_max_pool2d_strided_and_multi_plane(dtype, name, axes, layout):
    """Strided AND several planes, which neither existing sweep covers together.

    `test_vulkan_results_match_the_cpu_oracle` sweeps strided layouts but its
    image is (1, 1, 4, 4) — a single plane, where the plane offset is zero and
    cannot be wrong. The multi-plane cases above are contiguous, where the
    strides are the packed ones and cannot be wrong either. A defect in the
    strided PLANE offset needs both at once, so it would sit in the gap between
    them.

    Swapping (0, 1) as well as (2, 3) matters: the first makes the plane axes
    themselves non-contiguous, so `plane * H * W` no longer walks planes in
    memory order, while the second leaves the plane stride packed and only
    disturbs the rows.
    """
    rng = np.random.default_rng(13)
    image = rng.random((2, 3, 8, 6)).astype(dtype)

    outs = {}
    for device, where in ((V.cpu, "cpu"), (gpu_device(), "gpu")):
        outs[where] = V.max_pool2d(
            V.tensor(image, device=device).transpose(*axes), (2, 2)
        ).numpy().astype(np.float32)

    # CPU-against-Vulkan only, which is this file's half of the chain. The
    # CPU-against-PyTorch half for a strided max_pool2d lives in
    # test_layout_and_scale.py::test_max_pool2d_over_a_strided_input, which now
    # runs on both devices.
    assert np.array_equal(outs["gpu"], outs["cpu"]), (
        f"max_pool2d {name} {layout}: the backends disagree on a strided multi-plane image"
    )
    assert (outs["gpu"] != 0).any(), "an all-zero result is the shape of a bad plane offset"
