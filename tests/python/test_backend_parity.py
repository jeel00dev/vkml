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


def test_the_declined_set_is_known():
    """Pins exactly what Vulkan still refuses, so a change to it is deliberate.

    `prod` has no Vulkan kernel at all. `max_pool2d` requires a contiguous
    input, which its shader indexes planes directly to justify. Both are
    recorded in docs/VERIFICATION-AUDIT.md; implementing either makes this fail,
    which is the prompt to update the record.
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
        "f32-strided": ["max_pool2d", "prod"],
        "f16-contiguous": ["prod"],
        "f16-strided": ["max_pool2d", "prod"],
    }, f"the set of operators Vulkan declines changed: {declined}"
