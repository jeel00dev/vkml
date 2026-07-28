"""Operators under a non-default input layout, and at a size that spans workgroups.

Both are code paths rather than new operators, and both were measured to be
unexercised (`scripts/coverage_matrix.py`). They are grouped here because they
are one concern -- an operator can be correct on the inputs it has been given and
wrong on the inputs it has not.

NON-CONTIGUOUS INPUTS. A computed node's output is always contiguous, but its
sources need not be: a transpose, a permute or a strided slice produces a view
with rearranged strides, and every kernel indexes through `operand_offset` to
read it. An operator that has only ever been handed contiguous inputs has never
run that arithmetic, and the failure mode is silent -- it reads the right number
of elements from the wrong places.

SIZE. The default compute workgroup is 256 invocations (`vk_pipeline.h`). Below
that a kernel runs in one group, and every bug in its cross-group indexing is
invisible. The sizes here deliberately sit above it.

EVERY TEST RUNS ON BOTH BACKENDS. That is not symmetry for its own sake: the
first version of this file ran on the CPU only, and a mutation that broke
`operand_offset` in `shaders/common.glsl` survived it untouched. The CPU and the
GPU implement strided indexing separately, so covering one says nothing about
the other.

Every input is asserted non-contiguous, or large, before use. A test that meant
to exercise a path and quietly did not is worse than no test.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import vkml as V
from conftest import TOLERANCES, assert_close, make_input
from vkvalidate import gpu_device, vulkan_ready

REDUCE_TOL = TOLERANCES["reduction"]

DEVICES = [pytest.param("cpu", id="cpu")]
if vulkan_ready():
    DEVICES.append(pytest.param("gpu", id="vulkan"))

# prod has no Vulkan kernel. Excluded from the parametrised runs and pinned
# explicitly instead, so the omission is a stated contract rather than a silent
# hole -- see test_prod_has_no_vulkan_kernel.
CPU_ONLY_OPS = {"prod"}


def on(device):
    return gpu_device() if device == "gpu" else V.cpu


def transposed_pair(shape, seed, device, axes=(0, 1), shift=0.0):
    """A logically-`shape` tensor that is physically strided in both frameworks.

    Built by generating the axis-swapped array and transposing it back, so the
    two frameworks see identical values over identical non-contiguous layouts --
    a difference in either would test the harness rather than the kernel.

    `shift` moves the values away from zero, which `prod` needs to keep its
    precision and `log_softmax` needs to stay well conditioned.
    """
    i, j = axes
    base_shape = list(shape)
    base_shape[i], base_shape[j] = base_shape[j], base_shape[i]

    base = (make_input(tuple(base_shape), seed=seed) + shift).astype(np.float32)
    v = V.tensor(base, device=on(device)).transpose(i, j)
    t = torch.from_numpy(base.copy()).transpose(i, j)

    assert not v.is_contiguous, "input was meant to be strided; the test would be vacuous"
    assert not t.is_contiguous(), "torch input was meant to be strided"
    assert tuple(v.shape) == tuple(shape)
    return v, t, base


# ---------------------------------------------------------------------------
# Strided inputs
# ---------------------------------------------------------------------------

STRIDED_REDUCTIONS = [
    ("amax", lambda v, ax: V.amax(v, ax), lambda t, ax: torch.amax(t, dim=ax)),
    ("amin", lambda v, ax: V.amin(v, ax), lambda t, ax: torch.amin(t, dim=ax)),
    ("mean", lambda v, ax: V.mean(v, ax), lambda t, ax: torch.mean(t, dim=ax)),
    ("prod", lambda v, ax: V.prod(v, ax), lambda t, ax: torch.prod(t, dim=ax)),
    ("argmax", lambda v, ax: V.argmax(v, ax), lambda t, ax: torch.argmax(t, dim=ax)),
    ("argmin", lambda v, ax: V.argmin(v, ax), lambda t, ax: torch.argmin(t, dim=ax)),
    ("log_softmax", lambda v, ax: V.log_softmax(v, ax),
     lambda t, ax: torch.log_softmax(t, dim=ax)),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("name,vf,tf", STRIDED_REDUCTIONS, ids=[r[0] for r in STRIDED_REDUCTIONS])
def test_reduction_over_a_strided_input(name, vf, tf, axis, device):
    if name in CPU_ONLY_OPS and device == "gpu":
        pytest.skip(f"{name} has no Vulkan kernel; pinned by test_prod_has_no_vulkan_kernel")

    # Shifted away from zero: prod over a near-zero value loses all its
    # precision, and the shift keeps argmax/argmin away from ties, where the
    # comparison would be about convention rather than correctness.
    v, t, base = transposed_pair((5, 4), seed=2000 + axis, device=device, shift=1.5)

    assert_close(f"{name}(strided, axis={axis}, {device})", vf(v, axis), tf(t, axis),
                 REDUCE_TOL, inputs=[base])


@pytest.mark.parametrize("device", DEVICES)
def test_index_select_over_a_strided_input(device):
    v, t, base = transposed_pair((5, 4), seed=2010, device=device)
    i = np.array([0, 3, 3, 1], dtype=np.int64)

    assert_close("index_select(strided)", V.index_select(v, 0, V.tensor(i, device=on(device))),
                 t.index_select(0, torch.from_numpy(i.copy())), inputs=[base])


@pytest.mark.parametrize("device", DEVICES)
def test_scatter_add_over_a_strided_input(device):
    v, t, base = transposed_pair((5, 4), seed=2011, device=device)
    i = np.array([0, 2, 2, 1, 0], dtype=np.int64)

    want = torch.zeros(3, 4).index_add_(0, torch.from_numpy(i.copy()), t)
    assert_close("scatter_add(strided)",
                 V.scatter_add(v, 0, V.tensor(i, device=on(device)), 3),
                 want, REDUCE_TOL, inputs=[base])


@pytest.mark.parametrize("device", DEVICES)
def test_im2col_over_a_strided_input(device):
    """The image axes are the transposed ones, so the window walk reads through
    non-unit strides -- the case a contiguous test cannot reach."""
    v, t, base = transposed_pair((1, 2, 4, 6), seed=2020, device=device, axes=(2, 3))

    assert_close("im2col(strided)", V.im2col(v, (2, 2)),
                 torch.nn.functional.unfold(t, (2, 2)), inputs=[base])


@pytest.mark.parametrize("device", DEVICES)
def test_col2im_over_a_strided_input(device):
    v, t, base = transposed_pair((1, 9, 4), seed=2021, device=device, axes=(1, 2))

    assert_close("col2im(strided)", V.col2im(v, (4, 4), (3, 3)),
                 torch.nn.functional.fold(t, (4, 4), (3, 3)), REDUCE_TOL, inputs=[base])


def test_max_pool2d_over_a_strided_input():
    """CPU only. See test_strided_max_pool2d_is_refused_on_vulkan."""
    v, t, base = transposed_pair((1, 2, 4, 6), seed=2022, device="cpu", axes=(2, 3))

    assert_close("max_pool2d(strided)", V.max_pool2d(v, (2, 2)),
                 torch.nn.functional.max_pool2d(t, (2, 2)), inputs=[base])


@pytest.mark.skipif(not vulkan_ready(), reason="no Vulkan device available")
def test_strided_max_pool2d_is_refused_on_vulkan():
    """The backends disagree about what they accept, and the difference is
    visible to users.

    `VulkanBackend::supports` requires `src[0]->shape.is_contiguous()` for
    MaxPool2d and MaxPool2dBackward; the CPU kernel has no such requirement. An
    unsupported op raises rather than falling back (`executor.cpp`), so the same
    expression computes on the CPU and hard-fails on the GPU.

    Found by running this file on both backends after a coverage report showed
    the Vulkan strided path was untested. Pinned rather than worked around: it
    is a concrete instance of a question carried since the operator set was
    completed -- fall back by splitting the graph, or state that Vulkan is
    all-or-nothing -- and a silent divergence between backends is the worst of
    the available answers.
    """
    x = V.tensor(make_input((1, 2, 6, 4), seed=2025), device=gpu_device())

    V.max_pool2d(x, (2, 2)).numpy()  # contiguous: fine

    with pytest.raises(V.NotImplementedError_, match="cannot evaluate op 'max_pool2d'"):
        V.max_pool2d(x.transpose(2, 3), (2, 2)).numpy()


def test_max_pool2d_backward_through_a_strided_forward():
    """max_pool2d_backward routes gradient by argmax, and the argmax it recovers
    has to agree with the forward pass over the same strided layout.

    The KERNEL never sees a strided buffer here, and that is deliberate rather
    than a hole: the backward rule takes `grad.contiguous()` and
    `input.contiguous()` (autograd.cpp, OpKind::MaxPool2d), so the copy is made
    before the kernel runs. Measured, after a coverage report flagged this
    operator as never receiving a strided input -- it cannot. What this test
    pins is that the materialisation happens and the answer survives it; remove
    those `contiguous()` calls and the kernel reads the right count of elements
    from the wrong addresses, which its output-only assertion would not catch.
    """
    base = make_input((1, 2, 6, 4), seed=2023)

    v = V.tensor(base, requires_grad=True)
    t = torch.from_numpy(base.copy()).requires_grad_(True)

    V.sum(V.max_pool2d(v.transpose(2, 3), (2, 2))).backward()
    torch.nn.functional.max_pool2d(t.transpose(2, 3), (2, 2)).sum().backward()

    assert_close("max_pool2d_backward(strided)", v.grad, t.grad, REDUCE_TOL, inputs=[base])


@pytest.mark.parametrize("device", DEVICES)
def test_slice_backward_through_a_strided_forward(device):
    """slice_backward scatters a gradient back into a larger zeroed buffer, and
    given a transposed base the positions it must write are not sequential.

    As above, the kernel is handed a contiguous copy: the rule takes
    `grad.contiguous()` (autograd.cpp, OpKind::Slice). The CPU kernel asserts
    only that its OUTPUT is contiguous, so a strided input would be read wrongly
    rather than rejected -- which is what makes that copy load-bearing and worth
    a test rather than a comment.
    """
    base = make_input((6, 5), seed=2024)

    v = V.tensor(base, device=on(device), requires_grad=True)
    t = torch.from_numpy(base.copy()).requires_grad_(True)

    V.sum(v.transpose(0, 1)[1:4] * 2.0).backward()
    (t.transpose(0, 1)[1:4] * 2.0).sum().backward()

    assert_close("slice_backward(strided)", v.grad, t.grad, REDUCE_TOL, inputs=[base])


@pytest.mark.skipif(not vulkan_ready(), reason="no Vulkan device available")
def test_prod_has_no_vulkan_kernel():
    """Pins the one operator whose correctness chain is broken.

    `ARCHITECTURE.md` §7 verifies semantics on the CPU against PyTorch and then
    kernels on Vulkan against the CPU. `prod` only ever runs the first half, so
    nothing checks a GPU implementation -- because there is not one. Recorded as
    a test so that implementing it makes this fail, which is the prompt to add
    `prod` to the parametrised runs above.
    """
    x = V.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), device=gpu_device())
    with pytest.raises(V.NotImplementedError_, match="prod"):
        V.prod(x, 0).numpy()


# ---------------------------------------------------------------------------
# Sizes above one workgroup
# ---------------------------------------------------------------------------

# Prime dimensions, so the element count is not a multiple of the 256-invocation
# workgroup: an exact multiple hides a bad tail guard.
LARGE_ROWS, LARGE_COLS = 37, 41  # 1517 elements = six groups and a remainder

LARGE_REDUCTIONS = [
    ("amax", lambda v, ax: V.amax(v, ax), lambda t, ax: torch.amax(t, dim=ax)),
    ("amin", lambda v, ax: V.amin(v, ax), lambda t, ax: torch.amin(t, dim=ax)),
    ("mean", lambda v, ax: V.mean(v, ax), lambda t, ax: torch.mean(t, dim=ax)),
    ("argmax", lambda v, ax: V.argmax(v, ax), lambda t, ax: torch.argmax(t, dim=ax)),
    ("argmin", lambda v, ax: V.argmin(v, ax), lambda t, ax: torch.argmin(t, dim=ax)),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("name,vf,tf", LARGE_REDUCTIONS, ids=[r[0] for r in LARGE_REDUCTIONS])
def test_reduction_across_workgroups(name, vf, tf, axis, device):
    x = make_input((LARGE_ROWS, LARGE_COLS), seed=2100)
    # Distinct values, so argmax and argmin have a unique answer at a size where
    # a random tie becomes likely.
    x = x + np.arange(x.size, dtype=np.float32).reshape(x.shape) * 1e-3
    assert x.size > 256

    assert_close(f"{name}(large, axis={axis})", vf(V.tensor(x, device=on(device)), axis),
                 tf(torch.from_numpy(x.copy()), axis), REDUCE_TOL, inputs=[x])


def test_prod_across_workgroups():
    """CPU only, per test_prod_has_no_vulkan_kernel.

    The operands are drawn close to 1 because a product of 1517 arbitrary values
    overflows long before it says anything about indexing.
    """
    rng = np.random.default_rng(2101)
    x = rng.uniform(0.995, 1.005, size=(LARGE_ROWS, LARGE_COLS)).astype(np.float32)
    assert x.size > 256

    assert_close("prod(large)", V.prod(V.tensor(x), 1),
                 torch.prod(torch.from_numpy(x.copy()), dim=1), REDUCE_TOL, inputs=[x])


@pytest.mark.parametrize("device", DEVICES)
def test_cat_across_workgroups(device):
    a = make_input((200, 4), seed=2110)
    b = make_input((150, 4), seed=2111)
    assert (a.size + b.size) > 256

    assert_close("cat(large)",
                 V.cat([V.tensor(a, device=on(device)), V.tensor(b, device=on(device))], 0),
                 torch.cat([torch.from_numpy(a.copy()), torch.from_numpy(b.copy())], 0),
                 inputs=[a, b])


@pytest.mark.parametrize("device", DEVICES)
def test_index_select_across_workgroups(device):
    x = make_input((400, 3), seed=2120)
    i = np.random.default_rng(2121).integers(0, 400, size=300).astype(np.int64)
    assert (len(i) * 3) > 256

    assert_close("index_select(large)",
                 V.index_select(V.tensor(x, device=on(device)), 0, V.tensor(i, device=on(device))),
                 torch.from_numpy(x.copy()).index_select(0, torch.from_numpy(i.copy())),
                 inputs=[x])


@pytest.mark.parametrize("device", DEVICES)
def test_scatter_add_across_workgroups(device):
    """Also the size the O(n_out x index_len) scan is slowest at, so this is what
    would notice if that ever changed behaviour as well as cost."""
    x = make_input((300, 4), seed=2130)
    i = np.random.default_rng(2131).integers(0, 120, size=300).astype(np.int64)
    assert (120 * 4) > 256

    want = torch.zeros(120, 4).index_add_(0, torch.from_numpy(i.copy()),
                                          torch.from_numpy(x.copy()))
    assert_close("scatter_add(large)",
                 V.scatter_add(V.tensor(x, device=on(device)), 0,
                               V.tensor(i, device=on(device)), 120),
                 want, REDUCE_TOL, inputs=[x])
