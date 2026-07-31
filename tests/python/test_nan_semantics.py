"""NaN propagation, pinned against PyTorch on both backends.

WHY THIS FILE EXISTS
--------------------
Nothing in the suite fed a NaN to an operator, so no rule about what should
happen had ever been fixed (issue #27). Three behaviours had drifted apart, and
measuring found two more that the report had not:

| | was | torch | why it mattered |
|---|---|---|---|
| `relu(nan)` | 0 | nan | a NaN entering a ReLU network became 0 and training continued on numbers that looked healthy |
| `sign(nan)` | nan | +0.0 | matched numpy, not torch |
| `sign(-0.0)` | -0.0 | +0.0 | matched **neither** — numpy also returns +0.0 |
| Vulkan `amax`/`amin` | dropped it | nan | the two backends disagreed, which ARCHITECTURE.md §7 forbids |
| `d(relu)/dx` at nan | 0 | 1 | a forward that propagates and a backward that zeroes still hides the divergence |

**The policy is to match torch**, as the de facto reference, so a user gets the
same answer from vkml and from torch and the same answer from either backend.

The cause in every case is the same, and it is worth stating once: **every
comparison against NaN is false.** `x > 0 ? x : 0` therefore falls through to 0
and destroys the NaN, while `x <= 0 ? 0 : x` falls through to x and keeps it. The
two forms are identical on numbers. Choosing between them is not a style
question.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import vkml as V
from vkvalidate import gpu_device, vulkan_ready

NAN = float("nan")


def devices():
    """The CPU always; Vulkan when there is a device."""
    out = [(V.cpu, "cpu")]
    if vulkan_ready():
        out.append((gpu_device(), "vulkan"))
    return out


DEVICES = devices()
IDS = [name for _, name in DEVICES]


def _np(t):
    return t.numpy() if hasattr(t, "numpy") else np.asarray(t)


def assert_same(got, expected, what):
    """Equal, counting NaN as equal to NaN — which `==` does not."""
    g, e = np.asarray(got, dtype=np.float32), np.asarray(expected, dtype=np.float32)
    assert g.shape == e.shape, f"{what}: shape {g.shape} != {e.shape}"
    both_nan = np.isnan(g) & np.isnan(e)
    assert np.array_equal(g[~both_nan], e[~both_nan]) and np.array_equal(
        np.isnan(g), np.isnan(e)
    ), f"{what}: got {g.tolist()}, expected {e.tolist()}"


# ---------------------------------------------------------------------------
# The defect the issue was actually about
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device,name", DEVICES, ids=IDS)
def test_relu_and_its_equivalent_spellings_agree(device, name):
    """`relu(x)`, `maximum(x, 0)` and `clamp_min(x, 0)` are the same function.

    This is what made the original report a defect rather than a philosophical
    choice: vkml computed all three, and they disagreed with each other. Whichever
    answer is right, three spellings cannot be wrong while the fourth is right.
    """
    x = np.array([NAN, -1.0, 0.0, 2.0, -0.0], dtype=np.float32)
    t = V.tensor(x, device=device)
    zero = V.tensor(np.zeros_like(x), device=device)

    relu = _np(V.relu(t))
    maximum = _np(V.maximum(t, zero))
    clamp_min = _np(V.clamp_min(t, 0.0))

    assert_same(relu, maximum, f"relu vs maximum on {name}")
    assert_same(relu, clamp_min, f"relu vs clamp_min on {name}")


@pytest.mark.parametrize("device,name", DEVICES, ids=IDS)
def test_relu_propagates_nan_like_torch(device, name):
    x = np.array([NAN, -1.0, 0.0, 2.0], dtype=np.float32)

    got = _np(V.relu(V.tensor(x, device=device)))
    expected = torch.relu(torch.from_numpy(x.copy())).numpy()

    assert_same(got, expected, f"relu on {name}")
    assert np.isnan(got[0]), "a NaN entering relu must not become 0"


@pytest.mark.parametrize("device,name", DEVICES, ids=IDS)
def test_a_nan_survives_a_chain_of_operations(device, name):
    """The consequence, stated as the user would meet it.

    `relu(x) * 2 + 1` gave [1, 1, 5] where torch gives [nan, 1, 5]. A diverged
    model is supposed to announce itself; the old behaviour hid it.
    """
    x = np.array([NAN, -1.0, 2.0], dtype=np.float32)
    t = V.tensor(x, device=device)

    got = _np(V.add(V.mul(V.relu(t), 2.0), 1.0))
    xt = torch.from_numpy(x.copy())
    expected = (torch.relu(xt) * 2.0 + 1.0).numpy()

    assert_same(got, expected, f"relu chain on {name}")
    assert np.isnan(got[0]), "the NaN must still be visible at the end of the chain"


# ---------------------------------------------------------------------------
# sign -- where vkml matched neither reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device,name", DEVICES, ids=IDS)
def test_sign_matches_torch_including_the_zeros(device, name):
    """NaN and -0.0 both go to +0.0, and the sign BIT is checked, not just the value.

    `-0.0 == 0.0` is true, so a value comparison cannot tell the two apart. The
    old implementation returned -0.0 here, which matches neither torch nor numpy.
    """
    x = np.array([NAN, 0.0, -0.0, np.inf, -np.inf, 2.0, -2.0], dtype=np.float32)

    got = _np(V.sign(V.tensor(x, device=device)))
    expected = torch.sign(torch.from_numpy(x.copy())).numpy()

    assert_same(got, expected, f"sign on {name}")
    assert not np.signbit(got[0]), "sign(nan) must be +0.0, not -0.0"
    assert not np.signbit(got[2]), "sign(-0.0) must be +0.0, not -0.0"


# ---------------------------------------------------------------------------
# Reductions -- where the two backends disagreed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device,name", DEVICES, ids=IDS)
@pytest.mark.parametrize("position", [0, 1, 2], ids=["first", "middle", "last"])
@pytest.mark.parametrize("op", ["amax", "amin"])
def test_minmax_reductions_propagate_nan_wherever_it_sits(device, name, position, op):
    """Position matters, because the fold and the cross-lane tree are separate paths.

    The Vulkan reduction had TWO places that dropped NaN: the per-lane fold and
    the shared-memory tree. Fixing only the fold leaves a NaN in any lane but the
    first invisible, and with a 256-wide workgroup a three-element row exercises
    only one of them — hence the longer row below as well.
    """
    row = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    row[0, position] = NAN

    got = _np(getattr(V, op)(V.tensor(row, device=device), [1]))
    expected = getattr(torch, op)(torch.from_numpy(row.copy()), 1).numpy()

    assert_same(got, expected, f"{op} on {name} with NaN at {position}")
    assert np.isnan(got[0]), f"{op} dropped a NaN at position {position}"


@pytest.mark.parametrize("device,name", DEVICES, ids=IDS)
@pytest.mark.parametrize("op", ["amax", "amin"])
def test_minmax_propagates_nan_from_a_high_lane(device, name, op):
    """A row long enough that the NaN lands outside lane 0.

    This is the case the shared-memory tree owns: with a 256-wide workgroup, a
    NaN at index 700 is folded by a different lane and must survive the tree.
    """
    row = np.arange(1024, dtype=np.float32).reshape(1, 1024)
    row[0, 700] = NAN

    got = _np(getattr(V, op)(V.tensor(row, device=device), [1]))
    expected = getattr(torch, op)(torch.from_numpy(row.copy()), 1).numpy()

    assert_same(got, expected, f"{op} on {name}, NaN in a high lane")
    assert np.isnan(got[0]), f"{op} dropped a NaN folded by a non-zero lane"


@pytest.mark.parametrize("device,name", DEVICES, ids=IDS)
@pytest.mark.parametrize("op", ["sum", "mean"])
def test_summing_reductions_still_propagate(device, name, op):
    """Unchanged by this work, and asserted so a later fold rewrite cannot regress it."""
    row = np.array([[NAN, 1.0, -1.0, 2.0]], dtype=np.float32)

    got = _np(getattr(V, op)(V.tensor(row, device=device), [1]))
    expected = getattr(torch, op)(torch.from_numpy(row.copy()), 1).numpy()

    assert_same(got, expected, f"{op} on {name}")


@pytest.mark.parametrize("device,name", DEVICES, ids=IDS)
@pytest.mark.parametrize("op", ["amax", "amin"])
def test_minmax_without_nan_is_unchanged(device, name, op):
    """The NaN branch must not disturb the ordinary path, including infinities."""
    row = np.array([[3.0, -1.0, np.inf, 2.0, -np.inf, 0.0]], dtype=np.float32)

    got = _np(getattr(V, op)(V.tensor(row, device=device), [1]))
    expected = getattr(torch, op)(torch.from_numpy(row.copy()), 1).numpy()

    assert_same(got, expected, f"{op} on {name}, no NaN present")


# ---------------------------------------------------------------------------
# The backward pass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device,name", DEVICES, ids=IDS)
def test_relu_gradient_matches_torch_at_nan(device, name):
    """torch's relu backward masks on `x <= 0`, which NaN fails, so the grad flows.

    Asserted because a forward that propagates NaN and a backward that zeroes it
    would still hide a diverged model — one pass later, and harder to find.
    """
    x = np.array([NAN, -1.0, 0.0, 2.0], dtype=np.float32)

    t = V.tensor(x, device=device, requires_grad=True)
    V.sum(V.relu(t), [0]).backward()

    xt = torch.tensor(x.copy(), requires_grad=True)
    torch.relu(xt).sum().backward()

    assert_same(_np(t.grad), xt.grad.numpy(), f"d(relu)/dx on {name}")


# ---------------------------------------------------------------------------
# Cross-backend consistency, asserted directly
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not vulkan_ready(), reason="no Vulkan device")
@pytest.mark.parametrize(
    "op", ["relu", "sign", "abs", "neg", "sigmoid", "silu", "gelu", "exp", "tanh"]
)
def test_the_two_backends_agree_on_nan(op):
    """ARCHITECTURE.md §7 requires it, and `amax`/`amin` had broken it.

    Separate from the torch comparisons above: those could both drift together
    and still pass. This one only asks whether vkml agrees with itself.

    NaN-ness is compared EXACTLY; the finite values only within the 1e-5 gate
    from ARCHITECTURE.md §7.3. An earlier draft demanded exact values here and
    failed on `exp` and `tanh`, which differ between the backends by one ULP --
    the transcendental drift the gate exists to permit, and nothing to do with
    NaN. Asserting more than the question needs makes the test fail for reasons
    it is not asking about.
    """
    x = np.array([NAN, -1.0, 0.0, 2.0], dtype=np.float32)
    fn = getattr(V, op)

    gpu = _np(fn(V.tensor(x, device=gpu_device())))
    cpu = _np(fn(V.tensor(x, device=V.cpu)))

    assert np.array_equal(np.isnan(gpu), np.isnan(cpu)), (
        f"{op}: the backends disagree about WHICH elements are NaN -- "
        f"vulkan {gpu.tolist()}, cpu {cpu.tolist()}"
    )
    finite = ~np.isnan(gpu)
    np.testing.assert_allclose(gpu[finite], cpu[finite], rtol=1e-5, atol=1e-5)
