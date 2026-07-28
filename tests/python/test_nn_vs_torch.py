"""Layer, loss, optimizer and end-to-end training validation against PyTorch.

Weights are COPIED from the torch model into the vkml model rather than
re-derived from a shared seed. RNG parity is explicitly not a goal
(docs/ARCHITECTURE.md 7.2); what matters is that identical weights and
identical data produce identical trajectories.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import vkml as V
from conftest import TOLERANCES, Tol, assert_close, assert_shape, make_input

GRAD_TOL = TOLERANCES["reduction"]


def _copy_weights(vkml_model: V.nn.Module, torch_model: torch.nn.Module) -> None:
    state = {k: v.detach().numpy().copy() for k, v in torch_model.state_dict().items()}
    vkml_model.load_state_dict(state)


def test_linear_forward_and_backward():
    torch_lin = torch.nn.Linear(6, 4)
    vkml_lin = V.nn.Linear(6, 4)
    _copy_weights(vkml_lin, torch_lin)

    x = make_input((5, 6), seed=2000)
    vx = V.tensor(x, requires_grad=True)
    tx = torch.from_numpy(x.copy()).requires_grad_(True)

    vy = vkml_lin(vx)
    ty = torch_lin(tx)

    assert_shape("Linear forward", vy, ty)
    assert_close("Linear forward", vy, ty, TOLERANCES["matmul"], inputs=[x])

    V.sum(vy).backward()
    ty.sum().backward()

    assert_close("Linear grad wrt input", vx.grad, tx.grad, GRAD_TOL, inputs=[x])
    assert_close("Linear grad wrt weight", vkml_lin.weight.grad, torch_lin.weight.grad,
                 GRAD_TOL, inputs=[x])
    assert_close("Linear grad wrt bias", vkml_lin.bias.grad, torch_lin.bias.grad,
                 GRAD_TOL, inputs=[x])


def test_linear_without_bias():
    torch_lin = torch.nn.Linear(4, 3, bias=False)
    vkml_lin = V.nn.Linear(4, 3, bias=False)
    _copy_weights(vkml_lin, torch_lin)

    x = make_input((2, 4), seed=2010)
    assert_close("Linear(no bias)", vkml_lin(V.tensor(x)), torch_lin(torch.from_numpy(x.copy())),
                 TOLERANCES["matmul"], inputs=[x])


def test_sequential_matches_torch():
    torch_model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3)
    )
    vkml_model = V.nn.Sequential(V.nn.Linear(8, 16), V.nn.ReLU(), V.nn.Linear(16, 3))
    _copy_weights(vkml_model, torch_model)

    x = make_input((4, 8), seed=2100)
    assert_close("Sequential", vkml_model(V.tensor(x)), torch_model(torch.from_numpy(x.copy())),
                 TOLERANCES["matmul"], inputs=[x])


def test_state_dict_roundtrip():
    model = V.nn.Sequential(V.nn.Linear(3, 4), V.nn.ReLU(), V.nn.Linear(4, 2))
    names = sorted(dict(model.named_parameters()))
    assert names == ["0.bias", "0.weight", "2.bias", "2.weight"]

    saved = model.state_dict()
    model.load_state_dict(saved)
    for k, v in model.state_dict().items():
        np.testing.assert_array_equal(v, saved[k])


def test_state_dict_rejects_mismatch():
    model = V.nn.Linear(3, 4)
    with pytest.raises(KeyError):
        model.load_state_dict({"weight": np.zeros((4, 3), dtype=np.float32)})


def test_mse_loss():
    pred = make_input((5, 3), seed=2200)
    targ = make_input((5, 3), seed=2201)

    vp = V.tensor(pred, requires_grad=True)
    tp = torch.from_numpy(pred.copy()).requires_grad_(True)
    vt = V.tensor(targ)
    tt = torch.from_numpy(targ.copy())

    vl = V.nn.mse_loss(vp, vt)
    tl = torch.nn.functional.mse_loss(tp, tt)
    assert_close("mse_loss", vl, tl, TOLERANCES["reduction"], inputs=[pred, targ])

    vl.backward()
    tl.backward()
    assert_close("mse_loss grad", vp.grad, tp.grad, GRAD_TOL, inputs=[pred, targ])


def test_cross_entropy():
    logits = make_input((6, 5), seed=2300, low=-3.0, high=3.0)
    labels = np.array([0, 3, 1, 4, 2, 0], dtype=np.int64)

    vl_in = V.tensor(logits, requires_grad=True)
    tl_in = torch.from_numpy(logits.copy()).requires_grad_(True)

    vl = V.nn.cross_entropy(vl_in, V.tensor(labels))
    tl = torch.nn.functional.cross_entropy(tl_in, torch.from_numpy(labels))

    assert_close("cross_entropy", vl, tl, TOLERANCES["transcendental"], inputs=[logits])

    vl.backward()
    tl.backward()
    assert_close("cross_entropy grad", vl_in.grad, tl_in.grad, GRAD_TOL, inputs=[logits])


def test_cross_entropy_extreme_logits():
    """Where -log(softmax(x)) would be inf but log_softmax stays finite."""
    logits = np.array([[0.0, -300.0, -600.0]], dtype=np.float32)
    labels = np.array([2], dtype=np.int64)

    vl = V.nn.cross_entropy(V.tensor(logits), V.tensor(labels))
    tl = torch.nn.functional.cross_entropy(torch.from_numpy(logits.copy()),
                                           torch.from_numpy(labels))
    assert np.isfinite(vl.item())
    assert_close("cross_entropy(extreme)", vl, tl, TOLERANCES["transcendental"])


@pytest.mark.parametrize("momentum", [0.0, 0.9])
def test_sgd_parameter_trajectory(momentum):
    """Compare the whole trajectory, not just the endpoint.

    A wrong update rule can still reach a plausible final loss; comparing every
    step is what catches drift.
    """
    torch_model = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.ReLU(),
                                      torch.nn.Linear(8, 2))
    vkml_model = V.nn.Sequential(V.nn.Linear(6, 8), V.nn.ReLU(), V.nn.Linear(8, 2))
    _copy_weights(vkml_model, torch_model)

    x = make_input((10, 6), seed=2400)
    y = make_input((10, 2), seed=2401)

    v_opt = V.optim.SGD(vkml_model.parameters(), lr=0.05, momentum=momentum)
    t_opt = torch.optim.SGD(torch_model.parameters(), lr=0.05, momentum=momentum)

    for step in range(20):
        v_opt.zero_grad()
        v_loss = V.nn.mse_loss(vkml_model(V.tensor(x)), V.tensor(y))
        v_loss.backward()
        v_opt.step()

        t_opt.zero_grad()
        t_loss = torch.nn.functional.mse_loss(torch_model(torch.from_numpy(x.copy())),
                                              torch.from_numpy(y.copy()))
        t_loss.backward()
        t_opt.step()

        assert_close(f"SGD(momentum={momentum}) loss @ step {step}", v_loss, t_loss,
                     Tol(1e-5, 1e-5))

    for (vn, vp), (tn, tp) in zip(vkml_model.named_parameters(),
                                  torch_model.named_parameters()):
        assert vn == tn
        assert_close(f"SGD final param {vn}", vp, tp, Tol(1e-5, 1e-5))


def test_adam_parameter_trajectory():
    torch_model = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.Tanh(),
                                      torch.nn.Linear(7, 3))
    vkml_model = V.nn.Sequential(V.nn.Linear(5, 7), V.nn.Tanh(), V.nn.Linear(7, 3))
    _copy_weights(vkml_model, torch_model)

    x = make_input((8, 5), seed=2500)
    y = make_input((8, 3), seed=2501)

    v_opt = V.optim.Adam(vkml_model.parameters(), lr=1e-2)
    t_opt = torch.optim.Adam(torch_model.parameters(), lr=1e-2)

    for step in range(20):
        v_opt.zero_grad()
        v_loss = V.nn.mse_loss(vkml_model(V.tensor(x)), V.tensor(y))
        v_loss.backward()
        v_opt.step()

        t_opt.zero_grad()
        t_loss = torch.nn.functional.mse_loss(torch_model(torch.from_numpy(x.copy())),
                                              torch.from_numpy(y.copy()))
        t_loss.backward()
        t_opt.step()

        assert_close(f"Adam loss @ step {step}", v_loss, t_loss, Tol(1e-5, 1e-5))

    for (vn, vp), (_, tp) in zip(vkml_model.named_parameters(),
                                 torch_model.named_parameters()):
        assert_close(f"Adam final param {vn}", vp, tp, Tol(1e-4, 1e-4))


def test_mnist_style_mlp_training_parity():
    """The M0 exit gate: an MNIST-shaped MLP trained for 100 steps.

    Synthetic data of the right shape rather than the real dataset, so the test
    stays hermetic and fast. What is being validated is the training loop, not
    the data pipeline.
    """
    torch.manual_seed(0)
    torch_model = torch.nn.Sequential(
        torch.nn.Linear(784, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10)
    )
    vkml_model = V.nn.Sequential(
        V.nn.Linear(784, 128), V.nn.ReLU(), V.nn.Linear(128, 10)
    )
    _copy_weights(vkml_model, torch_model)

    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, size=(64, 784)).astype(np.float32)
    labels = rng.integers(0, 10, size=(64,))

    v_opt = V.optim.SGD(vkml_model.parameters(), lr=0.1)
    t_opt = torch.optim.SGD(torch_model.parameters(), lr=0.1)

    tx = torch.from_numpy(x.copy())
    tlabels = torch.from_numpy(labels.copy())
    vx = V.tensor(x)
    vlabels = V.tensor(labels)

    v_losses, t_losses = [], []

    for step in range(100):
        v_opt.zero_grad()
        v_loss = V.nn.cross_entropy(vkml_model(vx), vlabels)
        v_loss.backward()
        v_opt.step()
        v_losses.append(v_loss.item())

        t_opt.zero_grad()
        t_loss = torch.nn.functional.cross_entropy(torch_model(tx), tlabels)
        t_loss.backward()
        t_opt.step()
        t_losses.append(float(t_loss.item()))

        assert_close(f"MNIST-MLP loss @ step {step}", np.array(v_losses[-1]),
                     np.array(t_losses[-1]), Tol(1e-5, 1e-5))

    # The loop must actually be learning, not merely agreeing on a flat curve.
    assert v_losses[-1] < v_losses[0] * 0.5, (
        f"loss did not decrease: {v_losses[0]:.4f} -> {v_losses[-1]:.4f}"
    )

    for (name, vp), (_, tp) in zip(vkml_model.named_parameters(),
                                   torch_model.named_parameters()):
        assert_close(f"MNIST-MLP final {name}", vp, tp, Tol(1e-4, 1e-4))


def _trajectory(name, make_vkml_opt, make_torch_opt, steps=20, tol=Tol(1e-4, 1e-4)):
    """Run both frameworks in lockstep and compare every step.

    An optimiser with a wrong update rule can still reach a plausible final
    loss; comparing the whole trajectory is what catches drift, and state-
    carrying optimisers drift rather than jump.
    """
    torch_model = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh(),
                                      torch.nn.Linear(8, 3))
    vkml_model = V.nn.Sequential(V.nn.Linear(6, 8), V.nn.Tanh(), V.nn.Linear(8, 3))
    _copy_weights(vkml_model, torch_model)

    x = make_input((9, 6), seed=2600)
    y = make_input((9, 3), seed=2601)

    v_opt = make_vkml_opt(vkml_model.parameters())
    t_opt = make_torch_opt(torch_model.parameters())

    for step in range(steps):
        v_opt.zero_grad()
        v_loss = V.nn.mse_loss(vkml_model(V.tensor(x)), V.tensor(y))
        v_loss.backward()
        v_opt.step()

        t_opt.zero_grad()
        t_loss = torch.nn.functional.mse_loss(torch_model(torch.from_numpy(x.copy())),
                                              torch.from_numpy(y.copy()))
        t_loss.backward()
        t_opt.step()

        assert_close(f"{name} loss @ step {step}", v_loss, t_loss, tol)

    for (vn, vp), (tn, tp) in zip(vkml_model.named_parameters(),
                                  torch_model.named_parameters()):
        assert vn == tn
        assert_close(f"{name} final param {vn}", vp, tp, tol)


@pytest.mark.parametrize("weight_decay", [0.0, 1e-2, 0.1])
def test_adamw_parameter_trajectory(weight_decay):
    _trajectory(f"AdamW(wd={weight_decay})",
                lambda p: V.optim.AdamW(p, lr=1e-2, weight_decay=weight_decay),
                lambda p: torch.optim.AdamW(p, lr=1e-2, weight_decay=weight_decay))


@pytest.mark.parametrize("momentum,centered", [(0.0, False), (0.9, False), (0.0, True)])
def test_rmsprop_parameter_trajectory(momentum, centered):
    _trajectory(f"RMSProp(momentum={momentum},centered={centered})",
                lambda p: V.optim.RMSProp(p, lr=1e-3, momentum=momentum, centered=centered),
                lambda p: torch.optim.RMSprop(p, lr=1e-3, momentum=momentum,
                                              centered=centered))


def test_adamw_decouples_decay_from_adam():
    """AdamW is not Adam-with-weight_decay, and the difference is the point.

    Adam routes decay through the gradient, so it passes through the
    second-moment normalisation and is scaled by 1/sqrt(v). AdamW subtracts it
    from the parameter directly. With the same wd the two must diverge --
    otherwise one of them is implemented as the other.
    """
    x = make_input((6, 4), seed=2700)
    y = make_input((6, 2), seed=2701)

    finals = {}
    for name, ctor in (("adam", lambda p: V.optim.Adam(p, lr=1e-2, weight_decay=0.1)),
                       ("adamw", lambda p: V.optim.AdamW(p, lr=1e-2, weight_decay=0.1))):
        model = V.nn.Sequential(V.nn.Linear(4, 5), V.nn.Tanh(), V.nn.Linear(5, 2))
        opt = ctor(model.parameters())
        for _ in range(10):
            opt.zero_grad()
            V.nn.mse_loss(model(V.tensor(x)), V.tensor(y)).backward()
            opt.step()
        finals[name] = next(iter(model.named_parameters()))[1].numpy().copy()

    assert not np.allclose(finals["adam"], finals["adamw"], atol=1e-6), (
        "Adam and AdamW produced identical parameters; decay is not decoupled"
    )
