"""Layer, loss, optimizer and end-to-end training validation against PyTorch.

Weights are COPIED from the torch model into the vkml model rather than
re-derived from a shared seed. RNG parity is explicitly not a goal
(docs/ARCHITECTURE.md 7.2); what matters is that identical weights and
identical data produce identical trajectories.
"""

from __future__ import annotations

import math

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


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_binary_cross_entropy_with_logits(reduction):
    logits = make_input((6, 4), seed=2400, low=-4.0, high=4.0)
    # Soft targets, not just 0/1: torch allows them and the formula must not
    # quietly assume a Bernoulli label.
    target = make_input((6, 4), seed=2401, low=0.0, high=1.0)

    vx = V.tensor(logits, requires_grad=True)
    tx = torch.from_numpy(logits.copy()).requires_grad_(True)

    vl = V.nn.binary_cross_entropy_with_logits(vx, V.tensor(target), reduction=reduction)
    tl = torch.nn.functional.binary_cross_entropy_with_logits(
        tx, torch.from_numpy(target.copy()), reduction=reduction)

    assert_shape("bce_with_logits", vl, tl)
    assert_close("bce_with_logits", vl, tl, TOLERANCES["transcendental"],
                 inputs=[logits, target])

    vl.sum().backward()
    tl.sum().backward()
    assert_close("bce_with_logits grad", vx.grad, tx.grad, GRAD_TOL, inputs=[logits, target])


def test_bce_with_logits_survives_extreme_logits():
    """The whole reason this takes logits rather than probabilities.

    At |x| = 500 the naive -[y log s(x) + (1-y) log(1-s(x))] has already lost
    the losing term to underflow and returns inf. The rearranged form evaluates
    exp only at -|x|, so it stays finite and matches torch.
    """
    logits = np.array([[500.0, -500.0, 0.0]], dtype=np.float32)
    target = np.array([[1.0, 0.0, 1.0]], dtype=np.float32)

    vl = V.nn.binary_cross_entropy_with_logits(V.tensor(logits), V.tensor(target))
    tl = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.from_numpy(logits.copy()), torch.from_numpy(target.copy()))

    assert np.isfinite(vl.item()), "extreme logits produced a non-finite loss"
    assert_close("bce_with_logits(extreme)", vl, tl, TOLERANCES["transcendental"])


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("log_target", [False, True])
def test_kl_div(reduction, log_target):
    logits = make_input((5, 6), seed=2500, low=-2.0, high=2.0)
    other = make_input((5, 6), seed=2501, low=-2.0, high=2.0)

    vx = V.tensor(logits, requires_grad=True)
    tx = torch.from_numpy(logits.copy()).requires_grad_(True)

    # input is ALWAYS log-probabilities; target follows log_target.
    v_in = V.log_softmax(vx, -1)
    t_in = torch.log_softmax(tx, -1)
    t_tgt = torch.log_softmax(torch.from_numpy(other.copy()), -1)
    if not log_target:
        t_tgt = t_tgt.exp()
    target = t_tgt.numpy()

    vl = V.nn.kl_div(v_in, V.tensor(target), reduction=reduction, log_target=log_target)
    tl = torch.nn.functional.kl_div(t_in, t_tgt, reduction=reduction, log_target=log_target)

    assert_shape("kl_div", vl, tl)
    assert_close("kl_div", vl, tl, TOLERANCES["transcendental"], inputs=[logits, target])

    vl.sum().backward()
    tl.sum().backward()
    assert_close("kl_div grad", vx.grad, tx.grad, GRAD_TOL, inputs=[logits, target])


def test_kl_div_zero_target_is_zero_not_nan():
    """t log t is 0 at t = 0, but the arithmetic gets there via 0 * -inf.

    A target with exact zeros is ordinary -- any one-hot distribution has them --
    so this is the common case, not a corner.
    """
    log_input = np.log(np.array([[0.25, 0.25, 0.5]], dtype=np.float32))
    target = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

    vl = V.nn.kl_div(V.tensor(log_input), V.tensor(target), reduction="sum")
    tl = torch.nn.functional.kl_div(torch.from_numpy(log_input.copy()),
                                    torch.from_numpy(target.copy()), reduction="sum")

    assert np.isfinite(vl.item()), "a zero in the target produced NaN or inf"
    assert_close("kl_div(zero target)", vl, tl, TOLERANCES["transcendental"])


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("delta", [0.5, 1.0, 2.0])
def test_huber_loss(reduction, delta):
    pred = make_input((5, 4), seed=2600)
    targ = make_input((5, 4), seed=2601)

    vp = V.tensor(pred, requires_grad=True)
    tp = torch.from_numpy(pred.copy()).requires_grad_(True)

    vl = V.nn.huber_loss(vp, V.tensor(targ), reduction=reduction, delta=delta)
    tl = torch.nn.functional.huber_loss(tp, torch.from_numpy(targ.copy()),
                                        reduction=reduction, delta=delta)

    assert_shape("huber_loss", vl, tl)
    assert_close("huber_loss", vl, tl, TOLERANCES["reduction"], inputs=[pred, targ])

    vl.sum().backward()
    tl.sum().backward()
    assert_close("huber_loss grad", vp.grad, tp.grad, GRAD_TOL, inputs=[pred, targ])


def test_huber_loss_covers_both_branches():
    """Errors deliberately placed either side of delta, and exactly on it.

    A version that took only one branch would still pass a random-input test if
    the sample happened to land there, so the errors are chosen rather than
    drawn: 0.5 and -0.5 are quadratic, 5 and -5 linear, 1.0 is the join.
    """
    delta = 1.0
    pred = np.array([[0.5, -0.5, 5.0, -5.0, 1.0]], dtype=np.float32)
    targ = np.zeros_like(pred)

    vl = V.nn.huber_loss(V.tensor(pred), V.tensor(targ), reduction="none", delta=delta)
    tl = torch.nn.functional.huber_loss(torch.from_numpy(pred.copy()),
                                        torch.from_numpy(targ.copy()),
                                        reduction="none", delta=delta)
    assert_close("huber_loss(branches)", vl, tl, TOLERANCES["reduction"])

    # At the join the two pieces must agree, which is what makes the loss C1.
    at_join = vl.numpy()[0][4]
    assert abs(at_join - 0.5 * delta * delta) < 1e-6, f"discontinuous at delta: {at_join}"


def test_huber_loss_rejects_non_positive_delta():
    a = V.tensor(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="delta must be positive"):
        V.nn.huber_loss(a, a, delta=0.0)


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


# ---------------------------------------------------------------------------
# Convolution, pooling, normalisation and lookup layers
#
# Each is compared against its torch counterpart with weights copied across, so
# a mismatch is the layer rather than the initialisation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel,stride,padding,bias", [
    (3, 1, 1, True),
    (3, 1, 1, False),
    (2, 2, 0, True),
    ((3, 2), (2, 1), (1, 0), True),
])
def test_conv2d_layer(kernel, stride, padding, bias):
    torch_conv = torch.nn.Conv2d(3, 5, kernel, stride=stride, padding=padding, bias=bias)
    vkml_conv = V.nn.Conv2d(3, 5, kernel, stride=stride, padding=padding, bias=bias)
    _copy_weights(vkml_conv, torch_conv)

    x = make_input((2, 3, 8, 8), seed=3000)
    vx = V.tensor(x, requires_grad=True)
    tx = torch.from_numpy(x.copy()).requires_grad_(True)

    vy, ty = vkml_conv(vx), torch_conv(tx)
    assert_shape("Conv2d forward", vy, ty)
    assert_close("Conv2d forward", vy, ty, GRAD_TOL, inputs=[x])

    V.sum(vy).backward()
    ty.sum().backward()
    assert_close("Conv2d grad input", vx.grad, tx.grad, GRAD_TOL, inputs=[x])
    assert_close("Conv2d weight grad", vkml_conv.weight.grad, torch_conv.weight.grad, GRAD_TOL)


@pytest.mark.parametrize("layer,tlayer,shape", [
    (lambda: V.nn.MaxPool2d(2), lambda: torch.nn.MaxPool2d(2), (2, 3, 8, 8)),
    (lambda: V.nn.MaxPool2d(3, stride=1, padding=1),
     lambda: torch.nn.MaxPool2d(3, stride=1, padding=1), (1, 2, 6, 6)),
    (lambda: V.nn.AvgPool2d(2), lambda: torch.nn.AvgPool2d(2), (2, 3, 8, 8)),
    (lambda: V.nn.AvgPool2d(3, stride=1, padding=1),
     lambda: torch.nn.AvgPool2d(3, stride=1, padding=1), (1, 2, 6, 6)),
])
def test_pooling_layers(layer, tlayer, shape):
    x = make_input(shape, seed=3100)
    vx = V.tensor(x, requires_grad=True)
    tx = torch.from_numpy(x.copy()).requires_grad_(True)

    vy, ty = layer()(vx), tlayer()(tx)
    assert_shape("pool forward", vy, ty)
    assert_close("pool forward", vy, ty, GRAD_TOL, inputs=[x])

    V.sum(vy).backward()
    ty.sum().backward()
    assert_close("pool grad", vx.grad, tx.grad, GRAD_TOL, inputs=[x])


def test_flatten_layer():
    x = make_input((2, 3, 4, 5), seed=3200)
    got = V.nn.Flatten()(V.tensor(x))
    want = torch.nn.Flatten()(torch.from_numpy(x.copy()))
    assert_shape("Flatten", got, want)
    assert np.array_equal(got.numpy(), want.numpy())


def test_embedding_layer():
    torch_emb = torch.nn.Embedding(12, 5)
    vkml_emb = V.nn.Embedding(12, 5)
    _copy_weights(vkml_emb, torch_emb)

    tokens = np.array([[0, 3, 3], [7, 1, 0]], dtype=np.int64)
    vy = vkml_emb(V.tensor(tokens))
    ty = torch_emb(torch.from_numpy(tokens.copy()))

    assert_shape("Embedding forward", vy, ty)
    assert_close("Embedding forward", vy, ty)

    V.sum(V.mul(vy, vy)).backward()
    (ty * ty).sum().backward()
    assert_close("Embedding weight grad", vkml_emb.weight.grad, torch_emb.weight.grad, GRAD_TOL)


@pytest.mark.parametrize("affine", [True, False])
def test_layer_norm_layer(affine):
    torch_ln = torch.nn.LayerNorm(7, elementwise_affine=affine)
    vkml_ln = V.nn.LayerNorm(7, elementwise_affine=affine)
    if affine:
        _copy_weights(vkml_ln, torch_ln)

    x = make_input((3, 4, 7), seed=3300)
    vx = V.tensor(x, requires_grad=True)
    tx = torch.from_numpy(x.copy()).requires_grad_(True)

    vy, ty = vkml_ln(vx), torch_ln(tx)
    assert_close("LayerNorm forward", vy, ty, TOLERANCES["transcendental"], inputs=[x])

    V.sum(V.mul(vy, vy)).backward()
    (ty * ty).sum().backward()
    assert_close("LayerNorm grad", vx.grad, tx.grad, TOLERANCES["transcendental"], inputs=[x])


def test_batch_norm_training_forward_and_backward():
    torch_bn = torch.nn.BatchNorm2d(3)
    vkml_bn = V.nn.BatchNorm2d(3)
    _copy_weights(vkml_bn, torch_bn)
    torch_bn.train(); vkml_bn.train()

    x = make_input((4, 3, 5, 5), seed=3400)
    vx = V.tensor(x, requires_grad=True)
    tx = torch.from_numpy(x.copy()).requires_grad_(True)

    vy, ty = vkml_bn(vx), torch_bn(tx)
    assert_close("BatchNorm2d train forward", vy, ty, TOLERANCES["transcendental"], inputs=[x])

    V.sum(V.mul(vy, vy)).backward()
    (ty * ty).sum().backward()
    assert_close("BatchNorm2d grad", vx.grad, tx.grad, TOLERANCES["transcendental"], inputs=[x])


def test_batch_norm_running_statistics_track_torch_over_many_steps():
    """The biased/unbiased asymmetry, which one step cannot reveal.

    Normalisation uses the biased variance; the running estimate accumulates
    the unbiased one. Using either for both leaves the first step identical and
    diverges as the exponential average converges -- so this runs twenty
    batches and compares the buffers, not just the output.
    """
    torch_bn = torch.nn.BatchNorm2d(3)
    vkml_bn = V.nn.BatchNorm2d(3)
    _copy_weights(vkml_bn, torch_bn)
    torch_bn.train(); vkml_bn.train()

    for step in range(20):
        x = make_input((6, 3, 4, 4), seed=3500 + step)
        vkml_bn(V.tensor(x))
        torch_bn(torch.from_numpy(x.copy()))

    assert_close("running_mean", vkml_bn.running_mean, torch_bn.running_mean,
                 TOLERANCES["transcendental"])
    assert_close("running_var", vkml_bn.running_var, torch_bn.running_var,
                 TOLERANCES["transcendental"])


def test_batch_norm_eval_uses_running_statistics():
    """After training, eval must use the accumulated estimate rather than the
    batch's own -- so the same input gives a different answer in the two modes,
    and the eval answer matches torch's."""
    torch_bn = torch.nn.BatchNorm2d(2)
    vkml_bn = V.nn.BatchNorm2d(2)
    _copy_weights(vkml_bn, torch_bn)
    torch_bn.train(); vkml_bn.train()

    for step in range(10):
        x = make_input((5, 2, 3, 3), seed=3600 + step, low=1.0, high=4.0)
        vkml_bn(V.tensor(x))
        torch_bn(torch.from_numpy(x.copy()))

    # The probe is run through BOTH in training mode, because a training-mode
    # forward updates the running estimate -- feeding it to one model only
    # would leave the two buffers a batch apart and the eval comparison would
    # fail for that reason rather than for the one being tested.
    probe = make_input((5, 2, 3, 3), seed=3700, low=1.0, high=4.0)
    train_out = vkml_bn(V.tensor(probe)).numpy()
    torch_bn(torch.from_numpy(probe.copy()))

    vkml_bn.eval(); torch_bn.eval()
    v_eval = vkml_bn(V.tensor(probe))
    t_eval = torch_bn(torch.from_numpy(probe.copy()))

    assert_close("BatchNorm2d eval", v_eval, t_eval, TOLERANCES["transcendental"])
    assert not np.allclose(train_out, v_eval.numpy(), atol=1e-4), \
        "train and eval produced the same result; eval is not using running statistics"


def test_batch_norm_running_stats_are_not_parameters():
    """A running buffer carries no gradient, so an optimiser must not see it --
    otherwise it would be 'trained' and the estimate destroyed."""
    names = [n for n, _ in V.nn.BatchNorm2d(4).named_parameters()]
    assert names == ["weight", "bias"], names


def test_dropout_layer_advances_its_offset():
    """rand is a pure function, so a module reusing one offset would drop the
    SAME elements every step -- invisibly, while the loss curve still fell."""
    layer = V.nn.Dropout(0.5, seed=99)
    layer.train()
    x = V.full([4096], 1.0)

    first = layer(x).numpy()
    second = layer(x).numpy()
    assert not np.array_equal(first, second), "consecutive calls produced the same mask"

    # ...and the run is still reproducible from the seed.
    replay = V.nn.Dropout(0.5, seed=99)
    replay.train()
    assert np.array_equal(replay(x).numpy(), first)


def test_dropout_layer_is_identity_in_eval():
    layer = V.nn.Dropout(0.5, seed=1)
    layer.eval()
    x = make_input((10, 10), seed=3800)
    assert np.array_equal(layer(V.tensor(x)).numpy(), x)


def test_cnn_trains_and_matches_torch():
    """The whole stack together: conv, batch norm, pooling, flatten, linear.

    Compared step by step over a short run rather than at the endpoint, because
    a wrong layer can still reach a plausible final loss.
    """
    torch_model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 4, 3, padding=1), torch.nn.BatchNorm2d(4), torch.nn.ReLU(),
        torch.nn.MaxPool2d(2), torch.nn.Flatten(), torch.nn.Linear(4 * 4 * 4, 3))
    vkml_model = V.nn.Sequential(
        V.nn.Conv2d(1, 4, 3, padding=1), V.nn.BatchNorm2d(4), V.nn.ReLU(),
        V.nn.MaxPool2d(2), V.nn.Flatten(), V.nn.Linear(4 * 4 * 4, 3))
    _copy_weights(vkml_model, torch_model)

    x = make_input((8, 1, 8, 8), seed=3900)
    labels = np.random.default_rng(3901).integers(0, 3, size=8).astype(np.int64)

    v_opt = V.optim.SGD(vkml_model.parameters(), lr=0.05)
    t_opt = torch.optim.SGD(torch_model.parameters(), lr=0.05)

    for step in range(10):
        v_opt.zero_grad()
        v_loss = V.nn.cross_entropy(vkml_model(V.tensor(x)), V.tensor(labels))
        v_loss.backward()
        v_opt.step()

        t_opt.zero_grad()
        t_loss = torch.nn.functional.cross_entropy(
            torch_model(torch.from_numpy(x.copy())), torch.from_numpy(labels.copy()))
        t_loss.backward()
        t_opt.step()

        assert_close(f"CNN loss @ step {step}", v_loss, t_loss, Tol(1e-4, 1e-4))


# ---------------------------------------------------------------------------
# Attention
#
# Compared against torch's OWN MultiheadAttention with weights copied across,
# which the matching parameter layout makes possible. A reference written here
# would only prove the two agree with each other.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("embed_dim,heads,seq,batch", [(8, 2, 5, 2), (12, 3, 7, 1), (4, 4, 3, 3)])
def test_multihead_attention_forward(embed_dim, heads, seq, batch):
    torch_mha = torch.nn.MultiheadAttention(embed_dim, heads, batch_first=True)
    vkml_mha = V.nn.MultiheadAttention(embed_dim, heads)
    _copy_weights(vkml_mha, torch_mha)

    x = make_input((batch, seq, embed_dim), seed=4000)
    vx = V.tensor(x, requires_grad=True)
    tx = torch.from_numpy(x.copy()).requires_grad_(True)

    vy = vkml_mha(vx)
    ty, _ = torch_mha(tx, tx, tx, need_weights=False)

    assert_shape("MHA forward", vy, ty)
    assert_close("MHA forward", vy, ty, TOLERANCES["transcendental"], inputs=[x])

    V.sum(V.mul(vy, vy)).backward()
    (ty * ty).sum().backward()
    assert_close("MHA grad", vx.grad, tx.grad, TOLERANCES["transcendental"], inputs=[x])


def test_multihead_attention_causal():
    """Causal masking against torch's own is_causal path."""
    embed_dim, heads, seq = 8, 2, 6
    torch_mha = torch.nn.MultiheadAttention(embed_dim, heads, batch_first=True)
    vkml_mha = V.nn.MultiheadAttention(embed_dim, heads)
    _copy_weights(vkml_mha, torch_mha)

    x = make_input((2, seq, embed_dim), seed=4100)
    causal = torch.nn.Transformer.generate_square_subsequent_mask(seq)

    vy = vkml_mha(V.tensor(x), is_causal=True)
    ty, _ = torch_mha(*(torch.from_numpy(x.copy()),) * 3, attn_mask=causal, need_weights=False)

    assert_close("MHA causal", vy, ty, TOLERANCES["transcendental"], inputs=[x])


def test_causal_mask_actually_blocks_the_future():
    """Independent of torch, and the property causal attention exists for: an
    output position must not change when a LATER input position changes.

    A mask applied after the softmax, or one off by a diagonal, still produces
    plausible numbers and a falling loss -- this is what catches it.
    """
    seq, embed_dim = 6, 8
    mha = V.nn.MultiheadAttention(embed_dim, 2)

    x = make_input((1, seq, embed_dim), seed=4200)
    base = mha(V.tensor(x), is_causal=True).numpy()

    perturbed = x.copy()
    perturbed[0, -1, :] += 10.0            # change only the LAST position
    after = mha(V.tensor(perturbed), is_causal=True).numpy()

    # Every position but the last must be untouched.
    assert np.allclose(base[0, :-1], after[0, :-1], atol=1e-5), \
        "an earlier output changed when a later input did; attention is not causal"
    # ...and the last one must actually have moved, or the test proves nothing.
    assert not np.allclose(base[0, -1], after[0, -1], atol=1e-5)


def test_attention_scale_is_per_head():
    """1/sqrt(head_dim), not 1/sqrt(embed_dim). The wrong constant leaves the
    model trainable and merely worse, so only a direct comparison catches it --
    and the two differ only when head_dim != embed_dim, i.e. more than one head.
    """
    embed_dim, heads, seq = 16, 4, 5
    torch_mha = torch.nn.MultiheadAttention(embed_dim, heads, batch_first=True)
    vkml_mha = V.nn.MultiheadAttention(embed_dim, heads)
    _copy_weights(vkml_mha, torch_mha)

    x = make_input((1, seq, embed_dim), seed=4300, low=-4.0, high=4.0)
    vy = vkml_mha(V.tensor(x))
    ty, _ = torch_mha(*(torch.from_numpy(x.copy()),) * 3, need_weights=False)
    assert_close("MHA scale", vy, ty, TOLERANCES["transcendental"], inputs=[x])


def test_multihead_attention_cross_attention():
    """Query from one sequence, key and value from another -- the path where
    the three input projections are genuinely different tensors."""
    embed_dim, heads = 8, 2
    torch_mha = torch.nn.MultiheadAttention(embed_dim, heads, batch_first=True)
    vkml_mha = V.nn.MultiheadAttention(embed_dim, heads)
    _copy_weights(vkml_mha, torch_mha)

    q = make_input((2, 4, embed_dim), seed=4400)
    kv = make_input((2, 7, embed_dim), seed=4401)

    vy = vkml_mha(V.tensor(q), V.tensor(kv), V.tensor(kv))
    ty, _ = torch_mha(torch.from_numpy(q.copy()), torch.from_numpy(kv.copy()),
                      torch.from_numpy(kv.copy()), need_weights=False)

    assert_shape("MHA cross", vy, ty)
    assert_close("MHA cross", vy, ty, TOLERANCES["transcendental"], inputs=[q, kv])


def test_multihead_attention_rejects_indivisible_heads():
    with pytest.raises(ValueError):
        V.nn.MultiheadAttention(10, 3)


@pytest.mark.parametrize("norm_first", [False, True])
@pytest.mark.parametrize("activation", ["relu", "gelu"])
def test_transformer_encoder_layer(norm_first, activation):
    """Dropout is disabled on both sides: it is the one part that cannot match,
    since the two libraries draw from different generators by design
    (docs/ARCHITECTURE.md 7.2)."""
    d_model, nhead, ff = 8, 2, 16
    torch_layer = torch.nn.TransformerEncoderLayer(
        d_model, nhead, dim_feedforward=ff, dropout=0.0, activation=activation,
        batch_first=True, norm_first=norm_first)
    vkml_layer = V.nn.TransformerEncoderLayer(
        d_model, nhead, dim_feedforward=ff, dropout=0.0, activation=activation,
        norm_first=norm_first)
    _copy_weights(vkml_layer, torch_layer)
    torch_layer.eval(); vkml_layer.eval()

    x = make_input((2, 6, d_model), seed=4500)
    vx = V.tensor(x, requires_grad=True)
    tx = torch.from_numpy(x.copy()).requires_grad_(True)

    vy, ty = vkml_layer(vx), torch_layer(tx)
    assert_shape("encoder layer", vy, ty)
    assert_close(f"encoder layer(norm_first={norm_first},{activation})", vy, ty,
                 TOLERANCES["transcendental"], inputs=[x])

    V.sum(V.mul(vy, vy)).backward()
    (ty * ty).sum().backward()
    assert_close("encoder layer grad", vx.grad, tx.grad, TOLERANCES["transcendental"],
                 inputs=[x])


def test_transformer_stack_trains():
    """Two encoder layers plus an embedding and a head, trained for a few steps.

    Not compared against torch -- dropout is active and the generators differ by
    design. What is asserted is that the whole stack produces finite gradients
    and a falling loss, which a shape-only check would not.
    """
    vocab, d_model, seq = 20, 8, 5
    model = V.nn.Sequential(
        V.nn.Embedding(vocab, d_model),
        V.nn.TransformerEncoderLayer(d_model, 2, dim_feedforward=16, dropout=0.1, seed=7),
        V.nn.TransformerEncoderLayer(d_model, 2, dim_feedforward=16, dropout=0.1, seed=11),
        V.nn.Flatten(),
        V.nn.Linear(seq * d_model, 3),
    )
    model.train()
    opt = V.optim.Adam(model.parameters(), lr=1e-2)

    rng = np.random.default_rng(4600)
    tokens = V.tensor(rng.integers(0, vocab, size=(4, seq)).astype(np.int64))
    labels = V.tensor(rng.integers(0, 3, size=4).astype(np.int64))

    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = V.nn.cross_entropy(model(tokens), labels)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert all(np.isfinite(losses)), losses
    assert losses[-1] < losses[0], f"loss did not fall: {losses[0]:.4f} -> {losses[-1]:.4f}"


@pytest.mark.parametrize("weight_decay", [0.0, 0.01])
def test_sgd_nesterov_matches_torch_step_for_step(weight_decay):
    """Nesterov momentum against torch, every step.

    Compared per STEP rather than at the end, for the reason the classical
    momentum test gives: a wrong update rule still reaches a plausible final
    loss, and the two rules differ by one extra momentum term that only shows up
    as drift.

    weight_decay is parametrised because it is where the ordering can go wrong.
    Nesterov must reuse the DECAYED gradient in `g + momentum * buf`, not the
    raw one -- using the raw gradient there gives a subtly different trajectory
    that a final-loss check would very likely miss.
    """
    torch_model = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.ReLU(),
                                      torch.nn.Linear(8, 2))
    vkml_model = V.nn.Sequential(V.nn.Linear(6, 8), V.nn.ReLU(), V.nn.Linear(8, 2))
    _copy_weights(vkml_model, torch_model)

    x = make_input((10, 6), seed=2410)
    y = make_input((10, 2), seed=2411)

    v_opt = V.optim.SGD(vkml_model.parameters(), lr=0.05, momentum=0.9,
                        weight_decay=weight_decay, nesterov=True)
    t_opt = torch.optim.SGD(torch_model.parameters(), lr=0.05, momentum=0.9,
                            weight_decay=weight_decay, nesterov=True)

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

        assert_close(f"SGD(nesterov, wd={weight_decay}) loss @ step {step}", v_loss, t_loss,
                     Tol(1e-5, 1e-5))

    for (vn, vp), (tn, tp) in zip(vkml_model.named_parameters(),
                                  torch_model.named_parameters()):
        assert vn == tn
        assert_close(f"SGD nesterov final param {vn}", vp, tp, Tol(1e-5, 1e-5))


def test_sgd_nesterov_actually_differs_from_classical_momentum():
    """The NEGATIVE control for the test above.

    Without it, a `nesterov=True` that silently ignored the flag would still
    match torch on the classical path and pass everything else. The two rules
    must produce different parameters from identical starting conditions.
    """
    x = make_input((10, 6), seed=2412)
    y = make_input((10, 2), seed=2413)

    # ONE reference, built once. Constructing a fresh torch model per arm would
    # give the two arms different starting weights -- torch's global RNG is not
    # reset here -- and they would then differ for reasons that have nothing to
    # do with nesterov. That is not hypothetical: this test PASSED against an
    # implementation that ignored the flag entirely until it was fixed.
    torch_ref = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.ReLU(),
                                    torch.nn.Linear(8, 2))

    def train(nesterov):
        model = V.nn.Sequential(V.nn.Linear(6, 8), V.nn.ReLU(), V.nn.Linear(8, 2))
        _copy_weights(model, torch_ref)
        opt = V.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, nesterov=nesterov)
        for _ in range(5):
            opt.zero_grad()
            V.nn.mse_loss(model(V.tensor(x)), V.tensor(y)).backward()
            opt.step()
        return [p.numpy().copy() for p in model.parameters()]

    classical, nesterov = train(False), train(True)
    assert any(not np.allclose(a, b) for a, b in zip(classical, nesterov)), (
        "nesterov=True produced identical parameters to classical momentum; "
        "the flag is being ignored"
    )


def test_sgd_nesterov_requires_momentum():
    """Nesterov looks ahead along the momentum buffer. With no momentum there is
    nothing to look along and the update degenerates to plain SGD -- silently,
    which is worse than an error. torch rejects this too."""
    model = V.nn.Linear(3, 3)
    with pytest.raises(ValueError):
        V.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, nesterov=True)


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------
#
# THE ORACLE IS DIFFERENT IN KIND HERE, and the reason is worth stating.
# `torch.nn` has no PositionalEncoding — the reference implementation lives in a
# tutorial — so there is no torch class to compare against. Every other module
# in this file is validated against torch's own, which is stronger than a
# reference written twice.
#
# What replaces it is stronger still: the table has a CLOSED FORM (Vaswani et
# al. 2017 §3.5) with no implementation freedom, so the oracle is that formula
# evaluated in float64. A test that agreed with a second copy of the same code
# would prove only that both were written by the same person.


def _positional_reference(max_len, d_model, base=10000.0):
    """PE[pos, 2i] = sin(pos / base**(2i/d)), PE[pos, 2i+1] = cos(...), in float64.

    Written from the paper rather than from vkml/nn.py, deliberately: an oracle
    transcribed from the implementation checks transcription, not correctness.
    """
    table = np.empty((max_len, d_model), dtype=np.float64)
    for pos in range(max_len):
        for i in range(d_model // 2):
            angle = pos / (base ** (2.0 * i / d_model))
            table[pos, 2 * i] = math.sin(angle)
            table[pos, 2 * i + 1] = math.cos(angle)
    return table


@pytest.mark.parametrize("d_model,max_len", [(8, 16), (64, 200), (128, 512)])
def test_positional_encoding_matches_the_closed_form(d_model, max_len):
    pe = V.nn.PositionalEncoding(d_model=d_model, max_len=max_len)
    got = pe.pe.numpy().astype(np.float64)
    want = _positional_reference(max_len, d_model)

    # float32 round-off on a value in [-1, 1] and nothing else: the trig runs in
    # double and is narrowed once, so the only error admitted is the narrowing.
    assert got.shape == want.shape
    assert np.abs(got - want).max() <= np.finfo(np.float32).eps, (
        f"max |diff| {np.abs(got - want).max():.3e} exceeds one float32 eps; the table is "
        f"not the closed form to the last bit"
    )


def test_positional_encoding_first_rows_are_the_known_values():
    """Anchored on values anyone can check by hand, not on the generator."""
    pe = V.nn.PositionalEncoding(d_model=4, max_len=3)
    table = pe.pe.numpy()
    # pos=0: every angle is 0, so sin=0 and cos=1 all the way across.
    assert np.allclose(table[0], [0.0, 1.0, 0.0, 1.0], atol=1e-7)
    # pos=1, i=0: angle 1 exactly.
    assert table[1][0] == pytest.approx(math.sin(1.0), abs=1e-7)
    assert table[1][1] == pytest.approx(math.cos(1.0), abs=1e-7)
    # pos=1, i=1: angle 1 / 10000**(2/4) = 1/100.
    assert table[1][2] == pytest.approx(math.sin(0.01), abs=1e-7)
    assert table[1][3] == pytest.approx(math.cos(0.01), abs=1e-7)


@pytest.mark.parametrize("shape", [(10, 8), (3, 10, 8)])
def test_positional_encoding_adds_the_table_and_keeps_the_shape(shape):
    pe = V.nn.PositionalEncoding(d_model=8, max_len=32)
    x = np.random.default_rng(3).standard_normal(shape).astype(np.float32)
    got = pe(V.tensor(x)).numpy()

    seq = shape[0] if len(shape) == 2 else shape[1]
    want = x + pe.pe.numpy()[:seq]
    assert got.shape == x.shape
    assert np.allclose(got, want, atol=1e-6)


def test_positional_encoding_batch_first_false_uses_the_other_layout():
    """(S, B, E) must add along S, not along B. Getting this wrong still
    broadcasts and still produces a plausible tensor, so it is worth pinning."""
    pe = V.nn.PositionalEncoding(d_model=8, max_len=32, batch_first=False)
    x = np.random.default_rng(4).standard_normal((10, 3, 8)).astype(np.float32)
    got = pe(V.tensor(x)).numpy()
    want = x + pe.pe.numpy()[:10][:, None, :]
    assert np.allclose(got, want, atol=1e-6)


def test_positional_encoding_distinguishes_a_reordered_sequence():
    """The property the module exists for.

    Attention is permutation equivariant, so a sequence model without positions
    cannot tell "dog bites man" from "man bites dog". A test that only checked
    the table's values would pass on an encoding that added the same row
    everywhere.
    """
    pe = V.nn.PositionalEncoding(d_model=16, max_len=32)
    x = np.random.default_rng(5).standard_normal((1, 6, 16)).astype(np.float32)
    forward = pe(V.tensor(x)).numpy()
    reversed_ = pe(V.tensor(x[:, ::-1].copy())).numpy()
    assert not np.allclose(forward, reversed_[:, ::-1]), (
        "reversing the sequence produced the same encoded values in reverse; the "
        "encoding is not position-dependent"
    )


def test_positional_encoding_is_a_buffer_not_a_parameter():
    """It must not be trained, and it must move with `.to()`."""
    pe = V.nn.PositionalEncoding(d_model=8, max_len=16)
    assert list(pe.parameters()) == [], "the table must not be a trainable parameter"
    assert "pe" in dict(pe.named_buffers())
    assert "pe" in pe.state_dict()


def test_positional_encoding_rejects_what_it_cannot_represent():
    with pytest.raises(ValueError):
        V.nn.PositionalEncoding(d_model=7)          # odd: a sine with no cosine
    with pytest.raises(ValueError):
        V.nn.PositionalEncoding(d_model=0)
    with pytest.raises(ValueError):
        V.nn.PositionalEncoding(d_model=8, max_len=0)

    pe = V.nn.PositionalEncoding(d_model=8, max_len=4)
    with pytest.raises(ValueError):
        pe(V.tensor(np.zeros((1, 5, 8), dtype=np.float32)))    # longer than the table
    with pytest.raises(ValueError):
        pe(V.tensor(np.zeros((1, 4, 16), dtype=np.float32)))   # wrong d_model
    with pytest.raises(ValueError):
        pe(V.tensor(np.zeros((2, 2, 4, 8), dtype=np.float32)))  # rank the layout cannot mean


def test_a_transformer_with_positions_trains():
    """The point of adding this module: the parts now compose end to end.

    Embedding -> positions -> encoder layer -> pooled linear head, trained on a
    task that is UNLEARNABLE without positions -- the label is whether the first
    token is greater than the last, and a permutation-equivariant model cannot
    represent it.
    """
    V.nn.manual_seed(9)
    rng = np.random.default_rng(9)
    vocab, seq, d_model = 12, 6, 16

    tokens = rng.integers(0, vocab, size=(64, seq)).astype(np.int64)
    labels = (tokens[:, 0] > tokens[:, -1]).astype(np.int64)

    embed = V.nn.Embedding(vocab, d_model)
    pos = V.nn.PositionalEncoding(d_model=d_model, max_len=seq)
    block = V.nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=32, dropout=0.0)
    head = V.nn.Linear(d_model, 2)
    params = [*embed.parameters(), *block.parameters(), *head.parameters()]
    opt = V.optim.Adam(params, lr=3e-3)

    x = V.tensor(tokens)
    y = V.tensor(labels)

    def loss_now():
        h = block(pos(embed(x)))
        return V.nn.cross_entropy(head(V.mean(h, [1])), y)

    first = float(loss_now().item())
    for _ in range(60):
        opt.zero_grad()
        loss_now().backward()
        opt.step()
    last = float(loss_now().item())

    assert last < first * 0.6, (
        f"loss went {first:.4f} -> {last:.4f}; a transformer with positions should learn "
        f"an order-dependent label, and this one did not"
    )
