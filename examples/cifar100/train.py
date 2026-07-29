"""Train CIFAR-100 on vkml and check it against PyTorch step by step.

The successor to examples/mnist/train.py, and structured the same way: both
frameworks start from BYTE-IDENTICAL weights, copied rather than re-seeded, and
see the same batches in the same order. Nothing stochastic is in the model --
no dropout -- because RNG parity is explicitly not a goal
(docs/ARCHITECTURE.md 7.2) and a random mask would make the comparison
meaningless.

WHAT THIS IS FOR, beyond a second dataset. MNIST is too small to say anything
about where time goes: 28x28x1 through two convolutions finishes before the
measurement noise does. CIFAR-100 is 32x32x3 through three, with a 100-way head,
which is the smallest thing in this repository that can produce a PROFILE worth
acting on.

ACCURACY IS NOT THE CRITERION HERE. A small CNN on 100 classes lands far below
the number the same architecture reaches on 10, and chasing it would mean
augmentation and normalisation this example deliberately omits. What is being
tested is agreement with PyTorch, step by step.

Usage:
    python examples/cifar100/train.py                  # GPU if one is present
    python examples/cifar100/train.py --device cpu     # reference backend
    python examples/cifar100/train.py --no-compare     # skip the torch run
    python examples/cifar100/train.py --epochs 1 --train-size 5000   # quick pass
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Prefer an installed vkml; fall back to the in-tree package when there is not
# one, so this runs both from a clone and after `pip install .`.
import sys
import importlib.util
if importlib.util.find_spec("vkml") is None:
    sys.path.insert(0, str(HERE.parent.parent / "python"))

import vkml as V  # noqa: E402
from vkml.data import ArrayDataset, DataLoader  # noqa: E402

import data as cifar_data  # noqa: E402

CLASSES = 100

# 32x32 -> 16 -> 8 -> 4 through three pooled convolutions, then one linear head.
# Three blocks rather than MNIST's two: the point of this example is to put real
# work through the conv path, and the third block is where the channel count
# gets large enough for the GEMM inside im2col to matter.
#
# No BatchNorm and no augmentation. Both would raise the accuracy and neither
# would make the comparison against torch any sharper, which is what this run is
# for. Stated so their absence reads as a decision.
CONV_CHANNELS = (32, 64, 128)
FEATURES = CONV_CHANNELS[-1] * 4 * 4


def build_cnn() -> V.nn.Module:
    c1, c2, c3 = CONV_CHANNELS
    return V.nn.Sequential(
        V.nn.Conv2d(3, c1, 3, padding=1), V.nn.ReLU(), V.nn.MaxPool2d(2),
        V.nn.Conv2d(c1, c2, 3, padding=1), V.nn.ReLU(), V.nn.MaxPool2d(2),
        V.nn.Conv2d(c2, c3, 3, padding=1), V.nn.ReLU(), V.nn.MaxPool2d(2),
        V.nn.Flatten(),
        V.nn.Linear(FEATURES, CLASSES),
    )


def build_torch_cnn():
    import torch
    c1, c2, c3 = CONV_CHANNELS
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, c1, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(c1, c2, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(c2, c3, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(FEATURES, CLASSES),
    )


def resolve_device(name: str):
    """Turns a --device argument into a device, initialising Vulkan if needed.

    Same contract as the MNIST example: `auto` falls back to the CPU, a named
    device does not -- someone who typed `vulkan:1` wants that GPU, and quietly
    handing back the CPU would hide the thing they asked about.
    """
    if name == "cpu":
        return V.cpu

    if name == "auto":
        if not (V.has_vulkan and V.vulkan_available()):
            print("no Vulkan device found; running on the CPU")
            return V.cpu
        name = "vulkan:0"

    _, _, index = name.partition(":")
    if not index.isdigit():
        raise SystemExit(f"expected a device like 'vulkan:0', got '{name}'")

    V.init_vulkan(int(index))
    return V.device(name)


def evaluate(model: V.nn.Module, x: np.ndarray, y: np.ndarray, device,
             batch_size: int = 500) -> tuple[float, float]:
    """Accuracy and mean loss over a dataset, in inference mode."""
    model.eval()
    correct = 0
    loss_sum = 0.0
    seen = 0

    # Neither shuffled nor drop_last: every example must be scored exactly once,
    # and a dropped tail would quietly change the denominator.
    loader = DataLoader(ArrayDataset(x, y), batch_size=batch_size)

    with V.no_grad():
        for xb_np, yb_np in loader:
            logits = model(V.tensor(xb_np, device=device))
            loss = V.nn.cross_entropy(logits, V.tensor(yb_np, device=device))

            predicted = np.argmax(logits.numpy(), axis=-1)
            correct += int((predicted == yb_np).sum())
            loss_sum += float(loss.item()) * len(yb_np)
            seen += len(yb_np)

    model.train()
    return correct / seen, loss_sum / seen


def train(args) -> dict:
    dataset = cifar_data.load()
    device = resolve_device(args.device)

    train_x, train_y = dataset["train_x"], dataset["train_y"]
    test_x, test_y = dataset["test_x"], dataset["test_y"]
    if args.train_size:
        train_x, train_y = train_x[:args.train_size], train_y[:args.train_size]

    V.nn.manual_seed(args.seed)

    model = build_cnn()
    initial_state = model.state_dict()
    model.to(device)
    optimiser = V.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    torch_model = torch_opt = None
    if args.compare:
        import torch
        torch_model = build_torch_cnn()
        torch_model.load_state_dict(
            {k: torch.from_numpy(v.copy()) for k, v in initial_state.items()})
        torch_opt = torch.optim.SGD(torch_model.parameters(), lr=args.lr, momentum=0.9)

    history = {"step": [], "vkml_loss": [], "torch_loss": [], "epoch_seconds": [],
               "test_accuracy": [], "test_loss": [], "train_accuracy": []}
    max_divergence = 0.0
    step = 0
    started = time.perf_counter()

    # WHERE THE TIME GOES, in three buckets that answer different questions.
    #
    #   batch     the loader: fancy-indexing numpy in the calling thread. This is
    #             the ONLY bucket prefetching could hide, which is why it is
    #             measured separately rather than folded into a single wall time.
    #   transfer  host to device upload. Prefetch does not touch this; hiding it
    #             would need asynchronous copies, a different change entirely.
    #   compute   forward, backward, optimiser step, and the .item() that forces
    #             the lazy graph to realise -- so this includes the GPU wait.
    #
    # python/vkml/data.py records prefetch as deliberately absent, to be
    # revisited "when a profile shows the training loop waiting on data". These
    # numbers are that profile.
    seconds = {"batch": 0.0, "transfer": 0.0, "compute": 0.0}

    loader = DataLoader(ArrayDataset(train_x, train_y), batch_size=args.batch_size,
                        shuffle=True, drop_last=True, seed=args.seed)

    for epoch in range(args.epochs):
        epoch_started = time.perf_counter()

        # An explicit iterator, so the time spent producing a batch is
        # attributable. A `for` loop hides it inside the loop machinery.
        iterator = iter(loader)
        while True:
            mark = time.perf_counter()
            try:
                xb_np, yb_np = next(iterator)
            except StopIteration:
                break
            seconds["batch"] += time.perf_counter() - mark

            mark = time.perf_counter()
            xb = V.tensor(xb_np, device=device)
            yb = V.tensor(yb_np, device=device)
            seconds["transfer"] += time.perf_counter() - mark

            mark = time.perf_counter()
            optimiser.zero_grad()
            loss = V.nn.cross_entropy(model(xb), yb)
            loss.backward()
            optimiser.step()
            vkml_loss = float(loss.item())
            seconds["compute"] += time.perf_counter() - mark

            torch_loss = float("nan")
            if args.compare:
                import torch
                torch_opt.zero_grad()
                t_loss = torch.nn.functional.cross_entropy(
                    torch_model(torch.from_numpy(xb_np)), torch.from_numpy(yb_np))
                t_loss.backward()
                torch_opt.step()
                torch_loss = float(t_loss.item())
                max_divergence = max(max_divergence, abs(vkml_loss - torch_loss))

            if step % args.log_every == 0:
                history["step"].append(step)
                history["vkml_loss"].append(vkml_loss)
                history["torch_loss"].append(torch_loss)
                divergence = (f"  |diff| {abs(vkml_loss - torch_loss):.2e}"
                              if args.compare else "")
                print(f"  step {step:5d}  loss {vkml_loss:.6f}"
                      + (f"  torch {torch_loss:.6f}{divergence}" if args.compare else ""),
                      flush=True)
            step += 1

        epoch_seconds = time.perf_counter() - epoch_started
        accuracy, test_loss = evaluate(model, test_x, test_y, device)
        train_accuracy, _ = evaluate(model, train_x[:5000], train_y[:5000], device)

        history["epoch_seconds"].append(epoch_seconds)
        history["test_accuracy"].append(accuracy)
        history["test_loss"].append(test_loss)
        history["train_accuracy"].append(train_accuracy)
        print(f"epoch {epoch + 1}/{args.epochs}  {epoch_seconds:6.2f}s  "
              f"train acc {train_accuracy * 100:5.2f}%  "
              f"test acc {accuracy * 100:5.2f}%  test loss {test_loss:.4f}", flush=True)

    total_seconds = time.perf_counter() - started
    accuracy, test_loss = evaluate(model, test_x, test_y, device)

    torch_accuracy = None
    if args.compare:
        import torch
        torch_model.eval()
        correct = 0
        with torch.no_grad():
            for xb_np, yb_np in DataLoader(ArrayDataset(test_x, test_y), batch_size=500):
                predicted = torch_model(torch.from_numpy(xb_np)).argmax(dim=-1).numpy()
                correct += int((predicted == yb_np).sum())
        torch_accuracy = correct / len(test_x)

    weights_path = HERE / "cnn.vkml"
    V.save_module(weights_path, model, metadata={
        "model": "cnn",
        "dataset": "cifar100",
        "device": str(device),
        "epochs": args.epochs,
        "steps": step,
        "test_accuracy": accuracy,
    })

    stepped = sum(seconds.values())
    summary = {
        "model": "cnn",
        "dataset": "cifar100",
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "train_examples": len(train_x),
        "test_examples": len(test_x),
        "steps": step,
        "total_seconds": total_seconds,
        "seconds_per_epoch": history["epoch_seconds"],
        "step_seconds": seconds,
        "step_fractions": {k: (v / stepped if stepped else 0.0) for k, v in seconds.items()},
        "final_test_accuracy": accuracy,
        "final_test_loss": test_loss,
        "final_train_accuracy": history["train_accuracy"][-1],
        "max_loss_divergence_vs_torch": max_divergence if args.compare else None,
        "torch_test_accuracy": torch_accuracy,
        "weights": str(weights_path),
        "history": history,
    }
    (HERE / "cnn_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto",
                        help="auto (GPU if present), cpu, or vulkan:N")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--train-size", type=int, default=0,
                        help="use only the first N training examples (0 = all)")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--no-compare", dest="compare", action="store_false",
                        help="skip the PyTorch reference run")
    args = parser.parse_args()

    summary = train(args)
    fractions = summary["step_fractions"]

    print()
    print("=" * 62)
    print(f"  model              cnn on cifar100  on  {summary['device']}")
    print(f"  train / test       {summary['train_examples']} / {summary['test_examples']}")
    print(f"  epochs x batch     {summary['epochs']} x {summary['batch_size']}"
          f"   ({summary['steps']} steps)")
    print(f"  total time         {summary['total_seconds']:.2f}s"
          f"   ({np.mean(summary['seconds_per_epoch']):.2f}s/epoch)")
    print(f"  train accuracy     {summary['final_train_accuracy'] * 100:.2f}%")
    print(f"  TEST ACCURACY      {summary['final_test_accuracy'] * 100:.2f}%")
    print(f"  test loss          {summary['final_test_loss']:.4f}")
    if summary["max_loss_divergence_vs_torch"] is not None:
        print(f"  torch accuracy     {summary['torch_test_accuracy'] * 100:.2f}%"
              f"   (delta {abs(summary['final_test_accuracy'] - summary['torch_test_accuracy']) * 100:.2f} pp)")
        print(f"  max |vkml - torch| {summary['max_loss_divergence_vs_torch']:.3e}"
              "   (loss, over every training step)")
    print("  where the step goes:")
    for name in ("batch", "transfer", "compute"):
        print(f"    {name:9}        {summary['step_seconds'][name]:7.2f}s"
              f"   {fractions[name] * 100:5.1f}%")
    print(f"  weights            {summary['weights']}")
    print("=" * 62)


if __name__ == "__main__":
    main()
