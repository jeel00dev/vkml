"""Train MNIST on vkml and check it against PyTorch step by step.

This is docs/ARCHITECTURE.md's M0 gate made real: an MLP that trains, with a
loss curve matched against PyTorch rather than merely plausible. Every unit
test in the suite compares one operator; this compares a whole training run,
which is the only thing that can catch an error that is individually within
tolerance but compounds.

Both frameworks start from BYTE-IDENTICAL weights, copied rather than
re-seeded, and see the same batches in the same order. RNG parity is not a goal
(docs/ARCHITECTURE.md 7.2), so anything stochastic would make the comparison
meaningless -- there is no dropout here for that reason.

Usage:
    python examples/mnist/train.py                 # MLP on the GPU
    python examples/mnist/train.py --model cnn     # convolutional
    python examples/mnist/train.py --device cpu    # reference backend
    python examples/mnist/train.py --no-compare    # skip the torch run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# vkml is built in-tree rather than installed, so the package directory has to
# be on the path. Dropped once there is a real install step.
import sys
sys.path.insert(0, str(HERE.parent.parent / "python"))

import vkml as V  # noqa: E402

import data as mnist_data  # noqa: E402


def build_mlp() -> V.nn.Module:
    return V.nn.Sequential(
        V.nn.Flatten(),
        V.nn.Linear(784, 128), V.nn.ReLU(),
        V.nn.Linear(128, 10),
    )


def build_cnn() -> V.nn.Module:
    return V.nn.Sequential(
        V.nn.Conv2d(1, 8, 3, padding=1), V.nn.ReLU(), V.nn.MaxPool2d(2),
        V.nn.Conv2d(8, 16, 3, padding=1), V.nn.ReLU(), V.nn.MaxPool2d(2),
        V.nn.Flatten(),
        V.nn.Linear(16 * 7 * 7, 10),
    )


def build_torch_mlp():
    import torch
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(784, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 10),
    )


def build_torch_cnn():
    import torch
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(8, 16, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(16 * 7 * 7, 10),
    )


MODELS = {
    "mlp": (build_mlp, build_torch_mlp),
    "cnn": (build_cnn, build_torch_cnn),
}


def batches(n: int, batch_size: int, rng: np.random.Generator):
    """Shuffled index batches. The trailing partial batch is dropped so both
    frameworks always see the same shapes."""
    order = rng.permutation(n)
    for start in range(0, n - batch_size + 1, batch_size):
        yield order[start:start + batch_size]


def evaluate(model: V.nn.Module, x: np.ndarray, y: np.ndarray, device,
             batch_size: int = 500) -> tuple[float, float]:
    """Accuracy and mean loss over a dataset, in inference mode."""
    model.eval()
    correct = 0
    loss_sum = 0.0
    seen = 0

    with V.no_grad():
        for start in range(0, len(x), batch_size):
            xb = V.tensor(x[start:start + batch_size], device=device)
            yb_np = y[start:start + batch_size]
            logits = model(xb)
            loss = V.nn.cross_entropy(logits, V.tensor(yb_np, device=device))

            predicted = np.argmax(logits.numpy(), axis=-1)
            correct += int((predicted == yb_np).sum())
            loss_sum += float(loss.item()) * len(yb_np)
            seen += len(yb_np)

    model.train()
    return correct / seen, loss_sum / seen


def train(args) -> dict:
    dataset = mnist_data.load()
    device = V.cpu
    if args.device != "cpu":
        V.init_vulkan(0)
        device = V.device(args.device)

    train_x, train_y = dataset["train_x"], dataset["train_y"]
    test_x, test_y = dataset["test_x"], dataset["test_y"]
    if args.train_size:
        train_x, train_y = train_x[:args.train_size], train_y[:args.train_size]

    # Seed the weight initialisation, or the run cannot be reproduced -- and a
    # training result nobody can reproduce cannot be investigated when it
    # surprises you.
    V.nn.manual_seed(args.seed)

    build, build_torch = MODELS[args.model]
    model = build()
    initial_state = model.state_dict()
    model.to(device)
    optimiser = V.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    torch_model = torch_opt = None
    if args.compare:
        import torch
        torch_model = build_torch()
        torch_model.load_state_dict(
            {k: torch.from_numpy(v.copy()) for k, v in initial_state.items()})
        torch_opt = torch.optim.SGD(torch_model.parameters(), lr=args.lr, momentum=0.9)

    history = {"step": [], "vkml_loss": [], "torch_loss": [], "epoch_seconds": [],
               "test_accuracy": [], "test_loss": [], "train_accuracy": []}
    max_divergence = 0.0
    step = 0
    started = time.perf_counter()

    for epoch in range(args.epochs):
        rng = np.random.default_rng(args.seed + epoch)
        epoch_started = time.perf_counter()

        for index in batches(len(train_x), args.batch_size, rng):
            xb = V.tensor(train_x[index], device=device)
            yb = V.tensor(train_y[index], device=device)

            optimiser.zero_grad()
            loss = V.nn.cross_entropy(model(xb), yb)
            loss.backward()
            optimiser.step()
            vkml_loss = float(loss.item())

            torch_loss = float("nan")
            if args.compare:
                import torch
                torch_opt.zero_grad()
                t_loss = torch.nn.functional.cross_entropy(
                    torch_model(torch.from_numpy(train_x[index].copy())),
                    torch.from_numpy(train_y[index].copy()))
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

    # Torch's accuracy too, when comparing. Without it the report can say the
    # loss curves diverged but not whether that mattered -- and for a chaotic
    # optimisation the interesting question is whether the two runs end up
    # equally good, not whether they took the same path.
    torch_accuracy = None
    if args.compare:
        import torch
        torch_model.eval()
        correct = 0
        with torch.no_grad():
            for start in range(0, len(test_x), 500):
                logits = torch_model(torch.from_numpy(test_x[start:start + 500].copy()))
                predicted = logits.argmax(dim=-1).numpy()
                correct += int((predicted == test_y[start:start + 500]).sum())
        torch_accuracy = correct / len(test_x)

    # numpy persistence, not a checkpoint format: state_dict already returns
    # arrays, and designing a real format is separate work. Enough for the GUI
    # to load what was trained here.
    weights_path = HERE / f"{args.model}_weights.npz"
    np.savez(weights_path, **model.state_dict())

    summary = {
        "model": args.model,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "train_examples": len(train_x),
        "test_examples": len(test_x),
        "steps": step,
        "total_seconds": total_seconds,
        "seconds_per_epoch": history["epoch_seconds"],
        "final_test_accuracy": accuracy,
        "final_test_loss": test_loss,
        "final_train_accuracy": history["train_accuracy"][-1],
        "max_loss_divergence_vs_torch": max_divergence if args.compare else None,
        "torch_test_accuracy": torch_accuracy,
        "weights": str(weights_path),
        "history": history,
    }
    (HERE / f"{args.model}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), default="mlp")
    parser.add_argument("--device", default="vulkan:0", help="vulkan:0 or cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--train-size", type=int, default=0,
                        help="use only the first N training examples (0 = all)")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--no-compare", dest="compare", action="store_false",
                        help="skip the PyTorch reference run")
    args = parser.parse_args()

    summary = train(args)

    print()
    print("=" * 62)
    print(f"  model              {summary['model']}  on  {summary['device']}")
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
    print(f"  weights            {summary['weights']}")
    print("=" * 62)


if __name__ == "__main__":
    main()
