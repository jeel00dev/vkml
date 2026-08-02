#!/usr/bin/env python3
"""The manifesto's P1 module list, checked against the API that exists.

WHY THIS EXISTS. `PHASE2-MANIFESTO.md` names functional completeness as P1 and
lists the modules it means. `PROJECT-SCOPE-ANALYSIS.md` measured that list
against the code and got **three of them wrong in the same direction**:

    Attention / MultiHeadAttention   recorded absent · present as MultiheadAttention
    TransformerBlock                 recorded absent · present as TransformerEncoderLayer
    FeedForward                      recorded absent · present inside the encoder layer

All three had existed since `c12622b`, five days before the measurement. The
cause is one the document had *already confessed to once* in the same table:

> "I initially recorded all five losses as absent. That was a class-name grep
>  against an API that exposes them as functions."

It is the same mistake twice because the underlying condition is permanent.
**vkML deliberately uses PyTorch's spellings** — `nn.py` says so at the
TransformerEncoderLayer: *"Parameter names match torch.nn.TransformerEncoderLayer
so a state_dict loads unchanged"* — and the manifesto uses its own. A check
against either name alone is wrong for half the list, forever.

SO THE MAPPING IS DECLARED, NOT INFERRED. Every P1 item below names the API
symbol that satisfies it, or says why nothing does. That turns "did anyone
notice this got built?" from a grep somebody has to think of into a run.

WHAT IT DOES NOT CHECK. That a present module is *correct*, or complete against
torch's behaviour — the validation suite is what does that. This answers one
question: does the thing the manifesto asked for exist under some name.

    python scripts/check_module_coverage.py
    python scripts/check_module_coverage.py --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

# The manifesto's P1 list, each item mapped to what satisfies it.
#
#   ("manifesto name", "module.attr" or None, "note")
#
# A None target is an item nothing implements. The note is required either way:
# for a present item it explains a name that does not match, and for an absent
# one it is what a reader needs in order to decide whether to build it.
P1_MODULES: list[tuple[str, str | None, str]] = [
    ("Linear", "nn.Linear", ""),
    ("Conv1d", "nn.Conv1d",
     "composed from Conv2d with a height of 1 -- identical arithmetic, so a second "
     "kernel would be a second implementation of one algorithm"),
    ("Conv2d", "nn.Conv2d", ""),
    ("Conv3d", None, "absent; README lists it as a known gap"),
    ("Embedding", "nn.Embedding", ""),
    ("BatchNorm", "nn.BatchNorm2d", "2-D only; 1-D and 3-D not required by P1"),
    ("LayerNorm", "nn.LayerNorm", ""),
    ("Dropout", "nn.Dropout", ""),
    ("MaxPool2d", "nn.MaxPool2d", ""),
    ("AvgPool2d", "nn.AvgPool2d", ""),
    ("Flatten", "nn.Flatten", ""),
    ("Sequential", "nn.Sequential", ""),
    # The three the scope analysis got wrong, and why the name differs.
    ("Attention / MultiHeadAttention", "nn.MultiheadAttention",
     "torch's spelling -- lower-case h -- so a state_dict loads unchanged"),
    ("TransformerBlock", "nn.TransformerEncoderLayer", "torch's name for the same thing"),
    ("FeedForward", "nn.TransformerEncoderLayer",
     "not a standalone module: it is the encoder layer's linear1/activation/linear2 "
     "branch, which is where torch keeps it too"),
    ("PositionalEncoding", "nn.PositionalEncoding",
     "sinusoidal, Vaswani et al. 3.5. LEARNED positions are not a separate module "
     "and do not need one: they are Embedding(max_len, d_model) indexed by "
     "position, which is what GPT does"),
    # Optimisers and losses.
    ("SGD", "optim.SGD", ""),
    ("Momentum", "optim.SGD", "a parameter of SGD, as in torch"),
    ("Adam", "optim.Adam", ""),
    ("AdamW", "optim.AdamW", ""),
    ("RMSProp", "optim.RMSProp", ""),
    ("MSE loss", "nn.mse_loss", "a function, not a class -- as in torch.nn.functional"),
    ("CrossEntropy loss", "nn.cross_entropy", "a function"),
    ("BCE loss", "nn.binary_cross_entropy_with_logits", "a function"),
    ("KL divergence loss", "nn.kl_div", "a function"),
    ("Huber loss", "nn.huber_loss", "a function"),
    # Data and serialisation.
    ("Dataset", "data.Dataset", ""),
    ("DataLoader", "data.DataLoader", ""),
    ("DataLoader batching", "data.DataLoader", "the batch_size argument"),
    ("DataLoader shuffling", "data.DataLoader", "the shuffle argument"),
    ("DataLoader prefetch", None, "absent; tracker #22"),
    ("DataLoader transforms", None, "absent; tracker #23"),
    ("model save / load / checkpoint", "Checkpoint", ""),
    ("autograd checkpointing", None,
     "absent, and NOT the same thing as `Checkpoint`, which is model "
     "serialisation. Recomputing activations to trade compute for memory has no "
     "equivalent here"),
]


def resolve(vkml, dotted: str):
    """`nn.Linear` -> the attribute, or None if any step is missing."""
    obj = vkml
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print every item and its state")
    args = ap.parse_args()

    import vkml

    present, absent, broken = [], [], []
    for name, target, note in P1_MODULES:
        if target is None:
            absent.append((name, note))
        elif resolve(vkml, target) is None:
            broken.append((name, target, note))
        else:
            present.append((name, target, note))

    if args.list:
        print(f"\n  MANIFESTO P1 MODULES — {len(P1_MODULES)} items\n")
        for name, target, note in present:
            print(f"  ok  {name:<32} {target}")
            if note:
                print(f"        {note}")
        for name, note in absent:
            print(f"  --  {name:<32} not implemented")
            print(f"        {note}")

    print(f"  {len(present)} of {len(P1_MODULES)} P1 modules present, "
          f"{len(absent)} not implemented")

    # ABSENCE IS NOT A FAILURE. The manifesto is a plan, and an unbuilt item on
    # it is work, not a defect. What fails is a DECLARATION that no longer
    # matches the code: a target this file names that has been renamed or
    # removed, which would leave the count quietly wrong in the other
    # direction -- the exact failure this gate was written after.
    if broken:
        print(f"\n  {len(broken)} declared target(s) that no longer resolve:", file=sys.stderr)
        for name, target, _ in broken:
            print(f"    {name:<32} expected {target}", file=sys.stderr)
        print("\n  Either the module was renamed -- update the mapping -- or it was removed,\n"
              "  in which case this item is no longer satisfied and should say so.\n",
              file=sys.stderr)
        return 1

    if absent:
        print("  not implemented: " + ", ".join(name for name, _ in absent))
    return 0


if __name__ == "__main__":
    sys.exit(main())
