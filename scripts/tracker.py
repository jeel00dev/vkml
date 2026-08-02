#!/usr/bin/env python3
"""The tracker as a map of engineering work, not a list of issues.

Two INDEPENDENT dimensions, because forcing one hierarchy to encode both loses
the information that makes the tracker worth reading:

  LIFECYCLE  what kind of claim the task makes about reality
  PURPOSE    what kind of engineering work it is

They are orthogonal on purpose. "Confirmed defect + Product Evolution" is a real
combination (#19: a genuine defect whose fix was a speedup) and so is
"Engineering debt + Verification" (#29). Collapsing them would have hidden both.

A SIXTH LIFECYCLE VALUE, `planned`, is used here and is NOT one of the five that
were proposed. It is reported separately rather than folded in, because the
finding is that ~40% of this tracker is roadmap work -- the milestones that built
the product -- and none of "confirmed defect / engineering debt / design decision
/ deferred / speculative" describes it. Those five are a taxonomy of PROBLEMS.
A tracker that only holds problems cannot show what the project spent its time
on, and this one spent most of it building.

THE DELETION QUESTION. Every open task carries `delete_if`: the evidence that
would let it be closed WITHOUT implementing it. This is not hypothetical --
it has already happened four times here:

  #76  refuted. The 38.6% regression was GPU clock state; the code was faster.
  #9   premise disproven. The OpKinds it would have added were removed instead.
  #16  answered by a decision. Vulkan is all-or-nothing, and that is written down.
  #28  answered by a reason. prod is CPU-only because a parallel reduction
       reorders the fold, which for a product is a different answer.

Implementation is not the default path to closure. Better measurement, better
tooling and better understanding all close tasks, and they close them cheaper.

    python scripts/tracker.py            # the report
    python scripts/tracker.py --matrix   # the lifecycle x purpose grid
"""
from __future__ import annotations

import argparse
import collections
import sys

# lifecycle
DEFECT, DEBT, DESIGN, DEFERRED, SPEC, PLANNED = (
    "confirmed defect", "engineering debt", "design decision",
    "deferred", "speculative", "planned")
# purpose
CORRECT, VERIFY, OBSERVE, KNOW, PRODUCT = (
    "Correctness", "Verification", "Observability",
    "Knowledge Preservation", "Product Evolution")
# leverage -- NOT effort and NOT priority. "If I finish this, what becomes
# easier, or unnecessary?" Recorded for OPEN tasks only: on closed ones it is
# hindsight, and hindsight always reports that the work was leveraged.
LOCAL, SHARED, FOUND, MULT = ("local", "shared", "foundational", "multiplicative")

# id: (state, lifecycle, purpose, note-or-delete_if, leverage)
# `delete_if` is recorded for open tasks; for closed ones the note says how it
# actually closed, so the four no-implementation closures stay visible.
T: dict[int, tuple] = {
    1:  ("done", PLANNED,  PRODUCT, "", None),
    2:  ("done", PLANNED,  PRODUCT, "", None),
    3:  ("done", PLANNED,  PRODUCT, "", None),
    4:  ("done", PLANNED,  VERIFY,  "", None),
    5:  ("done", PLANNED,  PRODUCT, "", None),
    6:  ("done", PLANNED,  PRODUCT, "", None),
    7:  ("done", PLANNED,  PRODUCT, "", None),
    8:  ("done", PLANNED,  PRODUCT, "", None),
    9:  ("done", DESIGN,   PRODUCT, "CLOSED WITHOUT IMPLEMENTING: premise disproven", None),
    10: ("done", PLANNED,  PRODUCT, "", None),
    11: ("done", PLANNED,  VERIFY,  "", None),
    12: ("done", PLANNED,  PRODUCT, "", None),
    13: ("done", PLANNED,  PRODUCT, "", None),
    14: ("done", PLANNED,  VERIFY,  "", None),
    15: ("done", PLANNED,  VERIFY,  "", None),
    16: ("done", DESIGN,   KNOW,    "CLOSED WITHOUT IMPLEMENTING: decided all-or-nothing", None),
    17: ("open", DEFERRED, PRODUCT, "profiling showing layer_norm is not hot -- needs O3", LOCAL),
    18: ("open", DEFERRED, PRODUCT, "measurement showing scatter_add is never hot", LOCAL),
    19: ("done", DEFECT,   PRODUCT, "", None),
    20: ("done", DEBT,     PRODUCT, "", None),
    21: ("done", PLANNED,  VERIFY,  "", None),
    22: ("open", PLANNED,  PRODUCT, "RECLASSIFIED from speculative: PHASE2-MANIFESTO P1 names "
                                     "DataLoader prefetch as functional completeness", LOCAL),
    23: ("open", PLANNED,  PRODUCT, "RECLASSIFIED from speculative: PHASE2-MANIFESTO P1 names "
                                     "DataLoader transforms as functional completeness", LOCAL),
    24: ("done", DESIGN,   CORRECT, "", None),
    25: ("done", DEBT,     PRODUCT, "AMBIGUOUS: legal/distribution, fits no purpose well", None),
    26: ("done", DESIGN,   PRODUCT, "", None),
    27: ("done", DESIGN,   CORRECT, "", None),
    28: ("done", DESIGN,   KNOW,    "CLOSED WITHOUT IMPLEMENTING: reason recorded", None),
    29: ("done", DEBT,     VERIFY,  "", None),
    30: ("done", PLANNED,  PRODUCT, "", None),
    31: ("done", PLANNED,  PRODUCT, "", None),
    32: ("open", DEFERRED, PRODUCT, "measurement showing tile load is not the bottleneck -- needs O3", LOCAL),
    33: ("done", PLANNED,  VERIFY,  "", None),
    34: ("done", PLANNED,  VERIFY,  "", None),
    35: ("done", PLANNED,  VERIFY,  "", None),
    36: ("done", PLANNED,  PRODUCT, "", None),
    37: ("done", PLANNED,  PRODUCT, "", None),
    38: ("open", DEFERRED, PRODUCT, "measurement showing per-parameter dispatch is not the cost", LOCAL),
    39: ("open", DEFERRED, PRODUCT, "measurement showing detach's realization is not hot", SHARED),
    40: ("open", DEFERRED, PRODUCT, "blocked by #39; same evidence closes both", LOCAL),
    41: ("done", DEFECT,   CORRECT, "", None),
    42: ("done", PLANNED,  VERIFY,  "", None),
    43: ("done", PLANNED,  KNOW,    "", None),
    44: ("done", PLANNED,  KNOW,    "", None),
    45: ("done", PLANNED,  KNOW,    "", None),
    46: ("done", PLANNED,  KNOW,    "", None),
    47: ("done", PLANNED,  KNOW,    "", None),
    48: ("done", PLANNED,  KNOW,    "", None),
    49: ("done", PLANNED,  KNOW,    "", None),
    50: ("done", PLANNED,  KNOW,    "", None),
    51: ("done", PLANNED,  KNOW,    "", None),
    52: ("done", PLANNED,  KNOW,    "", None),
    53: ("done", PLANNED,  KNOW,    "", None),
    54: ("done", PLANNED,  KNOW,    "", None),
    55: ("done", PLANNED,  KNOW,    "", None),
    56: ("done", PLANNED,  KNOW,    "", None),
    57: ("done", PLANNED,  KNOW,    "", None),
    58: ("done", PLANNED,  KNOW,    "", None),
    59: ("done", PLANNED,  KNOW,    "", None),
    60: ("done", PLANNED,  KNOW,    "", None),
    61: ("done", PLANNED,  KNOW,    "", None),
    62: ("done", PLANNED,  KNOW,    "", None),
    63: ("done", PLANNED,  KNOW,    "", None),
    64: ("done", PLANNED,  KNOW,    "", None),
    65: ("done", PLANNED,  KNOW,    "", None),
    66: ("done", PLANNED,  KNOW,    "documented the env switches O1 found are invisible at runtime", None),
    67: ("done", PLANNED,  VERIFY,  "", None),
    68: ("done", PLANNED,  VERIFY,  "", None),
    69: ("done", DEBT,     CORRECT, "", None),
    70: ("done", DEBT,     VERIFY,  "", None),
    71: ("done", DEBT,     VERIFY,  "", None),
    72: ("done", DEFECT,   VERIFY,  "", None),
    73: ("done", DEFECT,   KNOW,    "", None),
    74: ("done", DESIGN,   PRODUCT, "", None),
    75: ("open", DEFECT,   PRODUCT, "a green CI run on the fixed image -- fixed in code, unconfirmed", LOCAL),
    76: ("done", DEFECT,   VERIFY,  "CLOSED WITHOUT IMPLEMENTING: refuted, it was clock state", None),
    77: ("done", DEFECT,   VERIFY,  "", None),
    78: ("done", DEFECT,   VERIFY,  "", None),
    79: ("open", DESIGN,   KNOW,    "a decision either way -- this is a question, not work", SHARED),
    80: ("done", DEBT,     PRODUCT, "", None),
    81: ("done", DEFECT,   PRODUCT, "", None),
    82: ("done", DEFECT,   PRODUCT, "", None),
    83: ("done", DEFECT,   PRODUCT, "", None),
    84: ("done", DEFECT,   PRODUCT, "", None),
    85: ("done", DEFECT,   PRODUCT, "", None),
    86: ("done", DEFECT,   PRODUCT, "", None),
    87: ("done", DEFECT,   PRODUCT, "", None),
    88: ("done", DEBT,     PRODUCT, "", None),
    89: ("open", DEBT,     PRODUCT, "a measurement showing the long tail is intentional variation", LOCAL),
    90: ("done", DEBT,     PRODUCT, "", None),
    91: ("done", DEFECT,   KNOW,    "", None),
    92: ("open", DESIGN,   PRODUCT, "already mostly closed: 3 accepted, 4 rejected with reasons", LOCAL),
    93: ("open", DEFECT,   OBSERVE, "a decomposition showing the 4 warnings are host-side scheduling", SHARED),
    94: ("done", DEFECT,   VERIFY,  "", None),
    95: ("open", DEFECT,   VERIFY,  "done in working tree; closes on commit", SHARED),
    96: ("open", DEFECT,   OBSERVE, "cannot be deleted: 18 switches, measured 14.6%; a Configuration consumer of O3", FOUND),
    97: ("open", DEFECT,   OBSERVE, "deleted as separate work by O3 -- becomes one decision site", LOCAL),
    98: ("done", DEBT,     OBSERVE, "CLOSED: the category-3 marker in vkvalidate.py is deleted; the "
                                     "test now observes the backend instead of reimplementing it", None),
    # ---- Performance, re-sequenced by EXTENSIBILITY-ROADMAP 4a ----
    99:  ("done", DEFECT,   OBSERVE, "CLOSED: DispatchId joined cost to choice, then intervals and "
                                     "submission retention made the join ADD UP; a CIFAR step now "
                                     "prints a per-kernel table with the remainder shown", None),
    100: ("open", DEFECT,   PRODUCT, "PREMISE CORRECTED by #99: submission cost is 44.8%, not 74%. "
                                     "Still the largest single item and larger than any kernel, so "
                                     "the work stands with a lower ceiling", FOUND),
    101: ("open", DESIGN,   PRODUCT, "verifying convolution is NOT materialised im2col+GEMM; the "
                                     "roadmap says verify before designing", SHARED),
    102: ("open", DEFERRED, PRODUCT, "M3.2: P0+P1 showing arithmetic is not the remaining cost", SHARED),
    103: ("open", DEFERRED, PRODUCT, "M3.3: the #102 search finding no configuration the two-level "
                                     "hierarchy ceilings", LOCAL),
    104: ("open", DEFERRED, PRODUCT, "M3.5: attribution showing epilogue stores are not measurable", LOCAL),
    105: ("open", DEFECT,   PRODUCT, "a decision that the examples model small-batch deliberately", LOCAL),
    # ---- Feature phases (EXTENSIBILITY-ROADMAP 4b) ----
    106: ("open", PLANNED,  PRODUCT, "a count showing operator addition is not a bottleneck", MULT),
    107: ("open", PLANNED,  PRODUCT, "none; capability work with a clear exit test", SHARED),
    108: ("open", PLANNED,  PRODUCT, "IN SCOPE by PHASE2-MANIFESTO P3 ('tiny GPT, BERT, Llama "
                                     "inference'). Blocked on #118, not on scope", SHARED),
    109: ("open", PLANNED,  PRODUCT, "attribution showing attention is not dominant at target sizes", LOCAL),
    110: ("open", DEFERRED, PRODUCT, "ALREADY ANSWERED: PHASE2-MANIFESTO defers quantisation to "
                                     "'Later'. Not a core requirement", LOCAL),
    111: ("open", PLANNED,  PRODUCT, "safetensors plus a model-definition path making ONNX redundant", LOCAL),
    112: ("open", DEFERRED, PRODUCT, "ALREADY ANSWERED: manifesto defers multi-GPU to 'Later'. Still "
                                     "invalidates the observability single-device assumption", LOCAL),
    # ---- Findings from this audit ----
    113: ("open", DEBT,     VERIFY,  "a build change making the two extension paths one binary", SHARED),
    114: ("open", DESIGN,   KNOW,    "PREMISE DISPROVEN: the manifesto defines scope. Residual is "
                                     "O-A..O-F in PROJECT-SCOPE-ANALYSIS.md", SHARED),
    118: ("open", DEFECT,   PRODUCT, "a maintainer decision narrowing the manifesto P1 list -- "
                                     "which must be written INTO the manifesto, it being the authority", FOUND),
    # ---- First release (R-series) ----
    119: ("open", PLANNED,  VERIFY,  "none -- this is the release's definition of tested", SHARED),
    120: ("open", PLANNED,  VERIFY,  "none -- PyTorch is the stated oracle", SHARED),
    121: ("open", PLANNED,  VERIFY,  "none -- coverage without mutation overstates the suite", FOUND),
    122: ("open", PLANNED,  VERIFY,  "none -- without it the performance claim is unfalsifiable", FOUND),
    123: ("open", PLANNED,  KNOW,    "none -- a public claim must be generated from measurement", SHARED),
    124: ("open", DESIGN,   KNOW,    "blocked on #114; the definition is Jeel's to make", FOUND),
    # ---- Observability architecture increments ----
    115: ("done", PLANNED,  OBSERVE, "recorder + Python query surface", None),
    116: ("done", PLANNED,  VERIFY,  "shipped 1 of 2 proposed checks; the dispatch-structure signal "
                                     "does not exist until #99", None),
    # ---- Latent CI failures, both from f595d73, both found 2026-08-02 ----
    125: ("done", DEFECT,   VERIFY,  "CLOSED: the format gate had been red for 15 commits. Now a "
                                     "script, so check_gate_coverage sees it and verify_gates has "
                                     "a control for it", None),
    126: ("done", DEFECT,   PRODUCT, "CLOSED: the observability bindings were compiled out with the "
                                     "Vulkan backend, so `import vkml` died on the three CPU-only "
                                     "CI jobs", None),
    117: ("open", PLANNED,  OBSERVE, "MEASURED 58.8ns / 131.7ns per publish; end-to-end could not "
                                     "resolve it (4 orders under noise). Regression gate remains", SHARED),
}

LIFECYCLES = [DEFECT, DEBT, DESIGN, DEFERRED, SPEC, PLANNED]
PURPOSES = [CORRECT, VERIFY, OBSERVE, KNOW, PRODUCT]


def report(matrix: bool) -> int:
    open_ = {k: v for k, v in T.items() if v[0] == "open"}
    done = {k: v for k, v in T.items() if v[0] == "done"}
    print(f"\n  TRACKER — {len(T)} tasks, {len(open_)} open, {len(done)} closed\n")

    for dim, idx, values in (("LIFECYCLE", 1, LIFECYCLES), ("PURPOSE", 2, PURPOSES)):
        print(f"  {dim}")
        print(f"  {'':26} {'open':>5} {'closed':>7} {'total':>6}")
        for v in values:
            o = sum(1 for t in open_.values() if t[idx] == v)
            c = sum(1 for t in done.values() if t[idx] == v)
            bar = "#" * round(28 * (o + c) / len(T))
            print(f"  {v:26} {o:>5} {c:>7} {o + c:>6}  {bar}")
        print()

    if matrix:
        print("  LIFECYCLE x PURPOSE  (open/closed)")
        w = 13
        print("  " + " " * 18 + "".join(f"{p.split()[0][:11]:>{w}}" for p in PURPOSES))
        for lc in LIFECYCLES:
            row = f"  {lc:<18}"
            for p in PURPOSES:
                o = sum(1 for t in open_.values() if t[1] == lc and t[2] == p)
                c = sum(1 for t in done.values() if t[1] == lc and t[2] == p)
                row += f"{('-' if not (o + c) else f'{o}/{c}'):>{w}}"
            print(row)
        print()

    print("  LEVERAGE  (open tasks only — on closed ones it is hindsight)")
    for lv in (MULT, FOUND, SHARED, LOCAL):
        ids = sorted(k for k, v in open_.items() if v[4] == lv)
        print(f"  {lv:<16} {len(ids):>2}   {', '.join(f'#{i}' for i in ids)}")
    print()

    pairs = collections.Counter((t[1], t[2]) for t in T.values())
    print("  DOMINANT COMBINATIONS")
    for (lc, p), n in pairs.most_common(5):
        print(f"    {n:>3}  {lc} + {p}")
    empty = [(lc, p) for lc in LIFECYCLES for p in PURPOSES if (lc, p) not in pairs]
    print(f"\n  EMPTY COMBINATIONS ({len(empty)} of {len(LIFECYCLES) * len(PURPOSES)})")
    for lc, p in empty:
        print(f"    {lc} + {p}")

    print("\n  OPEN TASKS — what would let us delete this without implementing it")
    for k, (_, lc, p, note, _lv) in sorted(open_.items()):
        print(f"    #{k:<3} {lc:<17} {p:<23} {open_[k][4]:<15} {note}")

    blocked = [k for k, v in open_.items() if "needs O3" in v[3] or "blocked by" in v[3]]
    print(f"\n  BLOCKED ONLY BY ANOTHER CATEGORY: {', '.join(f'#{k}' for k in sorted(blocked))}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", action="store_true")
    return report(ap.parse_args().matrix)


if __name__ == "__main__":
    sys.exit(main())
