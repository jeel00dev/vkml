#!/usr/bin/env python3
"""Turn a coverage recording into an operator x property matrix, and name the gaps.

docs/PHASE2-MANIFESTO.md P2 asks that every operator be tested for "forward,
backward, edge cases, empty tensors, broadcasting, mixed precision, layouts,
large and random tensors". Whether that has happened is a question about the
suite, and until now nobody had asked it -- every milestone added tests for the
thing it built, which answers "is the new code tested" and never answers "what is
NOT tested".

This reads what the suite actually executed (VKML_COVERAGE, see
include/vkml/util/coverage.h) and reports it against the operator inventory read
out of the enum itself.

    VKML_COVERAGE=1 python -m pytest tests/python -q      # records
    python scripts/coverage_matrix.py <dump.tsv>          # reports

WHAT A GAP HERE MEANS, and does not. An empty cell says the suite never ran that
combination. It does NOT say the combination is broken, and for several
operators it is not even meaningful -- a triangular mask has no rank-1 case, a
matmul has no bool case. So gaps are reported at three severities, and only the
first two are claims:

    BLOCKING  never executed at all, or a backward rule that never fired.
              Nothing about this code has been observed to work.
    ORACLE    executed on one backend only. docs/ARCHITECTURE.md 7 makes the
              correctness argument a chain -- CPU against PyTorch for semantics,
              then Vulkan against CPU for kernel bugs. One backend breaks it.
    PATH      a code path inside the kernel that was never taken: strided
              inputs, or a size crossing more than one workgroup.

Everything else -- dtypes, ranks, empty tensors -- is printed as data rather than
judged, because applicability varies per operator and a report that cries wolf on
meaningless cells is a report nobody reads twice.

Exit code is always 0. This is a measurement, not a gate; making it one requires
a baseline of accepted gaps, which is worth doing once the blocking ones are
closed.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

RANKS = ["rank0", "rank1", "rank2", "rank3", "rank4"]
SIZES = ["empty", "scalar", "one_group", "many_groups"]


@dataclass
class OpCoverage:
    """Everything the recording saw for one operator."""

    backends: set[str] = field(default_factory=set)
    dtypes: set[str] = field(default_factory=set)
    ranks: set[str] = field(default_factory=set)
    sizes: set[str] = field(default_factory=set)
    n_src: set[int] = field(default_factory=set)
    broadcast_input: bool = False
    strided_input: bool = False
    strided_output: bool = False
    count: int = 0

    @property
    def is_view(self) -> bool:
        return self.backends == {"view"}

    @property
    def takes_tensors(self) -> bool:
        """False for creation ops (full, arange, rand) and leaves, which have no
        source to be strided."""
        return any(n > 0 for n in self.n_src)


def read_dump(path: Path) -> tuple[dict[str, OpCoverage], dict[str, int]]:
    """Parse the recording. Returns per-op dispatch coverage and backward-rule counts."""
    dispatches: dict[str, OpCoverage] = defaultdict(OpCoverage)
    backward: dict[str, int] = {}

    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        parts = raw.split("\t")

        if parts[0] == "backward":
            if len(parts) != 3:
                raise ValueError(f"{path}:{line_no}: malformed backward record: {raw!r}")
            backward[parts[1]] = int(parts[2])
            continue

        if parts[0] != "dispatch":
            raise ValueError(f"{path}:{line_no}: unknown record kind {parts[0]!r}")
        if len(parts) != 11:
            raise ValueError(f"{path}:{line_no}: expected 11 fields, got {len(parts)}: {raw!r}")

        _, op, backend, dtype, rank, size, src, bcast, strided_in, strided_out, count = parts
        cov = dispatches[op]
        cov.backends.add(backend_family(backend))
        cov.dtypes.add(dtype)
        cov.ranks.add(rank)
        cov.sizes.add(size)
        cov.n_src.add(int(src.removeprefix("src")))
        cov.broadcast_input |= bcast == "broadcast"
        cov.strided_input |= strided_in == "strided_in"
        cov.strided_output |= strided_out == "strided_out"
        cov.count += int(count)

    return dict(dispatches), backward


def declared_operators() -> tuple[list[str], dict[str, str]]:
    """The inventory and each operator's category, read out of the enum.

    Not parsed from op.h: a regex over a header drifts silently the first time
    someone adds an operator, and a coverage denominator that is quietly wrong
    is the one failure this whole exercise exists to avoid.

    The category matters because the three kinds are observable to different
    degrees. A `leaf` arrives already realised, so `topological_order` prunes it
    and the executor never sees one -- reporting leaves as uncovered would be
    crying wolf on the one thing that cannot be otherwise.
    """
    import vkml as V

    categories = dict(V._vkml_core._op_categories())
    return list(V._vkml_core._op_names()), categories


def declared_backward_rules(source: Path) -> set[str]:
    """The rules apply_backward implements, parsed from its switch.

    There is no runtime enumeration of these -- the switch IS the inventory -- so
    this is a parse. It is narrow (one function, one statement form) and drift
    is caught: any rule that fires but is not in this set is reported.
    """
    text = source.read_text()
    start = text.find("void apply_backward(")
    if start < 0:
        raise ValueError(f"{source}: apply_backward not found; the parse needs updating")
    end = text.find("not_differentiable(node)", start)
    if end < 0:
        raise ValueError(f"{source}: the default case moved; the parse needs updating")

    rules = set(re.findall(r"case OpKind::(\w+)", text[start:end]))
    if not rules:
        raise ValueError(f"{source}: parsed zero backward rules, which cannot be right")
    return rules


def normalised(name: str) -> str:
    """A key that matches an enum identifier against the name op_name() returns.

    Comparison by stripped-and-lowered text rather than by converting camel case
    to snake case. A conversion has to guess where the underscores go, and it
    guesses wrong on exactly the operators whose names contain digits --
    `Im2Col` becomes `im2_col`, which matches nothing, so a tested rule is
    reported as never fired. Normalising both sides cannot make that mistake:
    `Im2Col` and `im2col` both reduce to `im2col`.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


def backend_family(name: str) -> str:
    """`vulkan:0` -> `vulkan`.

    The backend reports a device-qualified name. Left as-is, a machine with two
    GPUs would make every operator look like it ran on only some backends.
    """
    return name.split(":", 1)[0]


def bar(present: set[str], order: list[str]) -> str:
    """Compact presence strip, e.g. '. . 2 3 .' for ranks 2 and 3 seen."""
    return " ".join(v[-1] if v in present else "." for v in order)


def report(dump: Path) -> None:
    dispatches, backward_fired = read_dump(dump)
    declared, categories = declared_operators()
    rule_keys = {normalised(r) for r in declared_backward_rules(ROOT / "src/autograd/autograd.cpp")}

    # Everything downstream compares by normalised key, so an enum identifier and
    # the name op_name() prints are the same thing regardless of spelling.
    rules = sorted(op for op in declared if normalised(op) in rule_keys)
    fired = {op for op in declared if normalised(op) in {normalised(k) for k in backward_fired}}

    observable = [op for op in declared if categories.get(op) != "leaf"]
    executed = set(dispatches)
    never = [op for op in observable if op not in executed]
    backends_seen = set().union(*(c.backends for c in dispatches.values())) - {"view"}

    print("# Operator coverage\n")
    print(f"Recording: `{dump}`\n")
    print(f"- operators declared: **{len(declared)}** "
          f"({len(observable)} the executor can observe; "
          f"{len(declared) - len(observable)} are leaves, whose values are supplied "
          "rather than computed)")
    print(f"- operators executed: **{len(executed)} of {len(observable)}**")
    print(f"- backends observed: {', '.join(sorted(backends_seen)) or 'none'}")
    print(f"- backward rules declared: **{len(rules)}**, fired: **{len(fired)}**")
    print(f"- total node evaluations: {sum(c.count for c in dispatches.values()):,}\n")

    # -- BLOCKING -----------------------------------------------------------
    print("## BLOCKING — never executed\n")
    if never:
        print("Nothing about these has been observed to work.\n")
        for op in never:
            print(f"- `{op}`" + ("  *(has a backward rule)*" if op in rules else ""))
    else:
        print("None. Every operator the executor can observe was executed at least once.")
    print()

    print("## BLOCKING — backward rules that never fired\n")
    unfired = [op for op in rules if op not in fired]
    if unfired:
        print("A rule that never runs is a rule nobody has seen produce a number.\n")
        for op in unfired:
            print(f"- `{op}`")
    else:
        print("None. Every declared backward rule was exercised.")
    print()

    stray = sorted(k for k in backward_fired if normalised(k) not in rule_keys)
    if stray:
        print("**Parse drift:** these rules fired but were not found in the switch — "
              "`declared_backward_rules` needs updating.\n")
        for op in stray:
            print(f"- `{op}`")
        print()

    # -- ORACLE -------------------------------------------------------------
    print("## ORACLE — executed on one backend only\n")
    if len(backends_seen) < 2:
        print(f"Not assessable: only `{', '.join(backends_seen) or 'no'}` backend(s) ran, so "
              "every operator would show a false gap. Re-run with both.\n")
    else:
        single = [(op, c) for op, c in sorted(dispatches.items())
                  if not c.is_view and len(c.backends & backends_seen) < len(backends_seen)]
        if single:
            print("The correctness chain (`ARCHITECTURE.md` §7) is CPU-against-PyTorch for "
                  "semantics, then Vulkan-against-CPU for kernel bugs. One backend breaks it.\n")
            print("| operator | ran on | missing |")
            print("|---|---|---|")
            for op, c in single:
                have = c.backends & backends_seen
                print(f"| `{op}` | {', '.join(sorted(have))} | "
                      f"**{', '.join(sorted(backends_seen - have))}** |")
        else:
            print("None. Every computed operator ran on every observed backend.")
    print()

    # -- PATH ---------------------------------------------------------------
    print("## PATH — kernel paths never taken\n")
    no_strided = [op for op, c in sorted(dispatches.items())
                  if not c.is_view and c.takes_tensors and not c.strided_input]
    no_large = [op for op, c in sorted(dispatches.items())
                if not c.is_view and "many_groups" not in c.sizes]

    print("**Never given a non-contiguous input.** Kernels index their sources through "
          "`operand_offset`; an operator that has only ever seen contiguous inputs has "
          "never run that arithmetic.\n")
    print(("- " + "\n- ".join(f"`{op}`" for op in no_strided)) if no_strided
          else "None.")
    print()
    print("**Never run at a size crossing more than one workgroup** (>256 elements, the "
          "default in `vk_pipeline.h`). Single-group runs hide cross-group indexing errors.\n")
    print(("- " + "\n- ".join(f"`{op}`" for op in no_large)) if no_large else "None.")
    print()

    # -- DATA ---------------------------------------------------------------
    print("## Matrix\n")
    print("Ranks `0 1 2 3 4`, sizes `empty scalar one_group many_groups` — a digit or letter "
          "means seen, `.` means not. Judge applicability per operator: several cells are "
          "meaningless rather than missing.\n")
    print("| operator | kind | backends | dtypes | ranks | sizes | bcast in | strided in | strided out | evals |")
    print("|---|---|---|---|---|---|---|---|---|---:|")
    for op in declared:
        kind = categories.get(op, "?")
        c = dispatches.get(op)
        if c is None:
            note = "not schedulable" if kind == "leaf" else "**never executed**"
            print(f"| `{op}` | {kind} | {note} | — | — | — | — | — | — | 0 |")
            continue
        sizes = " ".join(s[0] if s in c.sizes else "." for s in SIZES)
        print(f"| `{op}` | {kind} | {','.join(sorted(c.backends))} | {','.join(sorted(c.dtypes))} "
              f"| {bar(c.ranks, RANKS)} | {sizes} "
              f"| {'yes' if c.broadcast_input else '.'} "
              f"| {'yes' if c.strided_input else '.'} "
              f"| {'yes' if c.strided_output else '.'} | {c.count:,} |")
    print()

    print("## Backward rules\n")
    by_key = {normalised(k): v for k, v in backward_fired.items()}
    print("| rule | times fired |")
    print("|---|---:|")
    for op in rules:
        print(f"| `{op}` | {by_key.get(normalised(op), 0):,} |")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        print(f"usage: {sys.argv[0]} <dump.tsv>", file=sys.stderr)
        return 2
    report(Path(sys.argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
