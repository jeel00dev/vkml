"""Reading what the CI workflow actually RUNS, as opposed to what it says.

One function, and it exists because of a specific mistake. `check_min_spec.py`
originally asked whether a step contained the string `VKML_MIN_SPEC=1`. Its
negative control deleted the variable from the command and the gate stayed
green, because the step's own explanatory comment still contained the string. It
was reading the prose about the thing instead of the thing.

Anything asking "does CI do X?" needs the shell lines and only the shell lines,
so that question is answered in one place. Two copies of this parser would drift,
and the drift would be invisible: both would keep returning something.

Deliberately not a YAML parser. PyYAML is not a dependency of the gates -- they
run in the cheapest job, before anything is installed -- and the structure being
read is one level deep.
"""
from __future__ import annotations

import re
from pathlib import Path

_RUN = re.compile(r"\s*run:\s*[|>]?\s*$")
_RUN_INLINE = re.compile(r"\s*run:\s*(\S.*)$")


def run_blocks(workflow: Path) -> list[str]:
    """The shell text of every `run:` step, comments stripped.

    Returns one string per step, so a caller can require that two things appear
    in the SAME command rather than merely somewhere in the file.
    """
    blocks: list[str] = []
    for step in workflow.read_text().split("- name:"):
        shell: list[str] = []
        indent: int | None = None
        for line in step.splitlines():
            if indent is not None:
                bare = line.strip()
                if bare and len(line) - len(line.lstrip()) <= indent:
                    indent = None                      # dedented out of the block
                elif not bare.startswith("#"):
                    shell.append(line)
            if indent is None:
                if _RUN.match(line):
                    indent = len(line) - len(line.lstrip())
                elif (m := _RUN_INLINE.match(line)) and not m.group(1).startswith("#"):
                    shell.append(m.group(1))           # `run: python scripts/x.py`
        blocks.append("\n".join(shell))
    return blocks


def runs_command(workflow: Path, *fragments: str) -> bool:
    """True when one step's shell contains every fragment."""
    return any(all(f in b for f in fragments) for b in run_blocks(workflow))
