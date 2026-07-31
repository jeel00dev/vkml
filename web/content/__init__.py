"""Authored documentation content, keyed by the names the module exports.

WHY THIS IS SEPARATE FROM THE CODE. Signatures are generated from the installed
module, so they cannot go stale. Prose cannot be generated -- 78 of the 99 public
functions have a docstring that is only their signature repeated -- so it is
written here instead, and merged at build time.

The shape of each entry follows PyTorch's reference pages, because that is the
layout readers of a tensor library already know how to scan:

    summary    one declarative sentence, the way torch.matmul opens with
               "Matrix product of two tensors."
    detail     behaviour and rules. Cases go in a list, not a paragraph.
    params     (name, type, description) -- rendered as a definition list
    returns    what comes back
    note       clarification that is not a hazard
    warning    a hazard, or a divergence from torch worth knowing before use
    example    a REPL transcript with >>> prompts and real printed output
    see        related entries, linked

Everything except `summary` is optional. An entry with no `summary` renders with
a visible "not written yet" marker rather than being quietly skipped.
"""
from __future__ import annotations

from .elementwise import E as _ELEMENTWISE
from .reduction import R as _REDUCTION
from .guide import PAGES
from .prose import PROSE

# Group files are merged in, so prose can be split by subject rather than
# accumulating in one file that nobody wants to open.
PROSE.update(_ELEMENTWISE)
PROSE.update(_REDUCTION)

__all__ = ["PAGES", "PROSE"]
