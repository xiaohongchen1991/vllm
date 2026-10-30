# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-case kernel inputs: one argument tuple, or several to tune jointly.

A case's ``input_generator`` value is either the usual argument tuple or a
tuple of argument tuples. The latter tunes a single config across several
synthetic inputs of the same shapes (e.g. different top-k routings for a MoE
kernel), so each candidate is scored on helion's joint objective instead of a
single draw.
"""

from __future__ import annotations

from typing import Any

# One argument tuple, or several for multi-input autotuning.
CaseInputs = tuple[Any, ...] | tuple[tuple[Any, ...], ...]


def normalize_arg_sets(inputs: CaseInputs) -> tuple[tuple[Any, ...], ...]:
    """Split a case's inputs into one argument tuple per input.

    Registered kernels take tensor/scalar arguments, so a nested tuple is
    unambiguous: ``(hidden, 8, None)`` is one input and
    ``((hidden, 8), (hidden, 16))`` is two.
    """
    if inputs and isinstance(inputs[0], tuple):
        return tuple(inputs)  # type: ignore[arg-type]
    return (tuple(inputs),)
