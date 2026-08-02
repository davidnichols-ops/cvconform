"""Automatic Failure Reduction — like compiler bug minimization, but for inputs.

When a target diverges on an input, this engine minimizes that input while the
divergence persists, reducing a large failing image to a small reproducing
patch. Two strategies:

  - spatial  : halve resolution / crop regions, keep the fragment that still fails
  - value    : reduce bit depth / zero-out channels or sub-blocks

Deterministic: seed + step budget recorded so repro is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Tuple

import numpy as np


@dataclass
class ReductionRecord:
    original_shape: tuple
    reduced_shape: tuple
    reduction_ratio: float
    steps: int
    strategy: str
    preserved: bool = True
    reduced_tensor: np.ndarray = None  # type: ignore[assignment]


def reduce_input(
    tensor: np.ndarray,
    divergence_fn: Callable[[np.ndarray], bool],
    steps_budget: int = 40,
    strategy: str = "spatial",
    seed: int = 0,
) -> ReductionRecord:
    """Minimize ``tensor`` while ``divergence_fn`` still returns True (holds).

    Returns a record including the reduced tensor and how much smaller it got.
    """
    if not divergence_fn(tensor):
        return ReductionRecord(tensor.shape, tensor.shape, 1.0, 0, strategy, False,
                               tensor.copy())

    current = tensor
    steps = 0
    if strategy == "spatial":
        current, steps = _reduce_spatial(current, divergence_fn, steps_budget)
    elif strategy == "value":
        current, steps = _reduce_value(current, divergence_fn, steps_budget)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    orig = float(np.prod(tensor.shape))
    red = float(np.prod(current.shape))
    ratio = red / orig if orig else 1.0
    return ReductionRecord(tensor.shape, current.shape, ratio, steps, strategy,
                           True, current.copy())


def _reduce_spatial(tensor, divergence_fn, budget):
    current = tensor
    steps = 0
    while steps < budget:
        h, w = current.shape[-2], current.shape[-1]
        if h <= 8 or w <= 8:
            break
        hh, ww = max(8, h // 2), max(8, w // 2)
        reduced = _imresize(current, (hh, ww))
        steps += 1
        if divergence_fn(reduced):
            current = reduced
            continue
        # try cropping each quadrant at current resolution
        found_crop = False
        for (y0, x0) in [(0, 0), (0, w // 2), (h // 2, 0), (h // 2, w // 2)]:
            crop = current[..., y0:y0 + h // 2, x0:x0 + w // 2]
            if crop.shape[-2] < 8 or crop.shape[-1] < 8:
                continue
            steps += 1
            if divergence_fn(np.ascontiguousarray(crop)):
                current = crop
                found_crop = True
                break
        if not found_crop:
            break
    return current, steps


def _reduce_value(tensor, divergence_fn, budget):
    current = tensor
    steps = 0
    while steps < budget:
        c = current.shape[1]
        if c <= 1:
            break
        keep = max(1, c // 2)
        variant = current.copy()
        variant[:, keep:] = 0.0
        steps += 1
        if divergence_fn(variant):
            current = variant
        else:
            break
    return current, steps


def _imresize(t, shape):
    """Nearest-neighbor resize (deterministic, dtype-preserving, index-safe)."""
    h, w = shape
    oh, ow = t.shape[-2], t.shape[-1]
    ys = np.clip((np.arange(h) * oh / h).astype(int), 0, oh - 1)
    xs = np.clip((np.arange(w) * ow / w).astype(int), 0, ow - 1)
    out = np.take(t, ys, axis=-2)
    out = np.take(out, xs, axis=-1)
    return np.ascontiguousarray(out)
