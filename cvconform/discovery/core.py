"""Failure Discovery Engine — find where reality breaks, not just 'test 100 images'.

Where a fixed dataset asks "do these inputs pass?", discovery asks "where does
this model stop being itself?" It generates a broad, seedable space of inputs
across several perturbation families and hunts for the inputs that maximize
divergence between a reference and a target runtime.

Perturbation families (each a deterministic generator parameterized by seed):
  - noise        : uniform + gaussian noise intensity sweep
  - edges        : high-contrast / structured grids and single-pixel probes
  - illumination : brightness / contrast / gamma sweeps
  - blur         : low-pass blur over sigma
  - compression  : coarse block quantization (JPEG-like artifacts)
  - extremes     : NaN, Inf, saturated, solid-color, zeroed inputs

Every discovered divergence seeds a candidate regression-memory entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np


@dataclass
class DiscoveryHit:
    """A discovered input that pushes a target toward divergence."""

    family: str
    params: Dict[str, Any]
    tensor: np.ndarray
    divergence_magnitude: float
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"family": self.family, "params": self.params,
                "divergence": round(self.divergence_magnitude, 6),
                "score": round(self.score, 2)}


class InputGenerator:
    """Deterministic generators for discovery inputs."""

    def __init__(self, shape: Tuple[int, ...], seed: int = 0):
        self.shape = shape
        self.rng = np.random.default_rng(seed)

    def _img(self, base: np.ndarray) -> np.ndarray:
        return np.clip(base, 0, 1).astype(np.float32)

    # -- families ----------------------------------------------------------
    def noise(self, gaussian: bool = False) -> np.ndarray:
        if gaussian:
            return np.clip(np.random.default_rng(self.rng.integers(1e9)).normal(0.5, 0.3, self.shape), 0, 1).astype(np.float32)
        return self._img(self.rng.uniform(0, 1, self.shape))

    def edges(self, period: int = 16) -> np.ndarray:
        img = np.zeros(self.shape, dtype=np.float32)
        for y in range(0, self.shape[-2], period):
            img[..., y, :] = 1.0
        for x in range(0, self.shape[-1], period):
            img[..., :, x] = 1.0
        return img

    def single_pixel(self, value: float = 1.0) -> np.ndarray:
        img = np.zeros(self.shape, dtype=np.float32)
        h, w = self.shape[-2:]
        img[0, 0, h // 2, w // 2] = value
        return img

    def illumination(self, brightness: float = 0.5, gamma: float = 1.0) -> np.ndarray:
        base = self.rng.uniform(0.1, 0.9, self.shape)
        img = base * brightness
        if gamma != 1.0:
            img = np.power(np.abs(img), 1.0 / gamma)
        return self._img(img)

    def blur(self, sigma: float = 1.0) -> np.ndarray:
        base = self.rng.uniform(0, 1, self.shape)
        k = max(1, int(sigma))
        img = base.copy()
        for _ in range(k):
            img = 0.25 * (np.roll(img, 1, axis=-1) + np.roll(img, -1, axis=-1) +
                          np.roll(img, 1, axis=-2) + np.roll(img, -1, axis=-2))
        return self._img(img)

    def compression(self, block: int = 8) -> np.ndarray:
        img = self.rng.uniform(0, 1, self.shape)
        h, w = self.shape[-2:]
        bh, bw = max(1, block), max(1, block)
        q = np.zeros(img.shape, dtype=np.float32)
        for y in range(0, h - bh + 1, bh):
            for x in range(0, w - bw + 1, bw):
                q[..., y:y + bh, x:x + bw] = img[..., y:y + bh, x:x + bw].mean()
        return self._img(q)

    def extremes(self, kind: str = "nan") -> np.ndarray:
        if kind == "nan":
            img = self.rng.uniform(0, 1, self.shape).astype(np.float32)
            img[0, 0, 0, 0] = float("nan")
            return img
        if kind == "inf":
            img = np.zeros(self.shape, dtype=np.float32)
            img[0, 0, 0, 0] = float("inf")
            return img
        if kind == "solid":
            return np.full(self.shape, 1.0, dtype=np.float32)
        if kind == "black":
            return np.zeros(self.shape, dtype=np.float32)
        if kind == "saturate":
            return np.full(self.shape, 0.99, dtype=np.float32)
        return self.rng.uniform(-1e4, 1e4, self.shape).astype(np.float32)


FAMILIES = ["noise", "edges", "single_pixel", "illumination", "blur",
            "compression", "extremes"]


class Expedition:
    """Sweep a configuration space and rank inputs by induced divergence."""

    def __init__(self, shape: Tuple[int, ...], seed: int = 0):
        self.gen = InputGenerator(shape, seed)
        self.hits: List[DiscoveryHit] = []

    def candidate_pool(self) -> List[Tuple[str, Dict[str, Any], np.ndarray]]:
        """Return a broad set of (family, params, tensor) candidates."""
        pool = []
        g = self.gen
        for b in (0.1, 0.5, 0.9):
            pool.append(("noise", {"brightness": b}, g.noise() * b))
        pool.append(("noise_gauss", {}, g.noise(gaussian=True)))
        for p in (8, 16, 32):
            pool.append(("edges", {"period": p}, g.edges(p)))
        pool.append(("single_pixel", {"value": 1.0}, g.single_pixel(1.0)))
        for b in (0.05, 0.5, 2.0):
            pool.append(("illumination", {"brightness": b}, g.illumination(brightness=b)))
        for ga in (0.5, 1.5):
            pool.append(("illumination", {"gamma": ga}, g.illumination(gamma=ga)))
        for s in (1, 3, 5):
            pool.append(("blur", {"sigma": s}, g.blur(s)))
        for block in (4, 8, 16):
            pool.append(("compression", {"block": block}, g.compression(block)))
        for k in ("nan", "inf", "solid", "black", "saturate"):
            pool.append(("extremes", {"kind": k}, g.extremes(k)))
        return pool

    def run(self, reference_fn: Callable[[np.ndarray], float],
            divergence_fn: Callable[[np.ndarray], float]) -> List[DiscoveryHit]:
        """Rank candidates by induced divergence magnitude."""
        hits = []
        for family, params, tensor in self.candidate_pool():
            try:
                reference_fn(tensor)
                div = divergence_fn(tensor)
            except Exception:
                continue
            score = div
            hits.append(DiscoveryHit(family, params, tensor, div, score))
        hits.sort(key=lambda h: h.divergence_magnitude, reverse=True)
        self.hits = hits
        return hits
