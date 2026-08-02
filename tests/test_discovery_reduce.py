"""Tests for Failure Discovery + Automatic Failure Reduction."""

from __future__ import annotations

import numpy as np

from cvconform.discovery import InputGenerator, Expedition, FAMILIES
from cvconform.reduce import reduce_input


def test_input_generator_families():
    g = InputGenerator((1, 3, 16, 16), seed=7)
    assert g.noise().shape == (1, 3, 16, 16)
    assert g.edges().max() <= 1.0
    assert g.illumination().min() >= 0.0
    assert g.compression().max() <= 1.0
    assert g.extremes("nan").dtype == np.float32
    assert g.extremes("black").max() == 0.0


def test_input_generator_is_seed_deterministic():
    g1 = InputGenerator((1, 3, 16, 16), seed=5)
    g2 = InputGenerator((1, 3, 16, 16), seed=5)
    assert np.array_equal(g1.noise(), g2.noise())


def test_expedition_ranks_by_divergence():
    shape = (1, 3, 32, 32)
    exp = Expedition(shape, seed=0)
    pool = exp.candidate_pool()
    assert len(pool) > 10, "expected a broad candidate pool"

    def ref_fn(x):
        return float(np.nanmean(x) if not np.isnan(x).all() else 0.0)

    def div_fn(x):
        # artificial diverging signal: integer-like patterns diverge more
        return float(np.mean(np.abs(x - np.round(x))))

    hits = exp.run(ref_fn, div_fn)
    assert len(hits) > 0
    # sorted descending by divergence
    divs = [h.divergence_magnitude for h in hits]
    assert divs == sorted(divs, reverse=True)


def test_reduce_spatial_minimizes():
    rng = np.random.default_rng(0)
    big = rng.uniform(0, 1, (1, 3, 128, 128)).astype(np.float32)

    def holds(x):
        # divergence holds on any high-value structured pattern present
        return np.any(x > 0.9)

    rec = reduce_input(big, holds, steps_budget=30, strategy="spatial", seed=0)
    assert rec.reduction_ratio < 1.0
    assert rec.original_shape == (1, 3, 128, 128)
    # reduced tensor must still be present and still hold divergence
    assert rec.reduced_tensor is not None
    assert holds(rec.reduced_tensor)


def test_reduce_preserves_divergence():
    rng = np.random.default_rng(1)
    big = rng.uniform(0, 0.5, (1, 1, 64, 64)).astype(np.float32)
    big[0, 0, 2:4, 2:4] = 1.0  # tiny hot patch that must survive reduction

    def holds(x):
        return x.max() == 1.0

    rec = reduce_input(big, holds, steps_budget=20, strategy="spatial", seed=1)
    # the 1.0 patch must survive spatial reduction -> still holds
    assert holds(rec.reduced_tensor)
    assert rec.reduction_ratio < 1.0
