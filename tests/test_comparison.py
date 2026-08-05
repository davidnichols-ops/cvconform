"""Unit tests for the comparison engine (scientific core)."""

from __future__ import annotations

import numpy as np

from cvconform.engines.comparison import (
    TolerancePolicy,
    compare_tensors,
    compare_boxes,
    compare_classes,
    compare_scores,
    score_from_metrics,
    compare_target,
)
from cvconform.schema import Output, Box


def _tensor(data, name="t"):
    return Output.from_tensor(np.asarray(data, dtype=np.float32), name=name)


def test_identical_tensors_conform():
    ref = _tensor(np.random.randn(4, 4), "t")
    tgt = _tensor(ref.tensor.copy(), "t")
    pol = TolerancePolicy()
    metrics = compare_tensors(ref, tgt, pol)
    assert metrics["max_abs_err"] == 0.0
    assert score_from_metrics(metrics, "tensor", pol) == 100.0


def test_drifted_tensor_scores_below_100():
    ref = _tensor(np.zeros((8, 8)), "t")
    tgt = _tensor(np.ones((8, 8)) * 0.5, "t")
    pol = TolerancePolicy()
    metrics = compare_tensors(ref, tgt, pol)
    score = score_from_metrics(metrics, "tensor", pol)
    assert score < 100.0


def test_shape_mismatch_scores_zero():
    ref = _tensor(np.zeros((4, 4)), "t")
    tgt = _tensor(np.zeros((4, 8)), "t")
    pol = TolerancePolicy()
    metrics = compare_tensors(ref, tgt, pol)
    assert score_from_metrics(metrics, "tensor", pol) == 0.0


def test_nan_lowers_score():
    ref = _tensor(np.zeros((4, 4)), "t")
    arr = np.zeros((4, 4), dtype=np.float32)
    arr[0, 0] = np.nan
    tgt = _tensor(arr, "t")
    pol = TolerancePolicy()
    metrics = compare_tensors(ref, tgt, pol)
    assert metrics["nan_tgt"] > 0
    assert score_from_metrics(metrics, "tensor", pol) < 100.0


def test_nan_target_emits_divergence():
    """NaN values in a target tensor must be reported as a divergence."""
    ref = _tensor(np.zeros((4, 4)), "t")
    arr = np.zeros((4, 4), dtype=np.float32)
    arr[0, 0] = np.nan
    tgt = _tensor(arr, "t")
    res = compare_target({"t": ref}, {"t": tgt}, "onnx", "ref", TolerancePolicy())
    assert any(d.metric == "nan_inf" for d in res.divergences)
    assert not res.is_conformant


def test_box_matching():
    from cvconform.engines.comparison import _iou

    a = [0, 0, 10, 10]
    b = [2, 2, 12, 12]
    assert _iou(a, b) == pytest.approx(64 / 136)  # overlap area / union
    c = [50, 50, 60, 60]
    assert _iou(a, c) == 0.0  # disjoint


def test_classes_agreement():
    ref = Output.from_values([1, 2, 3], "classes", "cls")
    tgt = Output.from_values([1, 2, 2], "classes", "cls")
    m = compare_classes(ref, tgt)
    assert m["agreement"] == pytest.approx(2 / 3)


def test_scores_error():
    ref = Output.from_values([0.9, 0.1, 0.5], "scores", "sc")
    tgt = Output.from_values([0.8, 0.2, 0.5], "scores", "sc")
    m = compare_scores(ref, tgt)
    assert m["mean_abs_err"] > 0


def test_compare_target_missing_output():
    ref = {"a": _tensor(np.zeros((2, 2)))}
    tgt = {}
    res = compare_target(ref, tgt, "onnx", "ref", TolerancePolicy())
    assert res.overall_score < 95.0
    assert any(d.metric == "missing_output" for d in res.divergences)


def test_compare_target_conformant():
    ref = {"a": _tensor(np.random.randn(4, 4))}
    tgt = {"a": _tensor(np.random.randn(4, 4))}
    # random unrelated -> should diverge; but identical should conform
    ref2 = {"a": _tensor(np.zeros((4, 4)))}
    tgt2 = {"a": _tensor(np.zeros((4, 4)))}
    res = compare_target(ref2, tgt2, "onnx", "ref", TolerancePolicy())
    assert res.is_conformant


import pytest  # noqa: E402
