"""Comparison engine — turn differential outputs into divergence evidence.

This is where "is this still the same model?" becomes numbers. We compare a
target runtime's normalized outputs against a reference's, per output, per
kind, and produce :class:`ComparisonResult` / :class:`Divergence` objects with
observable magnitudes and attribution (which node is suspected).

Every claim here is a *measurement*, not an opinion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from cvconform.schema import Output


@dataclass
class TolerancePolicy:
    """Numerical thresholds that define 'conformant'."""

    max_abs_err: float = 1e-3      # tensor: max abs elementwise error
    rel_err: float = 1e-3          # tensor: relative error on normed magnitudes
    mean_abs_err: float = 1e-4     # tensor: mean abs error
    iou_threshold: float = 0.5     # boxes: matching boxes must reach this IoU
    class_agreement: float = 1.0   # classes: fraction that must agree (1.0 = all)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class Divergence:
    """A single, attributed divergence between reference and target."""

    output: str
    kind: str
    metric: str
    observed: float
    threshold: float
    magnitude: float            # observed - threshold (or observed for non-threshold)
    first_divergence: str = ""  # suspected node (populated by analyzer)
    mechanism: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output": self.output,
            "kind": self.kind,
            "metric": self.metric,
            "observed": round(self.observed, 6),
            "threshold": round(self.threshold, 6),
            "magnitude": round(self.magnitude, 6),
            "first_divergence": self.first_divergence,
            "mechanism": self.mechanism,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class ComparisonResult:
    """Result of comparing reference vs one target on one batch of inputs."""

    target: str
    reference: str
    per_output: Dict[str, Dict[str, float]] = field(default_factory=dict)  # output -> {metric: val}
    divergences: List[Divergence] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)   # output -> 0..100
    overall_score: float = 100.0
    is_conformant: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "reference": self.reference,
            "per_output": self.per_output,
            "divergences": [d.to_dict() for d in self.divergences],
            "scores": {k: round(v, 2) for k, v in self.scores.items()},
            "overall_score": round(self.overall_score, 2),
            "is_conformant": bool(self.is_conformant),
        }


def _tensor_error(ref: np.ndarray, tgt: np.ndarray) -> Dict[str, float]:
    """Elementwise error metrics between two same-shaped tensor arrays."""
    if ref.shape != tgt.shape:
        return {"shape_mismatch": 1.0, "max_abs_err": float("inf"), "mean_abs_err": float("inf"),
                "rmse": float("inf"), "nan_tgt": float("nan")}
    ref = ref.astype(np.float64)
    tgt = tgt.astype(np.float64)
    diff = np.abs(ref - tgt)
    denom = np.maximum(np.abs(ref), 1e-8)
    rel = diff / denom
    return {
        "max_abs_err": float(diff.max()) if diff.size else 0.0,
        "mean_abs_err": float(diff.mean()) if diff.size else 0.0,
        "rmse": float(np.sqrt((diff ** 2).mean())) if diff.size else 0.0,
        "max_rel_err": float(rel.max()) if rel.size else 0.0,
        "nan_tgt": float(np.isnan(tgt).mean()) if tgt.size else 0.0,
        "inf_tgt": float(np.isinf(tgt).mean()) if tgt.size else 0.0,
    }


def compare_tensors(ref: Output, tgt: Output, policy: TolerancePolicy) -> Dict[str, float]:
    if ref.tensor is None or tgt.tensor is None:
        return {"error": 1.0}
    return _tensor_error(ref.tensor, tgt.tensor)


def compare_boxes(ref: Output, tgt: Output, policy: TolerancePolicy) -> Dict[str, float]:
    rb, tb = ref.boxes, tgt.boxes
    if not rb and not tb:
        return {"box_count_error": 0.0, "mean_iou": 1.0, "match_rate": 1.0}
    # Match ground-truth (ref) boxes to target boxes by class + max IoU.
    matched = 0.0
    ious = []
    unmatched_ref = 0
    for rb_box in rb:
        best = -1.0
        for tb_box in tb:
            if tb_box.class_id != rb_box.class_id:
                continue
            iou = _iou(rb_box.coords, tb_box.coords)
            if iou > best:
                best = iou
        if best >= policy.iou_threshold:
            matched += 1.0
            ious.append(best)
        else:
            unmatched_ref += 1
    total = max(len(rb), 1)
    return {
        "box_count_ref": len(rb),
        "box_count_tgt": len(tb),
        "match_rate": matched / total,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "unmatched_ref": unmatched_ref,
    }


def _iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(x) for x in a)
    bx1, by1, bx2, by2 = (float(x) for x in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def compare_classes(ref: Output, tgt: Output) -> Dict[str, float]:
    rv, tv = ref.values, tgt.values
    if rv is None or tv is None:
        return {"error": 1.0}
    rv = np.asarray(rv, dtype=np.float32).ravel()
    tv = np.asarray(tv, dtype=np.float32).ravel()
    n = min(len(rv), len(tv))
    if n == 0:
        return {"matches": 0.0, "agreement": 1.0}
    agreement = float(np.mean(rv[:n] == tv[:n]))
    return {"matches": int(np.sum(rv[:n] == tv[:n])), "agreement": agreement}


def compare_scores(ref: Output, tgt: Output) -> Dict[str, float]:
    rv, tv = ref.values, tgt.values
    if rv is None or tv is None:
        return {"error": 1.0}
    rv = np.asarray(rv, dtype=np.float64).ravel()
    tv = np.asarray(tv, dtype=np.float64).ravel()
    n = min(len(rv), len(tv))
    if n == 0:
        return {"mean_abs_err": 0.0, "max_abs_err": 0.0}
    return {
        "mean_abs_err": float(np.mean(np.abs(rv[:n] - tv[:n]))),
        "max_abs_err": float(np.max(np.abs(rv[:n] - tv[:n]))),
    }


def compare(ref: Output, tgt: Output, policy: TolerancePolicy) -> Dict[str, float]:
    """Compare two same-kind outputs and return metric dict, always float keys."""
    if ref.kind != tgt.kind:
        return {"kind_mismatch": 1.0, "mismatched_kinds": f"{ref.kind}->{tgt.kind}"}
    if ref.kind == "tensor":
        return compare_tensors(ref, tgt, policy)
    if ref.kind == "boxes":
        return compare_boxes(ref, tgt, policy)
    if ref.kind == "classes":
        return compare_classes(ref, tgt)
    if ref.kind == "scores":
        return compare_scores(ref, tgt)
    return {"unsupported_kind": 1.0}


def score_from_metrics(metrics: Dict[str, float], kind: str, policy: TolerancePolicy) -> float:
    """Map per-kind metric dict to a 0..100 conformance score."""
    if "kind_mismatch" in metrics or "error" in metrics or "unsupported_kind" in metrics:
        return 0.0
    if kind == "tensor":
        max_abs = metrics.get("max_abs_err", 0.0)
        mean_abs = metrics.get("mean_abs_err", 0.0)
        # Score decays with error relative to threshold; saturated with soft clamp.
        score = 100.0
        if max_abs > policy.max_abs_err:
            penalty = min(100.0, (max_abs - policy.max_abs_err) / max(policy.max_abs_err, 1e-9) * 25.0)
            score -= penalty
        if mean_abs > policy.mean_abs_err:
            penalty = min(100.0, (mean_abs - policy.mean_abs_err) / max(policy.mean_abs_err, 1e-9) * 25.0)
            score -= penalty
        nan_frac = metrics.get("nan_tgt", 0.0)
        inf_frac = metrics.get("inf_tgt", 0.0)
        score -= (nan_frac + inf_frac) * 100.0
        return float(max(0.0, min(100.0, score)))
    if kind == "boxes":
        match = metrics.get("match_rate", 0.0)
        return float(match * 100.0)
    if kind == "classes":
        return float(metrics.get("agreement", 1.0) * 100.0)
    if kind == "scores":
        mean_abs = metrics.get("mean_abs_err", 0.0)
        # relative to a score range; treat errors >0.1 as divergence
        score = 100.0 - min(100.0, mean_abs / 0.1 * 100.0)
        return float(max(0.0, score))
    return 0.0


def build_divergences(target: str, reference: str, per_output_metrics: Dict[str, Dict[str, float]],
                      policies: Dict[str, TolerancePolicy]) -> List[Divergence]:
    """Turn per-output metrics into Divergence objects where thresholds are exceeded."""
    divs: List[Divergence] = []
    for out, metrics in per_output_metrics.items():
        if "kind_mismatch" in metrics:
            divs.append(Divergence(out, metrics.get("mismatched_kinds", "kind"), "kind_mismatch",
                                   observed=1.0, threshold=0.0, magnitude=1.0,
                                   mechanism="Output kind changed across runtime"))
            continue
        if "error" in metrics or "unsupported_kind" in metrics:
            divs.append(Divergence(out, "unknown", "error", observed=1.0, threshold=0.0,
                                   magnitude=1.0, mechanism="No comparable output"))
            continue
        if "max_abs_err" in metrics:
            mae = metrics["max_abs_err"]
            th = policies["tensor"].max_abs_err
            if mae > th:
                divs.append(Divergence(out, "tensor", "max_abs_err", observed=mae, threshold=th,
                                       magnitude=mae - th, mechanism="Tensor value divergence"))
        if "mean_abs_err" in metrics and "max_abs_err" not in metrics:
            mae = metrics["mean_abs_err"]
            th = policies["scores"].mean_abs_err
            if mae > th:
                divs.append(Divergence(out, "scores", "mean_abs_err", observed=mae, threshold=th,
                                       magnitude=mae - th, mechanism="Score divergence"))
        if "match_rate" in metrics:
            mr = metrics["match_rate"]
            th = 1.0
            if mr < th:
                divs.append(Divergence(out, "boxes", "match_rate", observed=1.0 - mr,
                                       threshold=0.0, magnitude=(1.0 - mr),
                                       mechanism="Box matching divergence (different detections)"))
        if "agreement" in metrics and "max_abs_err" not in metrics:
            ag = metrics["agreement"]
            if ag < 1.0:
                divs.append(Divergence(out, "classes", "class_agreement", observed=1.0 - ag,
                                       threshold=0.0, magnitude=(1.0 - ag),
                                       mechanism="Class prediction divergence"))
    return divs


def compare_target(reference_outputs: Dict[str, Output],
                   target_outputs: Dict[str, Output],
                   target_name: str, reference_name: str,
                   policy: TolerancePolicy) -> ComparisonResult:
    """Compare reference vs target Siamese outputs into a ComparisonResult."""
    per_output_metrics: Dict[str, Dict[str, float]] = {}
    scores: Dict[str, float] = {}
    output_names = sorted(set(reference_outputs) | set(target_outputs))
    for name in output_names:
        if name not in reference_outputs or name not in target_outputs:
            # Missing output on one side is a strong divergence.
            missing = "reference" if name not in reference_outputs else "target"
            per_output_metrics[name] = {"missing_output": 1.0, "which": missing}
            scores[name] = 0.0
            continue
        metrics = compare(reference_outputs[name], target_outputs[name], policy)
        per_output_metrics[name] = metrics
        scores[name] = score_from_metrics(metrics, reference_outputs[name].kind, policy)

    # Build divergences
    policies = {"tensor": policy, "scores": policy, "boxes": policy, "classes": policy}
    divs = build_divergences(target_name, reference_name, per_output_metrics, policies)
    # Handle missing-output divergences
    for name in output_names:
        if "missing_output" in per_output_metrics.get(name, {}):
            divs.append(Divergence(name, "output", "missing_output", observed=1.0, threshold=0.0,
                                   magnitude=1.0,
                                   mechanism=f"Output {name!r} missing on {'reference' if per_output_metrics[name]['which']=='reference' else 'target'}"))

    overall = float(np.mean(list(scores.values()))) if scores else 100.0
    is_conformant = overall >= 95.0 and not divs
    return ComparisonResult(
        target=target_name, reference=reference_name,
        per_output=per_output_metrics, divergences=divs,
        scores=scores, overall_score=overall, is_conformant=is_conformant,
    )
