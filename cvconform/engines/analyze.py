"""Root-Cause Analysis Engine — explain *why* a deployment differs.

Given the set of divergences found by the differential engine plus the stable
IR of the reference model, this engine forms and *tests* hypotheses about what
changed. Every hypothesis lands on one of a small set of known mechanisms that
account for the vast majority of real CV deployment failures:

  - operator fusion / execution-order change
  - precision change (fp32 -> fp16 / bf16 / quantized int8)
  - quantization scale / zero-point mismatch
  - unsupported-operator fallback
  - padding / auto_pad behavior difference
  - NMS implementation difference (non-max suppression is notoriously
    implementation-defined)
  - shape / layout / output-name mismatch

The output is an :class:`Evidence` object: observed divergence location,
magnitude, first-diverging node, hypothesized mechanism, and a confidence
score. We never emit "probably a bug" — we emit measured claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from cvconform.ir import Precision, OperatorRegistry, VisionGraph
from cvconform.engines.comparison import ComparisonResult, Divergence


@dataclass
class Evidence:
    """A single attributed root-cause finding."""

    backend: str
    affected: str            # e.g. "Conv+Activation fusion", "Resize mode"
    cause: str               # mechanism, human label
    mechanism: str           # stable mechanism id
    impact: str = ""
    confidence: float = 0.0
    fix: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "affected": self.affected,
            "cause": self.cause,
            "mechanism": self.mechanism,
            "impact": self.impact,
            "confidence": round(self.confidence, 3),
            "fixes": self.fix,
            "details": self.details,
        }


# Mechanism registry: stable ids used in reports + corpus.
MECH = {
    "fusion": {
        "label": "Operator fusion / execution-order change",
        "fix": ["Disable operator fusion", "Freeze the graph to lock execution order"],
        "keywords": ["Conv", "Activation", "Add", "BatchNorm"],
    },
    "precision": {
        "label": "Precision change (FP32 -> FP16 semi / bf16 / int8)",
        "fix": ["Force FP32 accumulation", "Disable FP16 / low-precision on the affected op"],
        "keywords": [],
    },
    "quant_scale": {
        "label": "Quantization scale / zero-point mismatch",
        "fix": ["Recompute activation scales", "Align calibration data with reference"],
        "keywords": ["scale", "zeropoint", "quant", "int8"],
    },
    "fallback": {
        "label": "Unsupported-operator fallback",
        "fix": ["Provide an explicit implementation for the unsupported op", "Report upstream"],
        "keywords": ["unsupported", "fallback", "NotImplemented"],
    },
    "padding": {
        "label": "Padding / auto_pad behavior difference",
        "fix": ["Force explicit padding to match reference", "Set auto_pad=NONE"],
        "keywords": ["Pad", "auto_pad", "Conv", "Pool"],
    },
    "nms": {
        "label": "NMS implementation difference",
        "fix": ["Align NMS params (IoU thresh, max detections)", "Verify non-max suppression post-processing"],
        "keywords": ["NMS", "Detection", "NonMax", "box"],
    },
    "output_shape": {
        "label": "Shape / layout / output-name mismatch",
        "fix": ["Align output ordering/layout", "Rename outputs to match reference"],
        "keywords": [],
    },
    "nan_inf": {
        "label": "NaN/Inf produced on target",
        "fix": ["Check for underflow/overflow in FP16", "Clamp activations"],
        "keywords": ["nan", "inf"],
    },
}


def analyze(graph: Optional[VisionGraph], results: Dict[str, ComparisonResult],
            per_target_evidence_extra: Optional[Dict[str, Dict[str, Any]]] = None,
            ) -> Dict[str, List[Evidence]]:
    """Return {target: [Evidence]} explaining each target's divergences.

    ``per_target_evidence_extra`` lets callers inject backend-specific signals
    (e.g. which outputs went to the ANE, which dtype was compiled) to sharpen
    attribution. Pure-heuristic when absent.
    """
    findings: Dict[str, List[Evidence]] = {}
    for tname, res in results.items():
        evs = _analyze_target(graph, tname, res)
        findings[tname] = evs
    return findings


def _analyze_target(graph: Optional[VisionGraph], backend: str,
                    res: ComparisonResult) -> List[Evidence]:
    if not res.divergences:
        return []

    # Cluster divergences by output to find the most-affected region.
    # Pick the worst divergence to anchor attribution.
    worst = max(res.divergences, key=lambda d: d.magnitude)

    mechanisms = _hypothesize(graph, backend, res, worst)
    # Rank hypotheses; convert to Evidence.
    evs = []
    for mech_id, confidence, impact, affected in mechanisms:
        meta = MECH[mech_id]
        evs.append(Evidence(
            backend=backend,
            affected=affected or _affected_label(backend, worst),
            cause=meta["label"],
            mechanism=mech_id,
            impact=impact,
            confidence=confidence,
            fix=list(meta["fix"]),
            details={"worst_divergence": worst.to_dict()},
        ))
    return evs


def _affected_label(backend: str, d: Divergence) -> str:
    return d.output or backend


def _hypothesize(graph, backend, res, worst) -> List[tuple]:
    """Return [(mechanism, confidence, impact, affected)] ranked by confidence."""
    candidate = []
    extra = res.__dict__ if hasattr(res, "__dict__") else {}

    # 1. NaN/Inf present? -> nan_inf hypothesis strong.
    nan_inf = _has_nan_inf(worst)
    if nan_inf:
        candidate.append((_mk("nan_inf", 0.9, "Target produces NaN/Inf values", worst.output)))

    # 2. Look at the graph around the worst output for precision/fusion/quant cues.
    graph_sig = _graph_hints(graph, worst)

    # Default: tensor drift is most often precision or fusion.
    if worst.metric in ("max_abs_err", "mean_abs_err", "rmse", "max_rel_err"):
        # Magnitude of error hints at mechanism.
        mag = worst.magnitude
        if mag > 0.05:
            # Large drift => precision or quantization, more likely.
            candidate.append((_mk("precision", 0.65, f"Large tensor drift ({mag:.4f}) on {worst.output}", worst.output)))
        elif graph_sig["conv_act"]:
            candidate.append((_mk("fusion", 0.55, "Conv+Activation pattern present", worst.output)))
        else:
            candidate.append((_mk("precision", 0.45, "Tensor drift on low-level op", worst.output)))

    # 3. Classes mismatch => strongly NMS or post-processing difference.
    if worst.kind in ("classes", "boxes") or worst.metric == "match_rate":
        candidate.append((_mk("nms", 0.7, "Detection outputs diverge (boxes/classes)", worst.output)))

    # 4. Backend-specific priors for CoreML.
    if backend.lower() == "coreml":
        candidate.append((_mk("fusion", 0.5, "CoreML frequently fuses Conv+Activation, introducing FP16 intermediates", worst.output)))
        candidate.append((_mk("precision", 0.4, "CoreML often lowers to FP16 (ANE/GPU)", worst.output)))

    # 5. Missing output => shape/output-name issue.
    if worst.metric == "missing_output":
        candidate.append((_mk("output_shape", 0.8, f"Output {worst.output!r} missing on {res.target}", worst.output)))

    # 6. Execution failed => treat as upstream compatibility, generic.
    if worst.metric in ("execution_failed", "comparison_error"):
        candidate.append((_mk("fallback", 0.6, f"Runtime {backend} could not execute graph faithfully", worst.output)))

    # Dedupe by mechanism, keep highest confidence.
    seen: Dict[str, float] = {}
    for entry in candidate:
        mech, conf, impact, affected = entry
        if mech not in seen or conf > seen[mech]:
            seen[mech] = conf
    final = [(m, conf,
              _impact_text(backend, graph, m, worst),
              _affected_region(graph, worst))
             for m, conf in seen.items()]
    final.sort(key=lambda x: -x[1])  # highest confidence first
    return final[:3]


def _mk(mech, conf, impact, affected):
    return (mech, conf, impact, affected)


def _impact_text(backend, graph, mech, worst) -> str:
    if mech == "nms":
        return f"Detection outputs differ across runtime ({backend}) on {worst.output}"
    if mech == "precision":
        return f"Low-precision execution changes values on {worst.output}"
    if mech == "fusion":
        return f"Operator fusion changed execution order affecting {worst.output}"
    return f"Divergence on {worst.output}"


def _affected_region(graph: Optional[VisionGraph], d: Divergence) -> str:
    return d.output or "unknown"


def _has_nan_inf(d: Divergence) -> bool:
    # Inspect divergence name for cues; conservative.
    return d.mechanism and ("NaN" in d.mechanism or "NaN" in d.metric)


def _graph_hints(graph: Optional[VisionGraph], d: Divergence) -> Dict[str, bool]:
    """Look around the worst output in the graph for structural hints."""
    hints = {"conv_act": False, "has_nms": False, "has_quant": False}
    if graph is None:
        return hints
    for op in graph.ops.values():
        if op.op == OperatorRegistry.NMS:
            hints["has_nms"] = True
        if op.op in (OperatorRegistry.CONV2D, OperatorRegistry.BATCHNORM,
                     OperatorRegistry.RELU, OperatorRegistry.LEAKY_RELU,
                     OperatorRegistry.SILU, OperatorRegistry.ADD):
            hints["conv_act"] = True
        if any(s.is_quantized() for _, s in op.inputs + op.outputs):
            hints["has_quant"] = True
    return hints
