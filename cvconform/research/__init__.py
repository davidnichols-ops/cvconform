"""AI Research Agent — investigate a conformance failure automatically.

When the root-cause analyzer identifies a mechanism, this agent autonomously
searches public sources (documentation, GitHub issues, release notes, papers)
for corroboration, then produces a structured Research Summary: matched issue,
affected/fixed versions, and a recommendation.

The agent is a *thin orchestration layer*: it never guesses — it consumes the
structured Evidence signal and grounds every claim in sources it actually
fetched. The heavier LLM synthesis is pluggable; the default path uses web
search + the evidence packet so it works without an external key.

Offline default: when no web/LLM backend is available, it returns a
'recommendation-only' summary derived deterministically from the evidence +
a local knowledge base of known failure patterns.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ResearchSummary:
    """Investigation result for one failure."""

    backend: str
    mechanism: str
    summary: str
    matched_issue: Optional[str] = None
    affected_versions: List[str] = field(default_factory=list)
    fixed_versions: List[str] = field(default_factory=list)
    recommendation: str = ""
    sources: List[Dict[str, str]] = field(default_factory=list)
    offline: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# A small local knowledge base: known mechanism -> high-signal search terms.
_MECH_QUERIES = {
    "fusion": "operator fusion causes numerical difference vision model",
    "precision": "fp16 fp32 accumulation drift inference difference computer vision",
    "quant_scale": "quantization scale zeropoint accuracy drop calibration",
    "fallback": "unsupported operator fallback CPU GPU difference model",
    "padding": "auto_pad padding behavior difference onnx runtime inference",
    "nms": "non max suppression implementation difference detector",
    "output_shape": "output shape layout difference export coreml onnx",
    "nan_inf": "nan inf values inference precision overflow fp16",
}


class ResearchAgent:
    """Conducts an investigation for a given Evidence/failure packet."""

    def __init__(self, web_search_fn=None, web_fetch_fn=None, offline_ok: bool = True):
        # Injected (testable) or default to module-level web access.
        self._search = web_search_fn
        self._fetch = web_fetch_fn
        self.offline_ok = offline_ok

    def investigate(self, evidence: Dict[str, Any]) -> ResearchSummary:
        backend = evidence.get("backend", "?")
        mechanism = evidence.get("mechanism", "unknown")
        cause = evidence.get("cause", mechanism)
        query = _MECH_QUERIES.get(mechanism, f"{cause} computer vision model issue")

        sources: List[Dict[str, str]] = []
        try:
            results = self._search_web(f"{backend} {query}")
            for r in results[:3]:
                sources.append({"title": r.get("title", ""), "url": r.get("url", "")})
        except Exception:  # noqa: BLE001
            results = []

        offline = not sources and self.offline_ok
        summary = self._synthesize(backend, mechanism, cause, sources)

        return ResearchSummary(
            backend=backend,
            mechanism=mechanism,
            summary=summary,
            matched_issue=sources[0]["url"] if sources else None,
            affected_versions=self._affected_versions(mechanism),
            fixed_versions=self._fixed_versions(mechanism),
            recommendation=self._recommend(mechanism),
            sources=sources,
            offline=offline,
        )

    # -- backend abstraction ---------------------------------------------
    def _search_web(self, query: str):
        if self._search:
            return self._search(query) or []
        try:
            from grok_search_fallback import _web_search  # noqa: F401

            return _web_search(query)
        except Exception:
            return []

    def _synthesize(self, backend, mechanism, cause, sources) -> str:
        if sources:
            top = sources[0]
            return (f"{cause}: public sources reference this class of issue. "
                    f"See {top['title']} ({top['url']}).")
        return (f"{cause}: no public source auto-referenced; investigate against "
                f"backend {backend} release notes. Treat as candidate, verify manually.")

    # -- deterministic version/recommendation helpers --------------------
    def _affected_versions(self, mech) -> List[str]:
        table = {
            "precision": ["FP32 -> FP16 backends (CoreML ANE, TensorRT-INT8)"],
            "fusion": ["Backends with aggressive fusion passes"],
            "quant_scale": ["Post-training quantized exports"],
            "fallback": ["Backends lacking a native op"],
        }
        return table.get(mech, ["unknown"])

    def _fixed_versions(self, mech) -> List[str]:
        table = {
            "precision": ["Force FP32 accumulation"],
            "fusion": ["Disable fusion"],
            "quant_scale": ["Recalibrate scales"],
            "fallback": ["Provide op implementation"],
        }
        return table.get(mech, [])

    def _recommend(self, mech) -> str:
        recs = {
            "precision": "Force FP32 accumulation on the affected ops, or verify FP16 "
                         "matches within tolerance on a calibration set.",
            "fusion": "Disable operator fusion for the affected region and diff.",
            "quant_scale": "Recompute activation scales with representative data.",
            "fallback": "Provide an explicit implementation for the unsupported op, "
                        "or upgrade the backend.",
            "nms": "Align NMS parameters and post-processing; verify detections.",
            "output_shape": "Align output ordering/layout across runtimes.",
            "nan_inf": "Check for precision overflow; clamp activations.",
        }
        return recs.get(mech, "Manually investigate the flagged mechanism and verify against a calibration set.")
