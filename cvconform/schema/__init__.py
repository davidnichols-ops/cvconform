"""Output schema + normalization.

Every runtime produces :class:`Output` objects that share a single, comparable
schema. Outputs are keyed by *semantic name* (the graph-level output label), so
runtimes that reorder or rename the physical output bindings still compare
correctly.

Output kinds:
- tensor  — a dense numeric tensor (feature maps, logits, embeddings)
- boxes   — a set of bounding boxes (with class+score) — a detection head output
- masks   — segmentation masks
- classes — a flat list of class ids
- scores  — a flat list of confidence scores
- scalar  — a single scalar
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from cvconform.ir import Precision


@dataclass
class TensorStats:
    shape: List[int]
    dtype: str
    precision: str
    nan: float = 0.0
    inf: float = 0.0
    min: float = 0.0
    max: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    l2_norm: float = 0.0


@dataclass
class Box:
    coords: List[float]  # [x1, y1, x2, y2] (normalized or absolute — from model)
    class_id: int
    score: float


@dataclass
class Output:
    kind: str
    name: str
    # tensor kind
    tensor: Optional[np.ndarray] = None
    stats: Optional[TensorStats] = None
    # boxes kind
    boxes: List[Box] = field(default_factory=list)
    # classes/scores kind
    values: Optional[List[Any]] = None

    @classmethod
    def from_tensor(cls, arr: np.ndarray, name: str = "output") -> "Output":
        arr = np.asarray(arr)
        flat = arr.ravel()
        finite = flat[np.isfinite(flat)] if flat.size else flat
        stats = TensorStats(
            shape=list(arr.shape),
            dtype=str(arr.dtype),
            precision=Precision.from_dtype(str(arr.dtype)).value,
            nan=float(np.isnan(flat).mean()) if flat.size else 0.0,
            inf=float(np.isinf(flat).mean()) if flat.size else 0.0,
            min=float(finite.min()) if finite.size else 0.0,
            max=float(finite.max()) if finite.size else 0.0,
            mean=float(finite.mean()) if finite.size else 0.0,
            std=float(finite.std()) if finite.size else 0.0,
            l2_norm=float(np.linalg.norm(finite)) if finite.size else 0.0,
        )
        return cls(kind="tensor", name=name, tensor=arr, stats=stats)

    @classmethod
    def from_boxes(cls, boxes: List[Box], name: str = "detections") -> "Output":
        return cls(kind="boxes", name=name, boxes=boxes)

    @classmethod
    def from_values(cls, values: List[Any], kind: str, name: str) -> "Output":
        return cls(kind=kind, name=name, values=values)

    def describe(self) -> str:
        if self.kind == "tensor" and self.stats:
            s = self.stats
            return (f"[tensor {s.shape} {s.dtype} "
                    f"min={s.min:.3g} max={s.max:.3g} mean={s.mean:.3g} "
                    f"nan={s.nan:.2%} inf={s.inf:.2%}]")
        if self.kind == "boxes":
            return f"[boxes n={len(self.boxes)}]"
        if self.kind in ("classes", "scores"):
            return f"[{self.kind} n={len(self.values or [])}]"
        return f"[{self.kind}]"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"kind": self.kind, "name": self.name}
        if self.stats:
            d["tensor"] = {
                "shape": self.stats.shape,
                "dtype": self.stats.dtype,
                "precision": self.stats.precision,
                "nan_frac": self.stats.nan,
                "inf_frac": self.stats.inf,
                "min": self.stats.min,
                "max": self.stats.max,
                "mean": self.stats.mean,
                "std": self.stats.std,
                "l2_norm": self.stats.l2_norm,
            }
        if self.kind == "boxes":
            d["boxes"] = [{"coords": b.coords, "class": b.class_id, "score": b.score}
                          for b in self.boxes]
        if self.values is not None:
            d["values"] = self.values[:64]  # cap for readability
        return d
