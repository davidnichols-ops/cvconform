"""Model Registry — known-good per-architecture configs.

This is what lets ``cvconform verify`` pick sane input shapes, task type, and
tolerances without the user reading model source. It ships with curated
entries for common families (torchvision, timm, ultralytics, HF) and a
generous heuristic fallback for anything unknown.

Each entry:
  key          : canonical id used in configs / registry lookup
  matches      : substrings to match against the model name/path
  task         : detection | segmentation | classification | pose | ocr | generation | embedding
  input_shape  : default [N, C, H, W]
  input_name   : likely input feed name
  outputs      : likely semantic output names
  tolerance    : suggested TolerancePolicy overrides
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ModelEntry:
    key: str
    task: str
    input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    input_name: str = "input"
    outputs: List[str] = field(default_factory=list)
    tolerance: Dict[str, float] = field(default_factory=dict)
    matches: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "task": self.task,
            "input_shape": list(self.input_shape),
            "input_name": self.input_name,
            "outputs": list(self.outputs),
            "tolerance": dict(self.tolerance),
        }


# Curated registry. Order matters: first match wins (most specific first).
MODEL_REGISTRY: List[ModelEntry] = [
    # ---- detection / YOLO family ----
    ModelEntry("yolo", "detection", (1, 3, 640, 640), "images",
               ["output0", "output1", "output2"],
               {"iou_threshold": 0.5, "max_abs_err": 1e-2},
               ["yolo", "yolov", "yolo11", "yolov8", "yolov5", "rtdetr"]),
    # ---- SAM ----
    ModelEntry("sam", "segmentation", (1, 3, 1024, 1024), "image",
               ["masks", "iou_predictions", "low_res_logits"],
               {"iou_threshold": 0.7, "max_abs_err": 1e-3},
               ["sam", "segment-anything", "segment_anything"]),
    # ---- DETR / transformers detectors ----
    ModelEntry("detr", "detection", (1, 3, 800, 800), "pixel_values",
               ["logits", "pred_boxes"],
               {"iou_threshold": 0.5},
               ["detr", "deformable_detr", "conditional_detr"]),
    # ---- classification families ----
    ModelEntry("resnet", "classification", (1, 3, 224, 224), "input",
               ["output", "logits"],
               {"max_abs_err": 1e-3, "class_agreement": 1.0},
               ["resnet", "resnext", "wide_resnet"]),
    ModelEntry("efficientnet", "classification", (1, 3, 224, 224), "input",
               ["output", "logits"],
               {"max_abs_err": 1e-3},
               ["efficientnet", "efficientvit"]),
    # ---- embedding (must precede generic vit)
    ModelEntry("clip", "embedding", (1, 3, 224, 224), "pixel_values",
               ["image_embeds", "image_features", "embedding"],
               {"max_abs_err": 1e-3},
               ["clip", "image_encoder", "openai/clip"]),
    ModelEntry("vit", "classification", (1, 3, 224, 224), "pixel_values",
               ["logits", "output"],
               {"max_abs_err": 1e-3},
               ["vision_transformer", "deit", "vit-base-patch", "vit-large-patch",
                "vit_small", "vit_base", "vit_large", "vit_huge", "vit-tiny", "vit-small"]),
    ModelEntry("swin", "classification", (1, 3, 224, 224), "pixel_values",
               ["logits"],
               {"max_abs_err": 1e-3},
               ["swin"]),
    ModelEntry("convnext", "classification", (1, 3, 224, 224), "input",
               ["output"],
               {"max_abs_err": 1e-3},
               ["convnext"]),
    ModelEntry("mobilenet", "classification", (1, 3, 224, 224), "input",
               ["output"],
               {"max_abs_err": 1e-3},
               ["mobilenet"]),
    ModelEntry("densenet", "classification", (1, 3, 224, 224), "input",
               ["output"],
               {"max_abs_err": 1e-3},
               ["densenet"]),
    ModelEntry("alexnet", "classification", (1, 3, 224, 224), "input",
               ["output"],
               {"max_abs_err": 1e-3},
               ["alexnet", "vgg"]),
    # ---- segmentation ----
    ModelEntry("unet", "segmentation", (1, 3, 512, 512), "input",
               ["output", "masks"],
               {"iou_threshold": 0.7},
               ["unet", "segformer", "deeplab", "fcn"]),
    # ---- pose ----
    ModelEntry("pose", "pose", (1, 3, 640, 640), "images",
               ["output0", "output1"],
               {"iou_threshold": 0.5, "max_abs_err": 1e-2},
               ["pose", "keypoint", "openpose", "hrnet", "body"]),
    # ---- OCR ----
    ModelEntry("ocr", "ocr", (1, 3, 320, 320), "input",
               ["output", "logits"],
               {"class_agreement": 1.0},
               ["crnn", "easyocr", "paddleocr", "trocr", "donut"]),
    # ---- generation-ish ----
    ModelEntry("stable_diffusion", "generation", (1, 4, 64, 64), "latent",
               ["latent", "sample"],
               {"max_abs_err": 1e-2},
               ["stable_diffusion", "sdxl", "sd-vae", "vae"]),
]


class ModelRegistry:
    def __init__(self, entries: Optional[List[ModelEntry]] = None):
        self.entries = entries or MODEL_REGISTRY

    def lookup(self, name: str) -> Optional[ModelEntry]:
        """Match a model name/path against the registry (first match wins)."""
        n = (name or "").lower()
        for e in self.entries:
            for m in e.matches:
                if m in n:
                    return e
        return None

    def add(self, entry: ModelEntry) -> None:
        self.entries.append(entry)

    def to_dict(self) -> Dict[str, Any]:
        return {e.key: e.to_dict() for e in self.entries}

    def describe(self) -> str:
        lines = []
        for e in self.entries:
            lines.append(f"  {e.key:<16} task={e.task:<14} {list(e.input_shape)} "
                         f"in={e.input_name!r} outs={e.outputs}")
        return "\n".join(lines)


def default_entry() -> ModelEntry:
    return ModelEntry("unknown", "classification", (1, 3, 224, 224), "input",
                      ["output"], {"max_abs_err": 1e-3}, [])
