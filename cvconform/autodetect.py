"""Auto-discovery — find models, calibration data, and their contracts.

This is the zero-config brain: given a repo (or a single model path), it finds
model artifacts, detects their format, resolves their input/output contract
(from the artifact or the model registry), and locates calibration images.

Nothing here *requires* config — it's all best-effort so ``verify`` works with
zero files.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from cvconform.registry import ModelRegistry, ModelEntry, default_entry

# Format detection by extension.
FORMAT_BY_EXT = {
    ".pt": "torchscript", ".pth": "torchscript", ".jit": "torchscript",
    ".onnx": "onnx",
    ".mlmodel": "coreml", ".mlpackage": "coreml",
    ".tflite": "tflite",
    ".engine": "tensorrt", ".trt": "tensorrt",
    ".xml": "openvino",
    ".safetensors": "mlx", ".mlx": "mlx",
}

# Feature-map scales for known tasks (used for detection head output inference).
_TASK_FEATURE_MULT = {
    "detection": 32,      # input // 32 -> feature map dim
    "pose": 32,
    "segmentation": 32,
    "classification": 0,  # no spatial output
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".npy"}


@dataclass
class DetectedModel:
    path: str
    format: str
    name: str = ""
    task: str = "classification"
    input_name: str = "input"
    input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    outputs: List[str] = field(default_factory=list)
    tolerance: Dict[str, float] = field(default_factory=dict)
    registry_key: str = "heuristic"
    source: str = "file"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "format": self.format,
            "name": self.name,
            "task": self.task,
            "input_name": self.input_name,
            "input_shape": list(self.input_shape),
            "outputs": self.outputs,
            "tolerance": self.tolerance,
            "registry_key": self.registry_key,
        }


def detect_format(path: str) -> Optional[str]:
    ext = os.path.splitext(path)[1].lower()
    return FORMAT_BY_EXT.get(ext)


def scan_for_models(root: str, max_results: int = 8) -> List[str]:
    """Find candidate model artifacts in a directory (best-effort)."""
    found = []
    for ext in FORMAT_BY_EXT:
        if ext == ".xml":
            continue  # openvino needs a sibling .bin; treat carefully
        found.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
    # de-dup + drop obvious cache/venv paths
    dedup = []
    seen = set()
    for p in found:
        norm = os.path.normpath(p)
        if any(seg in ("venv", ".venv", "node_modules", ".git", "__pycache__")
               for seg in norm.split(os.sep)):
            continue
        if norm not in seen:
            seen.add(norm)
            dedup.append(norm)
    dedup.sort(key=lambda p: -os.path.getsize(p))  # largest (real) first
    return dedup[:max_results]


def scan_for_images(root: str, max_results: int = 100) -> List[str]:
    """Find calibration images in a directory tree."""
    found = []
    for ext in IMAGE_EXTS:
        found.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
    dedup = []
    for p in found:
        norm = os.path.normpath(p)
        if any(seg in ("venv", ".venv", "node_modules", ".git", "__pycache__",
                       "examples_output", "corpus", "results")
               for seg in norm.split(os.sep)):
            continue
        dedup.append(norm)
    return sorted(dedup)[:max_results]


def _read_onnx_contract(path: str) -> Tuple[Optional[Tuple[int, ...]], List[str], List[str]]:
    """Extract (input_shape, input_names, output_names) from an ONNX model."""
    try:
        import onnx

        m = onnx.load(path)
        inp = m.graph.input[0]
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            if d.HasField("dim_value"):
                dims.append(int(d.dim_value))
            else:
                dims.append(1)  # dynamic -> assume batch 1
        if dims and dims[0] != 1:
            dims[0] = 1
        in_names = [i.name for i in m.graph.input]
        out_names = [o.name for o in m.graph.output]
        return (tuple(dims), in_names, out_names)
    except Exception:
        return (None, [], [])


def _read_coreml_contract(path: str):
    try:
        import coremltools as ct

        m = ct.models.MLModel(path)
        spec = m.get_spec()
        inp = spec.description.input[0]
        if inp.type.HasField("imageType"):
            shape = (1, 3, inp.type.imageType.height, inp.type.imageType.width)
        elif inp.type.HasField("multiArrayType"):
            shape = tuple([1] + list(inp.type.multiArrayType.shape))
        else:
            shape = (1, 3, 224, 224)
        in_names = [i.name for i in spec.description.input]
        out_names = [o.name for o in spec.description.output]
        return (shape, in_names, out_names)
    except Exception:
        return (None, [], [])


def detect_model(path: str, registry: Optional[ModelRegistry] = None) -> DetectedModel:
    """Best-effort contract for a single model artifact."""
    reg = registry or ModelRegistry()
    fmt = detect_format(path)
    if fmt is None:
        raise ValueError(f"cannot detect model format for {path}")
    base = os.path.basename(path)
    entry = reg.lookup(base) or default_entry()

    shape, in_names, out_names = None, [], []
    if fmt == "onnx":
        shape, in_names, out_names = _read_onnx_contract(path)
    elif fmt == "coreml":
        shape, in_names, out_names = _read_coreml_contract(path)

    # Registry may know the architecture even if introspection fails.
    use_entry_shape = entry.key != "unknown" and shape is None
    if shape is None:
        shape = entry.input_shape if use_entry_shape else (1, 3, 224, 224)
    input_name = (in_names[0] if in_names else None) or entry.input_name
    outputs = out_names or entry.outputs or ["output"]
    task = entry.task

    return DetectedModel(
        path=path, format=fmt, name=base, task=task,
        input_name=input_name, input_shape=shape,
        outputs=outputs, tolerance=dict(entry.tolerance),
        registry_key=entry.key,
    )


def detect_repo(root: str, registry: Optional[ModelRegistry] = None) -> Dict[str, Any]:
    """Detect the full repo contract: models + calibration images."""
    reg = registry or ModelRegistry()
    models = scan_for_models(root)
    # Prefer .onnx (self-describing) then torchscript; pick the largest.
    ordered = sorted(models, key=lambda p: 0 if detect_format(p) == "onnx"
                     else (1 if detect_format(p) in ("torchscript", "coreml") else 2))
    images = scan_for_images(root)

    result = {"models": [], "images": [], "preferred_model": None}
    for p in ordered:
        try:
            result["models"].append(detect_model(p, reg).to_dict())
        except Exception:
            continue
    result["images"] = images[:64]
    if result["models"]:
        result["preferred_model"] = result["models"][0]["path"]
    return result
