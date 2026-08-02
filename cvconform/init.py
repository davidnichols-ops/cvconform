"""`cvconform init` — zero-config project bootstrap.

Scans a repo, detects the model contract + calibration data, writes a
`.cvconform.yaml`, and prints what it found. Idempotent and harmless on
model-less repos.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from cvconform.autodetect import detect_repo, detect_model, scan_for_images
from cvconform.config import CVConformConfig, find_config, write_config
from cvconform.registry import ModelRegistry


def init_project(root: str = ".", force: bool = False,
                 registry=None) -> Dict[str, Any]:
    """Bootstrap a .cvconform.yaml, returning a summary dict."""
    root = os.path.abspath(root)
    reg = registry or ModelRegistry()

    existing = find_config(root)
    if existing and not force:
        return {"status": "exists", "config": existing,
                "message": f"config already present: {existing} (pass --force to rewrite)"}

    detection = detect_repo(root, reg)
    models = detection["models"]
    images = detection["images"]

    cfg = CVConformConfig()
    if models:
        pref = models[0]
        # reference runtime depends on format availability; default:
        ref = _default_reference(pref["format"])
        cfg.model = {
            "path": os.path.relpath(pref["path"], root),
            "format": pref["format"],
            "task": pref["task"],
            "input": {"name": pref["input_name"], "shape": pref["input_shape"]},
            "outputs": pref["outputs"],
        }
        cfg.reference = ref
        cfg.calibration = {
            "source": "dir" if images else "auto",
            "path": _calib_subdir(root, images),
        }
    else:
        cfg.model = {}

    config_path = os.path.join(root, ".cvconform.yaml")
    write_config(config_path, cfg)

    return {
        "status": "written",
        "config": config_path,
        "models_found": len(models),
        "preferred_model": cfg.model.get("path") if cfg.model else None,
        "images_found": len(images),
        "reference": cfg.reference,
        "registry_match": pref.get("registry_key") if models else "n/a",
        "models": models,
    }


def _default_reference(fmt: str) -> str:
    # Maps a model format to a reasonable reference runtime for differential.
    # torchscript -> pytorch native; others -> onnxruntime (self-describing).
    if fmt == "torchscript":
        return "pytorch"
    if fmt == "coreml":
        return "coreml"
    return "onnx"


def _calib_subdir(root: str, images: list) -> str:
    """Choose a calibration path: dir with most images, else ''."""
    if not images:
        return ""
    # return directory of the first image (simplest, works for flat dirs)
    return os.path.dirname(images[0])


def summarize_init(result: Dict[str, Any]) -> str:
    if result["status"] == "exists":
        return result["message"]
    L = ["cvconform init"]
    L.append("-" * 50)
    L.append(f"config:           {result['config']}")
    L.append(f"models found:     {result['models_found']}")
    if result["preferred_model"]:
        L.append(f"preferred model:  {result['preferred_model']}")
        L.append(f"reference:        {result['reference']}")
        L.append(f"registry match:   {result['registry_key'] if False else result.get('registry_match')}")
    L.append(f"calibration imgs: {result['images_found']}")
    if not result["models_found"]:
        L.append("no model detected — .cvconform.yaml written but empty")
    L.append("")
    L.append("next:  cvconform verify")
    return "\n".join(L)
