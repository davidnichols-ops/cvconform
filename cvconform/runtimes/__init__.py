"""Runtimes subpackage.

Each runtime is a thin, version-reporting wrapper that:
  1. executes a model on a given runtime backend,
  2. normalizes its raw outputs into the common :mod:`cvconform.schema` shape,
  3. reports its exact backend version so results are reproducible.

Everything below this layer is runtime-agnostic.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class RuntimeInfo:
    """Version + environment fingerprint for one runtime execution."""

    runtime: str                # e.g. "onnxruntime"
    runtime_version: str
    framework: str              # e.g. "onnx"
    framework_version: str
    provider: str = "CPU"
    device: str = ""
    env_fingerprint: str = ""
    extra: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runtime": self.runtime,
            "runtime_version": self.runtime_version,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "provider": self.provider,
            "device": self.device,
            "env": self.env_fingerprint,
            "extra": self.extra or {},
        }


def env_fingerprint() -> str:
    """Coarse fingerprint of the Python environment for reproducibility."""
    try:
        import platform

        return "{}-py{}.{}.{}".format(platform.machine(), *sys.version_info[:3])
    except Exception:
        return "unknown"


def normalize_tensor_outputs(outputs: Any, names: List[str]) -> Dict[str, "Output"]:
    """Convert raw numpy/torch outputs into {name: Output} with a common schema.

    ``outputs`` may be a dict-like, a list, or a single tensor.
    """
    from cvconform.schema import Output

    result: Dict[str, Output] = {}
    values: List[np.ndarray] = []

    if isinstance(outputs, dict):
        items = list(outputs.items())
    elif isinstance(outputs, (list, tuple)):
        items = list(zip(names, outputs))
    else:
        items = [(names[0] if names else "output", outputs)]

    for name, val in items:
        arr = to_numpy(val)
        values.append(arr)
        result[name] = Output.from_tensor(arr)

    return result


def to_numpy(val: Any) -> np.ndarray:
    """Convert a torch/numpy/arraylike value to numpy, detached, contiguous."""
    if hasattr(val, "detach"):
        val = val.detach()
    if hasattr(val, "cpu"):
        val = val.cpu()
    if hasattr(val, "numpy"):
        val = val.numpy()
    arr = np.asarray(val)
    if arr.dtype == object:
        arr = np.asarray(arr.tolist())
    return arr


def timed(fn, **kwargs):
    start = time.monotonic()
    out = fn(**kwargs)
    return out, time.monotonic() - start
