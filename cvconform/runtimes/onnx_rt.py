"""Runtime: ONNX via onnxruntime.

This is the reference differential target — ONNX is the common interchange
format, and onnxruntime provides a faithful, widely-deployed executor. It runs
on CPU, and on Apple Silicon exposes a CoreMLExecutionProvider as well (we
default to CPU so tensors stay on-device and comparable, but allow overriding).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from cvconform.runtimes import RuntimeInfo, env_fingerprint, normalize_tensor_outputs, to_numpy

try:
    import onnxruntime as ort
    import onnx
    _HAS = True
except Exception:  # pragma: no cover
    ort = None
    onnx = None
    _HAS = False


def _remap_feeds(feeds, declared):
    """Map caller feed keys to the model's declared input names.

    If the caller already used the exact declared names, pass through. If the
    caller used positional/generic names (input_0, input, data, ...), assign by
    position.
    """
    if not declared:
        return feeds
    if any(k in declared for k in feeds):
        return feeds
    out = {}
    for i, (k, v) in enumerate(feeds.items()):
        if i < len(declared):
            out[declared[i]] = v
        else:
            out[k] = v
    return out


class OnnxRuntime:
    runtime_name = "onnxruntime"

    def __init__(self, providers: Optional[List[str]] = None,
                 sess_options: Optional[Dict[str, Any]] = None):
        if not _HAS:
            raise ImportError("onnxruntime and onnx are required for onnxruntime runtime")
        self.providers = providers
        self.sess_options = sess_options or {}

    # -- version reporting -------------------------------------------------
    def info(self) -> RuntimeInfo:
        eps = getattr(ort, "get_available_providers", lambda: [])()
        providers_used = self.providers or [p for p in eps if p in ort.get_available_providers()] or eps
        return RuntimeInfo(
            runtime=self.runtime_name,
            runtime_version=ort.__version__,
            framework="onnx",
            framework_version=onnx.__version__,
            provider=",".join(providers_used),
            device="cpu" if "CPUExecutionProvider" in providers_used else "other",
            env_fingerprint=env_fingerprint(),
            extra={"available_providers": eps},
        )

    # -- execution ---------------------------------------------------------
    def run(self, model_bytes: bytes, feeds: Dict[str, np.ndarray],
            output_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute an in-memory ONNX model on the given feeds.

        Feeds are remapped by the model's declared input names (falling back to
        position when the caller used generic names), so a caller can pass
        reference-style names and still hit the right tensors.
        """
        so = ort.SessionOptions()
        for k, v in self.sess_options.items():
            setattr(so, k, v)
        providers = self.providers or ort.get_available_providers()
        sess = ort.InferenceSession(model_bytes, sess_options=so, providers=providers)
        names = output_names or [o.name for o in sess.get_outputs()]
        declared = [i.name for i in sess.get_inputs()]
        feed_mapped = _remap_feeds(feeds, declared)
        res = sess.run(names, {k: np.asarray(v) for k, v in feed_mapped.items()})
        return dict(zip(names, res))

    def run_and_normalize(self, model_bytes: bytes, feeds: Dict[str, np.ndarray],
                          output_names: Optional[List[str]] = None) -> Dict[str, Any]:
        raw = self.run(model_bytes, feeds, output_names)
        return normalize_tensor_outputs(raw, list(raw.keys()))


def load_onnx_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()
