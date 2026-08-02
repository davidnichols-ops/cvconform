"""Runtime: PyTorch (reference backend).

PyTorch is the *training* environment — the FP32 reference that everything else
is compared against. We execute torch models (nn.Module or torch.jit module)
directly, detach to CPU, and normalize.

The reference runtime is where "this is the same model" originates; the
differential engine compares every other target back to this.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from cvconform.runtimes import RuntimeInfo, env_fingerprint
from cvconform.runtimes import normalize_tensor_outputs, to_numpy

try:
    import torch
    _HAS = True
except Exception:  # pragma: no cover
    torch = None
    _HAS = False


class PyTorchRuntime:
    runtime_name = "pytorch"

    def __init__(self, device: str = "cpu"):
        if not _HAS:
            raise ImportError("torch is required for the pytorch runtime")
        self.device = device

    def info(self) -> RuntimeInfo:
        extra = {}
        if torch:
            extra["cuda_available"] = bool(torch.cuda.is_available())
            mps_backend = getattr(torch.backends, "mps", None)
            try:
                extra["mps_available"] = bool(mps_backend.is_available()) if mps_backend else False
            except Exception:
                extra["mps_available"] = False
        return RuntimeInfo(
            runtime=self.runtime_name,
            runtime_version=torch.__version__,
            framework="torch",
            framework_version=torch.__version__,
            provider="pytorch",
            device=self.device,
            env_fingerprint=env_fingerprint(),
            extra=extra,
        )

    def run(self, model: Any, feeds: Dict[str, np.ndarray],
            output_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run a torch model. ``model`` is an nn.Module or jit ScriptModule.

        Feeds are mapped by input name when the model exposes named inputs
        (via :func:`torch.jit.script` / tracer), else by position.
        """
        torch.set_num_threads(1)  # determinism on CPU
        with torch.no_grad():
            inputs = [torch.as_tensor(np.asarray(v)).to(self.device) for v in feeds.values()]
            try:
                out = model(*inputs)
            except TypeError:
                # fallback: keyword inputs when model supports them
                kw = {k: torch.as_tensor(np.asarray(v)).to(self.device)
                      for k, v in feeds.items()}
                out = model(**kw)
        if isinstance(out, (list, tuple)):
            res = list(out)
        else:
            res = [out]
        # If caller provided too few/many names, regenerate to match count so
        # normalization never index-errors on an unknown-arity model.
        if output_names is not None and len(output_names) != len(res):
            output_names = None
        names = output_names or [f"output_{i}" for i in range(len(res))]
        normalized = []
        for o in res:
            normalized.append(to_numpy(o))
        return {names[i]: normalized[i] for i in range(len(normalized))}

    def run_and_normalize(self, model: Any, feeds: Dict[str, np.ndarray],
                          output_names: Optional[List[str]] = None) -> Dict[str, Any]:
        # PyTorch outputs may be boxes/classes if a model has a detection head;
        # for the generic path we treat all as tensors (specialized models
        # override normalization). See engines.differential for head awareness.
        raw = self.run(model, feeds, output_names)
        return normalize_tensor_outputs(raw, list(raw.keys()))
