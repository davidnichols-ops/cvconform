"""Fault Injection — deliberately break models to prove the system catches them.

Phase 4 of the mandate ("Break It"): we do not trust a conformance tool that has
never seen a real failure. This module injects known, category-labelled defects
into model artifacts and asserts that the verification + root-cause pipeline
*detects* each one and attributes a mechanism. Every injected archetype becomes
a permanent regression test (Regression Memory).

Fault kinds (each maps to a real-world deployment failure):
- ``weights_scale``     : scale all conv weights by a factor (value drift)
- ``precision_fp16``    : round activations/weights to fp16 (FP16 drift)
- ``weight_nan``        : inject NaN into a weight tensor (NaN propagation)
- ``weight_corrupt``    : zero out / randomize a conv weight block
- ``bias_shift``        : shift a bias (systematic output offset)
- ``output_permute``    : permute output channels (layout mismatch)
- ``act_clip``          : clip activations (clipping divergence)
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np

FAULT_CATALOG = [
    "weights_scale",
    "precision_fp16",
    "weight_nan",
    "weight_corrupt",
    "bias_shift",
    "output_permute",
    "act_clip",
]


class FaultInjector:
    """Applies a fault to a torch model graph (or returns a manifest)."""

    def __init__(self, model, seed: int = 0):
        self.model = model
        self.rng = np.random.default_rng(seed)

    # -- conv weight helpers ---------------------------------------------
    def _conv_modules(self) -> List[Any]:
        import torch

        mods = []
        for name, m in self.model.named_modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv1d)):
                mods.append((name, m))
        return mods

    def _first_conv(self):
        mods = self._conv_modules()
        return mods[0][1] if mods else None

    def _scale(self) -> float:
        return float(self.rng.uniform(2.0, 4.0))

    # -- fault application ------------------------------------------------
    def apply(self, fault: str) -> Dict[str, Any]:
        """Apply a labelled fault. Returns a manifest describing what changed."""
        import torch

        self.model.eval()
        manifest = {"fault": fault, "applied": True, "details": {}}
        with torch.no_grad():
            if fault == "weights_scale":
                conv = self._first_conv()
                conv.weight.mul_(self._scale())
                manifest["details"]["scale"] = self._scale()
            elif fault == "precision_fp16":
                conv = self._first_conv()
                # store fp16-rounded weights (simulate low-precision storage)
                wp = conv.weight.to(torch.float16).to(torch.float32)
                conv.weight.copy_(wp)
                manifest["details"]["round"] = "fp16"
            elif fault == "weight_nan":
                conv = self._first_conv()
                conv.weight.data[0, 0, 0, 0] = float("nan")
                manifest["details"]["nan_location"] = "first conv, first weight"
            elif fault == "weight_corrupt":
                conv = self._first_conv()
                conv.weight.data[0] = 0.0  # zero a whole output channel
                manifest["details"]["corruption"] = "channel 0 zeroed"
            elif fault == "bias_shift":
                conv = self._first_conv()
                if conv.bias is not None:
                    conv.bias.add_(0.5)
                manifest["details"]["bias_delta"] = 0.5
            elif fault == "output_permute":
                conv = self._first_conv()
                perm = torch.arange(conv.weight.shape[0])[::-1]
                conv.weight.copy_(conv.weight[perm])
                conv.bias.copy_(conv.bias[perm])
                manifest["details"]["permute"] = "output channels reversed"
            elif fault == "act_clip":
                conv = self._first_conv()
                torch.nn.utils.clip_grad_value_  # noqa: B018 (import guard only)
                conv.weight.data.clamp_(-0.1, 0.1)
                manifest["details"]["clip"] = "[-0.1, 0.1]"
            else:
                raise ValueError(f"unknown fault {fault!r}")
        return manifest


def _make_faulty_model(source_path: str, fault: str, seed: int = 0):
    """Build a faulty *traced* model from the clean source for a given fault."""
    import torch

    clean = torch.jit.load(source_path)
    # Un-trace to an eager module isn't trivial from a saved traced module;
    # instead re-derive via the demo builder so we can mutate weights.
    import sys

    sys.path.insert(0, "examples")
    from demo_model import build_demo_model
    from cvconform.runtimes.pytorch_rt import to_numpy  # noqa

    model = build_demo_model()
    model.eval()
    inj = FaultInjector(model, seed)
    inj.apply(fault)
    traced = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
    return traced
