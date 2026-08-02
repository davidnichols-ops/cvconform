"""Differential Execution Engine — the heart of cvconform.

Given a model, a reference runtime, and a set of target runtimes, this engine:

  1. generates (or loads) a seedable batch of inputs,
  2. executes the reference and every target on the *same* inputs,
  3. normalizes outputs into the common schema,
  4. compares each target to the reference,
  5. aggregates into a per-target conformance result.

Determinism contract: same model + same runtimes/versions + same seed
=> identical evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from cvconform.runtimes import to_numpy
from cvconform.schema import Output
from cvconform.engines.comparison import (
    TolerancePolicy,
    ComparisonResult,
    Divergence,
    compare_target,
    build_divergences,
)


@dataclass
class Differ:
    """Holds one target runtime + its compiled model artifact."""

    name: str
    runtime: Any
    artifact: Any  # compiled model handle (bytes, MLModel, torch module, ...)

    def setup_inputs(self, feeds: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Fine-tune feeds for this runtime (e.g. CoreML dtype/layouts)."""
        r = self.runtime
        # CoreML wants float32 and sometimes specific channel order; numpy
        # arrays are already right, but ensure contiguous float32 for MLModel.
        adapted = {}
        for k, v in feeds.items():
            arr = np.asarray(v)
            if getattr(r, "runtime_name", "") == "coreml" and arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            adapted[k] = arr
        return adapted


class DifferentialEngine:
    """Compares a reference runtime against multiple target runtimes."""

    def __init__(self, reference_runtime: Any, reference_artifact: Any,
                 reference_outputs: List[str],
                 policy: Optional[TolerancePolicy] = None):
        self.reference_runtime = reference_runtime
        self.reference_artifact = reference_artifact
        self.reference_outputs = reference_outputs
        self.policy = policy or TolerancePolicy()
        self.targets: List[Differ] = []

    def add_target(self, name: str, runtime: Any, artifact: Any) -> "DifferentialEngine":
        self.targets.append(Differ(name=name, runtime=runtime, artifact=artifact))
        return self

    def run(self, feeds: Dict[str, np.ndarray]) -> Dict[str, ComparisonResult]:
        """Execute reference + all targets on one feed batch, return per-target results."""
        # Reference
        ref_outputs = self._run_reference(feeds)
        results: Dict[str, ComparisonResult] = {}
        for tgt in self.targets:
            try:
                tgt_outputs = self._run_target(tgt, feeds)
            except Exception as e:  # noqa: BLE001
                # A target that fails to even run is a hard, attributable failure.
                res = ComparisonResult(
                    target=tgt.name, reference=self.reference_runtime.runtime_name,
                    per_output={}, divergences=[],
                    scores={}, overall_score=0.0, is_conformant=False,
                )
                res.divergences.append(
                    Divergence("<model>", "runtime", "execution_failed",
                               observed=1.0, threshold=0.0, magnitude=1.0,
                               mechanism=f"Runtime {tgt.name} failed to execute: {e}")
                )
                results[tgt.name] = res
                continue
            try:
                ref_for_cmp, tgt_for_cmp = _align_outputs(
                    self._filter_outputs(ref_outputs, self.reference_outputs),
                    self._filter_outputs(tgt_outputs, self.reference_outputs),
                )
                res = compare_target(
                    reference_outputs=ref_for_cmp,
                    target_outputs=tgt_for_cmp,
                    target_name=tgt.name,
                    reference_name=self.reference_runtime.runtime_name,
                    policy=self.policy,
                )
            except Exception as e:  # noqa: BLE001
                res = ComparisonResult(
                    target=tgt.name, reference=self.reference_runtime.runtime_name,
                    per_output={}, divergences=[],
                    scores={}, overall_score=0.0, is_conformant=False,
                )
                res.divergences.append(
                    Divergence("<model>", "runtime", "comparison_error",
                               observed=1.0, threshold=0.0, magnitude=1.0,
                               mechanism=f"Comparison for {tgt.name} failed: {e}")
                )
            results[tgt.name] = res
        return results

    def _run_reference(self, feeds: Dict[str, np.ndarray]) -> Dict[str, Output]:
        rt = self.reference_runtime
        raw = rt.run_and_normalize(self.reference_artifact, feeds,
                                   self.reference_outputs)
        return raw

    def _run_target(self, tgt: Differ, feeds: Dict[str, np.ndarray]) -> Dict[str, Output]:
        ad = tgt.setup_inputs(feeds)
        raw = tgt.runtime.run_and_normalize(tgt.artifact, ad, None)
        return raw

    def _filter_outputs(self, outputs: Dict[str, Output], names: List[str]) -> Dict[str, Output]:
        """Keep only the semantic output names we care about.

        If the requested names don't match what the model actually produced
        (e.g. a config guessed one 'output' but the head emits output_0/1/2),
        return all outputs rather than an empty/filtered set — otherwise the
        comparison silently sees nothing.
        """
        if names and set(names) <= set(outputs):
            return {n: outputs[n] for n in names}
        return outputs


def _align_outputs(ref: Dict[str, Output], tgt: Dict[str, Output]):
    """Align reference and target outputs for comparison.

    Prefers exact name matching. When names differ (e.g. CoreML renames
    outputs to internal ``var_`` names, and model heads emit positional tuples
    that become ``output_0..``), falls back to pure positional alignment: the
    i-th reference output is compared to the i-th target output, and both are
    re-keyed to a common ordinal label so comparison proceeds.
    """
    # Exact name overlap covers the common, well-named case.
    shared = [n for n in ref if n in tgt]
    if shared and len(shared) == max(len(ref), len(tgt)):
        return {n: ref[n] for n in shared}, {n: tgt[n] for n in shared}
    # Positional fallback: same count -> align by order under common keys.
    if len(ref) == len(tgt) and len(ref) > 0:
        r_keys = list(ref)
        t_keys = list(tgt)
        r_aligned, t_aligned = {}, {}
        for i in range(len(ref)):
            common = f"_pos_{i}"
            r_aligned[common] = ref[r_keys[i]]
            t_aligned[common] = tgt[t_keys[i]]
        return r_aligned, t_aligned
    return ref, tgt


def generate_inputs(shape: Tuple[int, ...], seed: int, n: int = 1,
                    rng_min: float = 0.0, rng_max: float = 1.0,
                    dtype=np.float32) -> List[np.ndarray]:
    """Generate seedable synthetic inputs (uniform noise by default).

    Deterministic: same shape+seed => same tensors.
    """
    rng = np.random.default_rng(seed)
    return [rng.uniform(rng_min, rng_max, size=shape).astype(dtype) for _ in range(n)]


def generate_image_inputs(shape: Tuple[int, ...], seed: int, n: int = 1,
                          kind: str = "noise", dtype=np.float32) -> List[np.ndarray]:
    """Generate seedable image-like inputs. kinds:
    - noise: uniform noise
    - smooth: slowly-varying gradients (more realistic)
    - edges: high-contrast grid
    """
    rng = np.random.default_rng(seed)
    imgs = []
    for _ in range(n):
        if kind == "noise":
            img = rng.uniform(0, 1, size=shape).astype(dtype)
        elif kind == "smooth":
            c, h, w = shape[0], shape[-2], shape[-1]
            base = np.zeros((h, w), dtype=np.float32)
            yy, xx = np.mgrid[0:h, 0:w]
            base = (xx / max(w - 1, 1)) * 0.8 + 0.1 + rng.uniform(-0.05, 0.05, (h, w))
            img = np.stack([base] * c, axis=0)[np.newaxis, ...]
            img = np.broadcast_to(img, shape).copy().astype(dtype)
        elif kind == "edges":
            img = np.zeros(shape, dtype=dtype)
            img[..., ::8, :] = 1.0
            img[..., :, ::8] = 1.0
        imgs.append(img)
    return imgs
