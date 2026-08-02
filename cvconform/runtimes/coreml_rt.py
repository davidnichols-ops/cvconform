"""Runtime: CoreML via coremltools.

CoreML is the exotic-but-critical target: it is where FP16 accumulation drift,
operator fusion, and quantized weights do the most silent damage. We compile a
CoreML model (from .mlmodel or .mlpackage) with coremltools' model load + a
CoreML prediction. We also expose the raw CoreML predicted output for
comparison.

Because CoreML executes on Apple's Neural Engine / GPU through
``coremltools.models.MLModel``, we keep the wrapper thin and never assume a
particular topology — we just normalize whatever the model returns.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from cvconform.runtimes import RuntimeInfo, env_fingerprint, normalize_tensor_outputs, to_numpy
from cvconform.schema import Box, Output

try:
    import coremltools as ct
    _HAS = True
except Exception:  # pragma: no cover
    ct = None
    _HAS = False


class CoreMLRuntime:
    runtime_name = "coreml"

    def __init__(self, compute_units=None, provider: str = "coreml"):
        if not _HAS:
            raise ImportError("coremltools is required for the coreml runtime")
        self.compute_units = compute_units
        self.provider = provider

    def info(self) -> RuntimeInfo:
        return RuntimeInfo(
            runtime=self.runtime_name,
            runtime_version=ct.__version__,
            framework="coremltools",
            framework_version=ct.__version__,
            provider=self.provider,
            device="neural_engine" if self.compute_units == "neural_engine" else "cpu",
            env_fingerprint=env_fingerprint(),
            extra={"compute_units": str(self.compute_units)},
        )

    def load(self, path: str):
        kw = {}
        if self.compute_units:
            cu = getattr(ct, f"ComputeUnit.{self.compute_units}", None)
        model = ct.models.MLModel(path)
        return model

    def run(self, model: Any, feeds: Dict[str, np.ndarray],
            output_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run a CoreML model. ``model`` is a ct.models.MLModel.

        Feeds map CoreML input names -> numpy arrays (matching model input
        names). CoreML returns a dict of output name -> numpy/pyobj.
        """
        # CoreML expects inputs keyed by the model's declared input names.
        feed = {}
        in_spec = model.get_spec().description.input
        declared = [i.name for i in in_spec]
        # Remap caller feeds to declared names (positional fallback).
        src = list(feeds.items())
        if any(k in declared for k in feeds):
            src = list(feeds.items())
        else:
            src = list(zip(declared, [v for _, v in list(feeds.items())[:len(declared)]]))
        for k, v in src:
            if k in declared:
                feed[k] = v
        if not feed:
            raise KeyError(f"CoreML model has no usable input; inputs={declared} feed={list(feeds)}")
        pred = model.predict(feed)  # compute units fixed at compile time (CPU_ONLY)
        # Normalize all outputs
        return pred

    def run_and_normalize(self, model: Any, feeds: Dict[str, np.ndarray],
                          output_names: Optional[List[str]] = None) -> Dict[str, Any]:
        raw = self.run(model, feeds, output_names)
        # CoreML may return np arrays, scalars, or lists
        result = {}
        out_spec = model.get_spec().description.output
        out_names = [o.name for o in out_spec]
        for i, name in enumerate(out_names):
            if name in raw:
                val = raw[name]
                if isinstance(val, (list, tuple)) and not isinstance(val, (np.ndarray, np.generic)):
                    # multi-channel box output -> treat as tensor of arrays; represent
                    # as combined array when shapes match
                    result[name] = classify_coreml_value(val, name)
                else:
                    result[name] = classify_coreml_value(val, name)
        return result


def classify_coreml_value(val, name):
    """Turn a CoreML output value into an Output object.

    CoreML detection outputs often arrive as a list of np arrays
    (e.g. [boxes_xywh, scores, class_ids]). We emit a `boxes` kind output when
    we can identify triples, else keep tensors.
    """
    from cvconform.schema import Output

    # list of arrays with detectable box-like structure
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        try:
            arrays = [np.asarray(v, dtype=np.float32) for v in val]
            if all(a.size > 0 for a in arrays):
                # Heuristic: detect [boxes, scores, classes] pattern
                if arrays[0].ndim == 2 and arrays[0].shape[1] in (4, 5):
                    boxes_arr = arrays[0]
                    boxes = []
                    for row in boxes_arr:
                        c = row[:4].tolist()
                        box_class = int(row[4]) if row.shape[0] >= 5 else 0
                        score = float(row[4]) if row.shape[0] == 5 else 1.0
                        boxes.append(Box(coords=c, class_id=box_class, score=score))
                    return Output.from_boxes(boxes)
            return Output.from_tensor(np.concatenate([a.reshape(-1) for a in arrays]))
        except Exception:
            pass
    try:
        arr = to_numpy(val)
        return Output.from_tensor(arr)
    except Exception:
        return Output.from_values([val], kind="tensor", name=name)
