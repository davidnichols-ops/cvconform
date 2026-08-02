"""High-level verify() — the product's public entry point.

``verify`` models the real deployment pipeline. A *source* artifact (usually a
PyTorch model) is compiled to each target format (ONNX, CoreML, ...) and every
format is executed as a differential target, then compared against a reference
runtime. Root-cause analysis explains any divergence.

The compilation step is explicit and recorded: it is exactly where silent
behavioral changes are introduced (export, quantization, operator fusion), so
cvconform treats it as part of the evidence, not as magic.

Python API:
    from cvconform import verify
    result = verify(model="model.pt",
                    reference="pytorch",
                    targets=["onnx", "coreml"],
                    seed=0)

CLI:
    cvconform verify model.onnx --reference onnx --targets onnxruntime,coreml
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cvconform.engines.differential import DifferentialEngine, generate_image_inputs
from cvconform.engines.comparison import TolerancePolicy
from cvconform.engines.analyze import analyze
from cvconform.report import render_human

# (runtime key, module, class, [compile helper]) — auto-registered real runtimes.
# The compile helper turns the source artifact into this runtime's artifact.
_AUTO_REAL_RUNTIMES = [
    ("pytorch", "cvconform.runtimes.pytorch_rt", "PyTorchRuntime", None),
    ("onnx", "cvconform.runtimes.onnx_rt", "OnnxRuntime", "compile_onnx"),
    ("coreml", "cvconform.runtimes.coreml_rt", "CoreMLRuntime", "compile_coreml"),
]


def _import_obj(module_name: str, name: str):
    import importlib

    return getattr(importlib.import_module(module_name), name)


SOURCE_KIND_BY_EXT = {
    ".pt": "pytorch", ".pth": "pytorch", ".jit": "pytorch",
    ".onnx": "onnx",
    ".mlmodel": "coreml", ".mlpackage": "coreml",
}

# autodetect format string -> verify source_kind
FORMAT_TO_SOURCE_KIND = {
    "torchscript": "pytorch",
    "onnx": "onnx",
    "coreml": "coreml",
    "tflite": "tflite",
    "tensorrt": "tensorrt",
    "mlx": "mlx",
    "openvino": "openvino",
}


def _resolve_runtime(name: str, **kwargs):
    for key, mod, cls, _ in _AUTO_REAL_RUNTIMES:
        if name == key:
            try:
                return _import_obj(mod, cls)(**kwargs)
            except ImportError:
                return None
    return None


# --------------------------------------------------------------------------
# Source artifact loading
# --------------------------------------------------------------------------

def _load_source(model_path: str, source_kind: str):
    if source_kind == "pytorch":
        import torch

        obj = torch.load(model_path, map_location="cpu", weights_only=False)
        # Accept both a raw module and a state-dict-in-module pattern.
        if hasattr(obj, "eval"):
            return obj
        # If it's a dict with 'state_dict', minimal: assume a module was saved
        # via torch.save(module) -> already handled; else error.
        raise TypeError("torch artifact is neither a module nor recognized")
    if source_kind == "onnx":
        import onnx

        return onnx.load(model_path)  # ModelProto (inlines external data on serialize)
    if source_kind == "coreml":
        import coremltools as ct

        return ct.models.MLModel(model_path)
    raise ValueError(f"unknown source format {source_kind!r}")


def _source_io_names(source_kind: str, model_path: str):
    if source_kind == "onnx":
        import onnx

        m = onnx.load(model_path)
        return [i.name for i in m.graph.input], [o.name for o in m.graph.output]
    if source_kind == "coreml":
        import coremltools as ct

        m = ct.models.MLModel(model_path)
        s = m.get_spec()
        return [i.name for i in s.description.input], [o.name for o in s.description.output]
    # pytorch: positional inputs, output names from a forward pass convention
    return ["input_0"], None


def _input_shape_for(source_kind: str, model_path: str) -> Tuple[int, ...]:
    if source_kind == "onnx":
        import onnx

        m = onnx.load(model_path)
        inp = m.graph.input[0]
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            dims.append(int(d.dim_value) if d.HasField("dim_value") else 1)
        return tuple(dims)
    if source_kind == "coreml":
        import coremltools as ct

        m = ct.models.MLModel(model_path)
        inp = m.get_spec().description.input[0]
        if inp.type.HasField("imageType"):
            return (1, 3, inp.type.imageType.height, inp.type.imageType.width)
        if inp.type.HasField("multiArrayType"):
            return tuple([1] + list(inp.type.multiArrayType.shape))
        return (1, 3, 224, 224)
    return (1, 3, 224, 224)


# --------------------------------------------------------------------------
# Per-target compilation + loading
# --------------------------------------------------------------------------

def _target_artifact(target: str, source_kind: str, source_artifact, model_path: str,
                     input_shape: Tuple[int, ...]):
    """Compile the source artifact into the target runtime's artifact.

    Returns (runtime_artifact, compile_record).
    """
    # Case A: target format equals source format -> direct reuse.
    if target == source_kind:
        return source_artifact, {"method": "direct", "from": source_kind}

    # Case B: compile via the registered helper.
    if target == "onnx":
        onnx_bytes = _compile_onnx(source_kind, source_artifact, input_shape, model_path)
        return onnx_bytes, {"method": "export_onnx", "from": source_kind}
    if target == "coreml":
        mlmodel = _compile_coreml(source_kind, source_artifact, input_shape, model_path)
        return mlmodel, {"method": "convert_coreml", "from": source_kind}
    # Case C: unsupported path.
    raise ValueError(f"cannot compile source {source_kind} to target {target}")


def _compile_onnx(source_kind, source_artifact, input_shape, model_path) -> bytes:
    """Export a torch source to self-contained ONNX bytes. (Other sources pass-through.)"""
    if source_kind == "onnx":
        return source_artifact.SerializeToString()
    if source_kind == "pytorch":
        import torch

        model = source_artifact
        model.eval()
        dummy = torch.randn(*input_shape)
        # Probe output arity so we never hardcode the wrong number of outputs
        # (detectors emit 3, classifiers/segmenters emit 1).
        import io

        with torch.no_grad():
            probe = model(dummy)
        n_out = len(probe) if isinstance(probe, (tuple, list)) else 1
        out_names = [f"output_{i}" for i in range(n_out)]
        buf = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        buf.close()
        try:
            with torch.no_grad():
                # Legacy TorchScript exporter: handles nn.Module AND ScriptModule,
                # and emits named outputs for clean named comparisons.
                torch.onnx.export(
                    model, (dummy,), buf.name,
                    input_names=["input"],
                    output_names=out_names,
                    opset_version=17, dynamic_axes=None, dynamo=False,
                )
            import onnx

            return onnx.load(buf.name).SerializeToString()
        finally:
            os.unlink(buf.name)
    raise ValueError(f"cannot export {source_kind} to onnx")


def _compile_coreml(source_kind, source_artifact, input_shape, model_path):
    """Convert a source to a CoreML model."""
    if source_kind == "coreml":
        return source_artifact
    if source_kind == "onnx":
        import coremltools as ct

        return ct.convert(model_path, source="onnx", convert_to="mlprogram",
                          compute_units=ct.ComputeUnit.CPU_ONLY)
    if source_kind == "pytorch":
        import torch

        import coremltools as ct

        # Trace (not script) to get a pure data-flow graph — scripting can
        # introduce prim::If control flow that coremltools cannot convert.
        traced = torch.jit.trace(source_artifact, torch.randn(*input_shape))
        ml = ct.convert(
            traced,
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.CPU_ONLY,
            minimum_deployment_target=ct.target.macOS13,
            inputs=[ct.TensorType(name="input", shape=input_shape, dtype=np.float32)],
        )
        return ml
    raise ValueError(f"cannot convert {source_kind} to coreml")


def _source_output_names(source_kind, source_artifact, model_path, input_shape):
    """Best-effort output names for the source (used to align comparisons)."""
    if source_kind == "onnx":
        return [o.name for o in source_artifact.graph.output]
    if source_kind == "coreml":
        return [o.name for o in source_artifact.get_spec().description.output]
    return ["output_0", "output_1", "output_2"]  # pytorch: positional heuristic


# --------------------------------------------------------------------------
# verify()
# --------------------------------------------------------------------------

def verify(
    model: str,
    reference: str = "pytorch",
    targets: Optional[List[str]] = None,
    dataset: str = "synthetic",
    seed: int = 0,
    num_samples: int = 1,
    input_shape: Optional[Tuple[int, ...]] = None,
    policy: Optional[TolerancePolicy] = None,
    output_names: Optional[List[str]] = None,
    source_kind: Optional[str] = None,
    input_names: Optional[List[str]] = None,
    reference_path: Optional[str] = None,
    target_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Run differential conformance verification on a model.

    Args:
        model: path to the *source* artifact (pytorch .pt / .pth, or .onnx).
        reference: runtime to treat as ground truth.
        targets: runtimes to verify. Default: all real runtimes except reference.
        dataset: 'synthetic' (seeded) or a dir of .npy samples.
        seed, num_samples: seeded synthetic input generation.
        input_shape: override input tensor shape (default inferred).
        policy: conformance tolerance policy.
        output_names: which semantic outputs to compare (default: all).
        source_kind: force source format (optionally with a pre-built
            ``target_paths`` mapping if artifacts already exist on disk).
        reference_path / target_paths[: explicitly supplied compiled artifacts.
    """
    if not os.path.exists(model):
        raise FileNotFoundError(f"model not found: {model}")

    source_kind = source_kind or SOURCE_KIND_BY_EXT.get(os.path.splitext(model)[1].lower())
    # autodetect may pass a format string like 'torchscript'
    if source_kind in FORMAT_TO_SOURCE_KIND:
        source_kind = FORMAT_TO_SOURCE_KIND[source_kind]
    if source_kind is None:
        raise ValueError(f"cannot detect source format from {model}")

    if targets is None:
        targets = [k for k, _, _, _ in _AUTO_REAL_RUNTIMES if k != reference]

    ref_rt = _resolve_runtime(reference)
    if ref_rt is None:
        raise ValueError(f"reference runtime {reference!r} not installed/supported")

    # ---- input generation ------------------------------------------------
    if input_shape is None:
        input_shape = _input_shape_for(source_kind, model)
    if dataset == "synthetic":
        imgs = generate_image_inputs(input_shape, seed, n=num_samples, kind="noise")
    else:
        imgs = _load_dataset_samples(dataset, num_samples, input_shape, seed)

    # ---- source + reference artifact ------------------------------------
    source_artifact = _load_source(model, source_kind)
    ref_artifact = reference_path or source_artifact
    if isinstance(ref_artifact, str):
        ref_artifact = _load_source(ref_artifact, source_kind)

    _, source_out = _source_io_names(source_kind, model)
    output_names = output_names or source_out or _source_output_names(
        source_kind, source_artifact, model, input_shape)
    if reference == "pytorch" and isinstance(ref_artifact, bytes):
        # reference onnx -> load as module? only when source is onnx; keep bytes
        pass
    ref_input = input_names or ["input_0"]

    # ---- differential over samples --------------------------------------
    all_results: Dict[str, Any] = {}
    compile_records: Dict[str, Dict] = {}

    for target in targets:
        t_rt = _resolve_runtime(target)
        if t_rt is None:
            all_results[target] = _target_result_stub(target, "runtime not installed", ref_rt)
            continue
        # Resolve or compile the target artifact.
        if target_paths and target in target_paths:
            t_artifact = _target_artifact_from_path(target_paths[target], target)
            compile_records[target] = {"method": "provided", "path": target_paths[target]}
        else:
            t_artifact, crec = _target_artifact(target, source_kind, source_artifact,
                                                model, input_shape)
            compile_records[target] = crec

        eng = DifferentialEngine(ref_rt, ref_artifact, output_names or [], policy=policy)
        eng.add_target(target, t_rt, t_artifact)

        sample_results = []
        for i in range(num_samples):
            feed = {nm: np.asarray(imgs[i]) for nm in (ref_input or ["input_0"])}
            sample_results.append(eng.run(feed))
        all_results[target] = _aggregate_target(target, sample_results, ref_rt)

    # ---- environment + analysis -----------------------------------------
    env = ref_rt.info().to_dict()
    findings_map = _build_findings(all_results, source_kind, model)
    research = _research_findings(findings_map)

    report = {
        "model": os.path.basename(model),
        "source": source_kind,
        "reference": reference,
        "dataset": dataset,
        "seed": seed,
        "input_shape": list(input_shape),
        "environment": env,
        "compilation": compile_records,
        "targets": {t: {"overall_score": r["overall_score"], "is_conformant": r["is_conformant"],
                        "scores": r["scores"], "divergences": r["divergences"]}
                    for t, r in all_results.items()},
        "findings": findings_map,
        "research": research,
        "target_status": {t: ("conformant" if r["is_conformant"] else
                              "degraded" if r["overall_score"] > 0 else "error")
                          for t, r in all_results.items()},
    }
    return report


def _research_findings(findings_map: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pass each finding through the AI Research Agent to add an investigation."""
    from cvconform.research import ResearchAgent

    agent = ResearchAgent(offline_ok=True)
    out = []
    for f in findings_map:
        if not f:
            continue
        try:
            s = agent.investigate(f)
            out.append(s.to_dict())
        except Exception:  # noqa: BLE001
            out.append(None)
    return [o for o in out if o]


def _target_artifact_from_path(path, target):
    if target == "onnx":
        return _load_source(path, "onnx").SerializeToString()
    if target == "coreml":
        return _load_source(path, "coreml")
    return _load_source(path, target)


def _target_result_stub(target, msg, ref_rt):
    return {
        "overall_score": 0.0,
        "is_conformant": False,
        "scores": {},
        "divergences": [{
            "output": "<model>", "kind": "runtime", "metric": "execution_failed",
            "observed": 1.0, "threshold": 0.0, "magnitude": 1.0,
            "first_divergence": "", "mechanism": msg, "confidence": 0.0,
        }],
    }


def _aggregate_target(target, sample_results, ref_rt):
    if not sample_results:
        return _target_result_stub(target, "no samples", ref_rt)
    all_divs = []
    score_sums = {}
    score_counts = {}
    for sr in sample_results:
        res = sr.get(target)
        if res is None:
            continue
        all_divs.extend(res.divergences)
        for out, score in res.scores.items():
            score_sums[out] = score_sums.get(out, 0.0) + score
            score_counts[out] = score_counts.get(out, 0) + 1
    avg_scores = {k: (v / max(1, score_counts[k])) for k, v in score_sums.items()}
    overall = float(np.mean(list(avg_scores.values()))) if avg_scores else 0.0
    is_conformant = overall >= 95.0 and not all_divs
    seen = set()
    deduped = []
    for d in all_divs:
        key = (d.output, d.metric, round(d.magnitude, 4))
        if key not in seen:
            seen.add(key)
            deduped.append(d)
    return {
        "overall_score": overall,
        "is_conformant": is_conformant,
        "scores": {k: round(v, 2) for k, v in avg_scores.items()},
        "divergences": [d.to_dict() for d in deduped],
    }


def _build_findings(all_results, source_kind, model_path):
    from cvconform.engines.analyze import analyze as analyze_fn
    from cvconform.engines.comparison import ComparisonResult, Divergence

    findings = []
    for target, r in all_results.items():
        divs = [Divergence(**{k: v for k, v in d.items()}) for d in r.get("divergences", [])]
        res = ComparisonResult(
            target=target, reference="reference", per_output={}, divergences=divs,
            scores=r.get("scores", {}), overall_score=r.get("overall_score", 100.0),
            is_conformant=r.get("is_conformant", True),
        )
        graph = None
        if source_kind == "onnx":
            try:
                from cvconform.loaders.onnx_loader import load_onnx
                graph = load_onnx(model_path)
            except Exception:
                graph = None
        evs = analyze_fn(graph, {target: res})
        for e in evs.get(target, []):
            findings.append(e.to_dict())
    return findings


def _load_dataset_samples(dirpath, num_samples, shape, seed):
    import glob

    files = sorted(glob.glob(os.path.join(dirpath, "*.npy")))[:num_samples]
    if not files:
        return generate_image_inputs(shape, seed, n=num_samples, kind="noise")
    return [np.load(f) for f in files]


def verify_and_render(model, **kwargs) -> str:
    return render_human(verify(model, **kwargs))


def verify_to_json(model, path: str, **kwargs) -> Dict[str, Any]:
    report = verify(model, **kwargs)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return report
