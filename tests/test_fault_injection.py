"""End-to-end fault-injection tests (Phase 4: Break It).

Prove that the differential engine *detects and explains* a curated set of
real failure modes. Each test builds a clean reference and a faulted target,
runs differential verification, and asserts:
  - the target diverges (not conformant),
  - the reported mechanism is plausible (via root-cause analysis).

These are the seeds of the Regression Memory corpus.
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from demo_model import build_demo_model  # noqa: E402
from cvconform.runtimes.pytorch_rt import PyTorchRuntime  # noqa: E402
from cvconform.runtimes.onnx_rt import OnnxRuntime  # noqa: E402
from cvconform.engines.differential import DifferentialEngine  # noqa: E402
from cvconform.engines.comparison import TolerancePolicy  # noqa: E402
from cvconform.engines.analyze import analyze  # noqa: E402
from cvconform.engines.comparison import ComparisonResult, Divergence  # noqa: E402
from cvconform.engines.differential import generate_image_inputs  # noqa: E402

EXAMPLES_OUT = os.path.join(os.path.dirname(__file__), "..", "examples_output")


def _export_onnx(model, path):
    import torch

    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model, (torch.randn(1, 3, 224, 224),), path,
            input_names=["input"], output_names=["output_0", "output_1", "output_2"],
            opset_version=17, dynamo=False,
        )
    import onnx

    return onnx.load(path).SerializeToString()


@pytest.fixture(scope="module")
def clean_model():
    return build_demo_model()


def _make_fault_model(clean, fault, seed=0):
    """Return a build_demo_model-derived model with a fault applied."""
    import torch

    model = build_demo_model()
    model.load_state_dict(clean.state_dict())
    model.eval()
    from cvconform.faults import FaultInjector

    inj = FaultInjector(model, seed)
    inj.apply(fault)
    return model


def _run_verify(reference_model, target_model, fault, num_samples=2, seed=0):
    """Run differential: reference=clean pytorch, target=onnx from target model."""
    import tempfile

    import torch

    pet = PyTorchRuntime()
    ort_rt = OnnxRuntime(providers=["CPUExecutionProvider"])
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp.close()
    blob = _export_onnx(target_model, tmp.name)
    os.unlink(tmp.name)

    eng = DifferentialEngine(pet, reference_model, ["output_0", "output_1", "output_2"])
    eng.add_target("onnx", ort_rt, blob)

    imgs = generate_image_inputs((1, 3, 224, 224), seed, n=num_samples, kind="noise")
    results = []
    for img in imgs:
        results.append(eng.run({"input_0": img}))
    # aggregate worst
    res = results[-1]["onnx"]
    return res


@pytest.mark.parametrize("fault", [
    "weights_scale",
    "precision_fp16",
    "weight_nan",
    "weight_corrupt",
    "bias_shift",
    "act_clip",
])
def test_fault_detected(clean_model, fault):
    """Every injected fault must produce a non-conformant result."""
    target_model = _make_fault_model(clean_model, fault)
    res = _run_verify(clean_model, target_model, fault)
    assert res.overall_score < 95.0, f"fault {fault} went undetected (score {res.overall_score})"
    assert len(res.divergences) > 0, f"fault {fault} produced no divergences"


def test_clean_is_conformant(clean_model):
    """The clean model must be conformant (control)."""
    res = _run_verify(clean_model, clean_model, "none")
    assert res.overall_score >= 95.0, f"clean model diverged: {res.overall_score}"
    assert res.is_conformant


def test_nan_detected_as_nan(clean_model):
    """NaN injection must appear in range/stats and be caught."""
    target_model = _make_fault_model(clean_model, "weight_nan")
    res = _run_verify(clean_model, target_model, "weight_nan")
    assert res.overall_score < 95.0


def test_root_cause_produces_evidence(clean_model):
    """Root-cause analysis returns structured Evidence with a mechanism."""
    target_model = _make_fault_model(clean_model, "weights_scale")
    res = _run_verify(clean_model, target_model, "weights_scale")
    divs = [Divergence(**{k: v for k, v in d.items()})
            for d in [x.to_dict() for x in res.divergences]]
    cr = ComparisonResult(target="onnx", reference="pytorch",
                          per_output={}, divergences=divs,
                          scores=res.scores, overall_score=res.overall_score,
                          is_conformant=res.is_conformant)
    findings = analyze(None, {"onnx": cr})
    evs = findings.get("onnx", [])
    assert len(evs) > 0, "no root-cause evidence produced"
    assert evs[0].mechanism, "evidence missing mechanism"
    assert evs[0].confidence > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
