"""Integration tests — the full product surface, end to end.

These exercise the public API + CLI as a user would, and seed the Regression
Memory corpus from a real fault injection (proving the moat loop works).
Marked so they can be deselected in constrained CI (they need torch + onnx).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from cvconform import verify
from cvconform.corpus import ConformanceCorpus

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")
OUT = os.path.join(os.path.dirname(__file__), "..", "examples_output")
SOURCE = os.path.join(OUT, "YOLO11_demo_traced.pt")

pytestmark = pytest.mark.skipif(
    not os.path.exists(SOURCE),
    reason="demo artifact not built; run examples/demo_model.py",
)


def test_api_verify_returns_report():
    report = verify(model=SOURCE, reference="pytorch", targets=["onnx"],
                    seed=0, num_samples=1,
                    output_names=["output_0", "output_1", "output_2"])
    assert "targets" in report
    assert "onnx" in report["targets"]
    assert isinstance(report["targets"]["onnx"]["overall_score"], float)
    assert "target_status" in report
    assert "environment" in report


def test_api_verify_is_reproducible():
    r1 = verify(model=SOURCE, reference="pytorch", targets=["onnx"],
                seed=42, num_samples=2,
                output_names=["output_0", "output_1", "output_2"])
    r2 = verify(model=SOURCE, reference="pytorch", targets=["onnx"],
                seed=42, num_samples=2,
                output_names=["output_0", "output_1", "output_2"])
    assert r1["targets"]["onnx"]["overall_score"] == \
        r2["targets"]["onnx"]["overall_score"]


def test_cli_verify_emits_json(tmp_path):
    out = tmp_path / "report.json"
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")
    subprocess.run(
        [sys.executable, "-m", "cvconform.cli", "verify", SOURCE,
         "--reference", "pytorch", "--targets", "onnx",
         "--json", str(out)],
        capture_output=True, env=env, check=False,
    )
    assert out.exists()
    data = json.loads(out.read_text())
    assert "targets" in data


def test_seed_corpus_from_real_fault():
    """Record a real detected failure into the corpus (the moat loop)."""
    import tempfile

    sys.path.insert(0, EXAMPLES)
    from demo_model import build_demo_model
    from cvconform.faults import FaultInjector
    from cvconform.engines.differential import DifferentialEngine
    from cvconform.engines.differential import generate_image_inputs
    from cvconform.runtimes.pytorch_rt import PyTorchRuntime

    clean = build_demo_model()
    faulty = build_demo_model()
    faulty.load_state_dict(clean.state_dict())
    FaultInjector(faulty, seed=1).apply("precision_fp16")

    pet = PyTorchRuntime()
    eng = DifferentialEngine(pet, clean, ["output_0", "output_1", "output_2"])
    eng.add_target("pytorch_fp16", pet, faulty)
    img = generate_image_inputs((1, 3, 224, 224), 0, n=1, kind="noise")[0]
    res = eng.run({"input_0": img})["pytorch_fp16"]

    assert res.overall_score < 95.0, "fp16 fault must be detected"

    # seed the corpus
    tmp = tempfile.mkdtemp()
    corpus = ConformanceCorpus(root=tmp)
    corpus.record_failure("pytorch", "fp16_accumulation", {
        "backend": "pytorch", "fault": "precision_fp16",
        "signal": {"overall_score": res.overall_score,
                   "divergences": [d.to_dict() for d in res.divergences]},
    })
    entries = corpus.list_failures("pytorch")
    assert len(entries) == 1
    assert entries[0]["fault"] == "precision_fp16"
