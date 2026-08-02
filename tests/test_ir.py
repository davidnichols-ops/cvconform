"""Unit tests for the VisionGraph IR + ONNX loader."""

from __future__ import annotations

import os

import pytest


@pytest.fixture
def onnx_blob():
    """A small ONNX model exported from the demo model (bytes)."""
    out_dir = os.path.join(os.path.dirname(__file__), "..", "examples_output")
    path = os.path.join(out_dir, "YOLO11_demo.onnx")
    if not os.path.exists(path):
        pytest.skip("demo onnx not generated; run examples/demo_model.py")
    with open(path, "rb") as f:
        return f.read()


def test_onnx_loader_builds_graph(onnx_blob):
    import onnx

    from cvconform.loaders.onnx_loader import from_onnx_model

    model = onnx.ModelProto()
    model.ParseFromString(onnx_blob)
    g = from_onnx_model(model)
    assert g.ops, "graph should contain ops"
    assert g.inputs, "graph should have inputs"
    assert g.outputs, "graph should have outputs"
    # topological order must be valid
    order = g.topological_order()
    assert len(order) == len(g.ops)


def test_onnx_graph_has_detection_head(onnx_blob):
    import onnx

    from cvconform.loaders.onnx_loader import from_onnx_model

    model = onnx.ModelProto()
    model.ParseFromString(onnx_blob)
    g = from_onnx_model(model)
    ops = {op.op for op in g.ops.values()}
    assert "cv.Conv2D" in ops, "conv backbone expected"
    # schema hash should be stable
    h1 = g.schema_hash()
    h2 = g.schema_hash()
    assert h1 == h2


def test_tensor_spec_precision():
    from cvconform.ir import TensorSpec, Precision

    s = TensorSpec(shape=(1, 3, 224, 224), dtype="float32")
    assert s.precision == Precision.FP32
    assert s.is_quantized() is False
    assert "NCHW" in s.desc()
    assert s.rank() == 4


def test_graph_schema_hash_changes_on_structure(onnx_blob):
    """A graph with a different op set must hash differently (sensitivity)."""
    import onnx

    from cvconform.loaders.onnx_loader import from_onnx_model
    from cvconform.ir import Operator, OperatorRegistry, TensorSpec

    model = onnx.ModelProto()
    model.ParseFromString(onnx_blob)
    g = from_onnx_model(model)
    base = g.schema_hash()
    g2 = from_onnx_model(model)
    # add a harmless extra op -> hash must differ
    g2.add_op(Operator(op=OperatorRegistry.IDENTITY, name="extra_identity",
                       inputs=[("x", TensorSpec(shape=(1,), dtype="float32"))],
                       outputs=[("y", TensorSpec(shape=(1,), dtype="float32"))]))
    assert g2.schema_hash() != base
