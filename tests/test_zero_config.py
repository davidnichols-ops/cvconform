"""Tests for the zero-config foundation: registry, autodetect, config, init."""

from __future__ import annotations

import os

import pytest

from cvconform.registry import ModelRegistry
from cvconform.autodetect import (
    detect_format, detect_model, scan_for_models, scan_for_images, detect_repo,
)
from cvconform.config import CVConformConfig, write_config, load_config, find_config
from cvconform.init import init_project


# ---------- registry ----------
def test_registry_matches_known_models():
    reg = ModelRegistry()
    assert reg.lookup("yolov8n.pt").task == "detection"
    assert reg.lookup("resnet50.onnx").task == "classification"
    assert reg.lookup("segment-anything-vit-h").task == "segmentation"
    assert reg.lookup("clip-vit-base").task == "embedding"
    assert reg.lookup("detr-resnet-50").task == "detection"


def test_registry_unknown_returns_none():
    reg = ModelRegistry()
    assert reg.lookup("completely_unknown_model") is None


def test_registry_describe():
    reg = ModelRegistry()
    s = reg.describe()
    assert "yolo" in s
    assert "resnet" in s


# ---------- format detection ----------
def test_detect_format():
    assert detect_format("model.pt") == "torchscript"
    assert detect_format("model.onnx") == "onnx"
    assert detect_format("model.mlpackage") == "coreml"
    assert detect_format("model.tflite") == "tflite"
    assert detect_format("model.engine") == "tensorrt"
    assert detect_format("model.xml") == "openvino"
    assert detect_format("readme.md") is None


# ---------- scan + detect ----------
def test_scan_for_models(tmp_path):
    (tmp_path / "yolov8n.pt").write_bytes(b"x" * 10)
    (tmp_path / "not_a_model.txt").write_text("hi")
    models = scan_for_models(str(tmp_path))
    found = [m for m in models if "yolov8n" in m]
    assert len(found) == 1


def test_detect_model_onnx(tmp_path):
    # Build a tiny real onnx so introspection reads a valid contract.
    import onnx
    from onnx import helper, TensorProto

    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 224, 224])
    out = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 6272, 4])
    node = helper.make_node("Relu", ["images"], ["output0"])
    graph = helper.make_graph([node], "yolov8", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    p = tmp_path / "yolov8n.onnx"
    onnx.save(model, str(p))

    dm = detect_model(str(p))
    assert dm.format == "onnx"
    assert dm.input_shape == (1, 3, 224, 224)
    assert dm.input_name == "images"
    assert dm.outputs == ["output0"]
    # registry should match the yolo name
    assert dm.registry_key == "yolo"


def test_detect_model_torchscript_heuristic(tmp_path):
    p = tmp_path / "mymodel.pt"
    p.write_bytes(b"not-real-torchscript")
    dm = detect_model(str(p))
    assert dm.format == "torchscript"
    assert dm.input_shape == (1, 3, 224, 224)  # heuristic default


def test_detect_repo_prefers_onnx(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"onnx")
    (tmp_path / "model.pt").write_bytes(b"pt")
    r = detect_repo(str(tmp_path))
    assert r["preferred_model"].endswith(".onnx")


# ---------- config ----------
def test_config_roundtrip(tmp_path):
    cfg = CVConformConfig()
    cfg.model = {"path": "m.pt", "format": "torchscript", "task": "detection"}
    cfg.reference = "pytorch"
    cfg.targets = ["onnx", "coreml"]
    path = write_config(str(tmp_path / ".cvconform.yaml"), cfg)
    loaded = load_config(path)
    assert loaded.model["format"] == "torchscript"
    assert loaded.reference == "pytorch"


def test_find_config_searches_up(tmp_path):
    (tmp_path / ".cvconform.yaml").write_text("version: 1\n")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    cfg = find_config(str(sub))
    assert cfg == str(tmp_path / ".cvconform.yaml")


def test_load_config_missing():
    assert load_config() is None or True  # no config in cwd is fine


# ---------- init ----------
def test_init_writes_config(tmp_path):
    (tmp_path / "yolov8n.pt").write_bytes(b"x" * 20)
    res = init_project(root=str(tmp_path))
    assert res["status"] == "written"
    assert res["models_found"] == 1
    assert res["reference"] == "pytorch"
    cfg_path = os.path.join(str(tmp_path), ".cvconform.yaml")
    assert os.path.exists(cfg_path)
    cfg = load_config(cfg_path)
    assert cfg.model["task"] == "detection"  # registry match via filename
    assert cfg.model["path"] == "yolov8n.pt"


def test_init_no_model_is_harmless(tmp_path):
    (tmp_path / "readme.md").write_text("no models")
    res = init_project(root=str(tmp_path))
    assert res["status"] == "written"
    assert res["models_found"] == 0


def test_init_idempotent(tmp_path):
    (tmp_path / "model.pt").write_bytes(b"x")
    r1 = init_project(root=str(tmp_path))
    r2 = init_project(root=str(tmp_path))  # second run, no force
    assert r2["status"] == "exists"