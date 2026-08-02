"""Demo model: a tiny YOLO-style detector (conv backbone + dense detection head).

Not a demo stub — a real, executable model with genuinely interesting
properties for conformance testing:

- a conv backbone (with BatchNorm + SiLU activation fusion opportunity),
- a dense detection head emitting fixed-shape predictions.

Design decision: the model outputs **dense** head tensors — per-cell box
predictions, class logits, and objectness — NOT on-model NMS. This is
deliberate and matches how real deployments are built:

  1. The ops used (Conv, BatchNorm, SiLU, Permute, Reshape) export to ONNX
     *and* convert to CoreML cleanly — this is exactly where FP16 accumulation
     drift and operator-fusion differences appear on device.
  2. NMS / post-processing is data-dependent and notoriously
     implementation-defined; putting it on the model makes conformance
     testing fragile and conflates NMS differences with real numerical drift.

So the model's contract is:  [B, HW, 4] boxes + [B, HW, C] class logits +
[B, HW] objectness. The differential engine compares these dense tensors.
NMS, where needed, is a clearly-separated post-processing step the user owns.

Run:  python examples/demo_model.py   (writes examples_output/YOLO11_demo.onnx)
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyYoloBackbone(nn.Module):
    """A tiny conv backbone with BatchNorm + SiLU activation."""

    def __init__(self, in_channels=3, base=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(base)
        self.conv2 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(base * 2)
        self.conv3 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(base * 4)

    def forward(self, x):
        x = F.silu(self.bn1(self.conv1(x)))
        x = F.silu(self.bn2(self.conv2(x)))
        x = F.silu(self.bn3(self.conv3(x)))
        return x


class TinyYoloHead(nn.Module):
    """Dense detection head: per-cell boxes + class logits + objectness.

    Uses only CoreML-convertible ops (conv, permute, reshape) so the model
    round-trips through ONNX and CoreML for real differential testing.
    """

    def __init__(self, in_channels, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        self.box_conv = nn.Conv2d(in_channels, 4, 1)
        self.cls_conv = nn.Conv2d(in_channels, num_classes, 1)
        self.obj_conv = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x):
        # Static reshape: fixed input -> fixed feature map (224 -> 28x28 => 784 cells).
        boxes = self.box_conv(x).permute(0, 2, 3, 1).reshape(-1, 4)
        clss = self.cls_conv(x).permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        objs = self.obj_conv(x).permute(0, 2, 3, 1).reshape(-1)
        return boxes, clss, objs


class TinyYolo(nn.Module):
    def __init__(self, in_channels=3, base=16, num_classes=4):
        super().__init__()
        self.backbone = TinyYoloBackbone(in_channels, base)
        self.head = TinyYoloHead(4 * base, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)


def build_demo_model() -> TinyYolo:
    torch.manual_seed(42)
    return TinyYolo()


def export_onnx(model: TinyYolo, path: str, input_shape=(1, 3, 224, 224)):
    model.eval()
    dummy = torch.randn(*input_shape)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            path,
            input_names=["input"],
            output_names=["boxes", "classes", "objectness"],
            opset_version=17,
            dynamic_axes=None,
            dynamo=False,
        )
    return path


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "examples_output")
    os.makedirs(out_dir, exist_ok=True)
    # ONNX export
    path = os.path.join(out_dir, "YOLO11_demo.onnx")
    model = build_demo_model()
    export_onnx(model, path)
    print(f"Wrote {os.path.abspath(path)}")
    import onnx

    m = onnx.load(path)
    print("inputs:", [i.name for i in m.graph.input])
    print("outputs:", [o.name for o in m.graph.output])
    print("nodes:", len(m.graph.node))
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feed = {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)}
    out = sess.run(None, feed)
    print("ORT outputs:", [o.shape for o in out])

    # TorchScript save (self-contained, no class import needed on load)
    import torch

    sm = torch.jit.script(model)
    sm_path = os.path.join(out_dir, "YOLO11_demo_scripted.pt")
    torch.jit.save(sm, sm_path)
    print(f"Wrote {os.path.abspath(sm_path)}")

    # CoreML conversion check (CPU-only) + save
    try:
        import coremltools as ct

        ml = ct.convert(
            sm,
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.CPU_ONLY,
            minimum_deployment_target=ct.target.macOS13,
            inputs=[ct.TensorType(name="input", shape=(1, 3, 224, 224), dtype=np.float32)],
        )
        ml_path = os.path.join(out_dir, "YOLO11_demo.mlpackage")
        ml.save(ml_path)
        print(f"CoreML OK -> {ml_path}")
        print("outputs:", [o.name for o in ml.get_spec().description.output])
    except Exception as e:  # noqa: BLE001
        print("CoreML conversion failed:", type(e).__name__, str(e)[:200])
