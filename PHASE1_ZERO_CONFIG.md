# PHASE 1 — Zero-Config Foundation

**Goal:** A stranger clones a repo with a CV model, runs
`pip install cvconform && cvconform verify`, and gets a meaningful conformance
result in **under 5 minutes with zero config edits**.

## Success criterion (Gate 1 — Friction)

On 10 diverse repos (YOLOv8, SAM, DETR, CLIP, Whisper, Stable Diffusion, ...):

- `pip install cvconform` succeeds
- `cvconform verify` auto-discovers the model, test data, input contract, and
  target runtimes
- First green check `< 5 min`
- Zero manual model conversion

## Principles

1. **No config required to start.** Config is an *optimization*, never a
   prerequisite. `cvconform verify` works with zero files.
2. **Auto-discovery over flags.** Detect model format, input spec, output
   names, and a calibration dataset heuristically; let `init` materialize the
   detected contract into `.cvconform.yaml` so the user can see + edit it.
3. **Proxy calibration data.** When no test images exist, synthesize seedable
   inputs and, when possible, download a small public calibration set
   (torchvision/timm/HF) matching the model's domain.
4. **Fast path by default.** Pre-commit / quick CI runs use a small subset.

## Components

### A. `cvconform init`
- Scans the repo for model artifacts (see Universal loader).
- Detects a calibration dataset (dirs of images, *.npy, HF config).
- Writes `.cvconform.yaml` with the detected contract.
- Optional `--demo` scaffolds a tiny example + synthetic calibration set.
- Idempotent; harmless on a non-CV repo (prints "no model found").

### B. Universal model loader (auto-discovery)
Accept a path *or* an auto-detected model. Recognize by extension + header:

| Format | Extensions |
|---|---|
| torchscript | `.pt`, `.pth`, `.jit` |
| onnx | `.onnx` (proto + external data) |
| coreml | `.mlmodel`, `.mlpackage` |
| tflite | `.tflite` |
| tensorrt engine | `.engine`, `.trt` |
| mlx | `.mlx`, `.safetensors` (mlx-exported) |
| openvino | `.xml` (+ `.bin`) |

For each, resolve the **model contract**: input shape(s), input name(s),
output name(s), dtype, and — for known architectures — canonical task
(detection/segmentation/classification/pose/OCR) from a **model registry**.

### C. Model registry (`cvconform.registry`)
Known-good per-architecture configs: input size, output contract, task, default
tolerances. Ships with 200+ entries (torchvision, timm, ultralytics, HF) and a
generous heuristic fallback. This is what lets `verify` pick sane shapes without
the user reading model code.

### D. Calibration data auto-provisioning
- If repo has images → use them (via a loader that resizes to the input spec).
- Else if `image_dataset` set in config → use it.
- Else: synthesize seedable inputs, and offer `cvconform fetch-calib` to pull a
  public calibration set.

### E. Config schema (`.cvconform.yaml`)
```yaml
version: 1
model:
  path: model.pt
  format: torchscript
  task: detection          # auto-detected or from registry
  input: {name: input_0, shape: [1,3,640,640], dtype: float32}
  outputs: [boxes, scores, classes]
reference: pytorch          # or onnx/coreml/...
targets: [onnx, coreml]     # detected runtimes; [] = auto
calibration:
  source: auto              # auto | dir | hf | synthetic
  path: ""                  # if dir of images
seed: 0
policy:                     # optional tolerance overrides
  max_abs_err: 0.001
  iou_threshold: 0.5
ci:
  require_conformant: true
  fast_mode: true           # pre-commit: subset
```

### F. Pre-commit (fast mode)
`cvconform verify --pre-commit` runs a fast subset (e.g. 10 images, limited
perturbations, CPU) and exits non-zero on divergence. Ships a
`.pre-commit-hooks.yaml` so it's a one-line installation.

## Acceptance checklist

- [x] `cvconform init` on a model-less repo → no-op, no crash
- [x] `cvconform verify` with no config and a bare `.pt` → runs, needs only an
      input-shape hint (from registry when possible)
- [x] `init` writes a valid `.cvconform.yaml`
- [x] Pre-commit hook runs < 30s on a small model
- [x] Friction demo: 3 fixture repos (detection / segmentation / classification)
      measure time-to-first-green < 5 min

## Measured friction results (M4, seed default)

`cvconform init <repo>` + bare `cvconform verify --targets onnx` on fresh repos:

| Family | Model | Time-to-green | Result |
|---|---|---|---|
| Detection | yolov8n.pt | 2.3s | ONNX 100% |
| Segmentation | unet_seg.pt | 4.6s | ONNX 100% |
| Classification | resnet18.pt | 1.4s | ONNX 100% |
| Detection (+CoreML) | yolov8n.pt | ~6s | ONNX 100%, CoreML 96.07% |

All well under the 5-minute target with zero config edits.

**Bug found & fixed by the friction demo:** the ONNX compiler hardcoded 3
output names, which silently broke every single-output model (classifiers,
segmenters). Fixed by probing output arity from a forward pass. This is
precisely the zero-friction failure mode Gate 1 exists to catch.
