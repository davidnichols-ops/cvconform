<p align="center">
  <strong>cvconform</strong><br/>
  <em>the correctness and reliability layer for computer vision</em>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey" alt="Platform"></a>
  <a href="#"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
</p>

---

Computer vision has a hidden reliability crisis. A model is trained once
(PyTorch, CUDA, FP32) and then silently transformed — ONNX, TensorRT, CoreML,
OpenVINO, TFLite, quantization, operator fusion. Every transformation creates
opportunities for the model to stop being itself.

The industry mostly asks **"Can this model run?"**

`cvconform` answers the question nobody was asking: **"Is this still the same
model?"** It is differential conformance verification for vision models —
execute the *same* model on *multiple* runtimes with the *same* seeded inputs,
compare the outputs, and when they differ, know *exactly why*.

## ✨ Zero config

`cvconform` works on a fresh clone with **no configuration**.

```bash
pip install cvconform
cd your-model-repo

cvconform init      # auto-detects your model + contract, writes .cvconform.yaml
cvconform verify    # auto-runs every installed runtime, prints a conform report
```

It auto-discovers model format (torchscript, onnx, coreml, tflite, tensorrt
engine, mlx, openvino), input shape, output names, and a calibration set — from
the artifact itself or a registry of 200+ known architectures, with a sensible
heuristic fallback.

<details>
<summary><strong>Friction demo — measured time to first green check</strong></summary>

`cvconform init` + bare `cvconform verify --targets onnx` on three fresh repos
(Apple M4):

| Family | Model | Time | Result |
|---|---|---|---|
| Detection | yolov8n.pt | 2.3s | ONNX 100% |
| Segmentation | unet_seg.pt | 4.6s | ONNX 100% |
| Classification | resnet18.pt | 1.4s | ONNX 100% |

</details>

## Install

```bash
pip install cvconform                # core: numpy, onnx, onnxruntime, pyyaml
pip install 'cvconform[pytorch]'     # PyTorch reference backend
pip install 'cvconform[coreml]'      # CoreML target (macOS)
pip install 'cvconform[openvino]'    # OpenVINO target
pip install 'cvconform[all]'         # every optional backend
```

> Requires Python 3.10+. On Apple Silicon you get CoreML + MLX paths for free.

## Usage

### Python API

```python
from cvconform import verify

result = verify(
    model="yolo11.pt",
    reference="pytorch",
    targets=["onnx", "coreml"],
    seed=0,
)
```

### CLI

```bash
# Clear a model — auto-discover everything else
cvconform verify

# Explicit
cvconform verify model.pt --reference pytorch --targets onnx,coreml

# Machine-readable report for CI/CD
cvconform verify model.pt --json report.json

# CI gate: non-zero exit if any runtime diverges
cvconform verify --require-conformant

# Pre-commit fast mode
cvconform verify --pre-commit

# Store findings in your regression-memory corpus
cvconform verify --corpus corpus/
```

### Output

```
VISION CONFORM REPORT
Model:       yolo11.pt
Reference:   pytorch

CONFORMANCE SCORES
  onnx            100.00%
  coreml           96.36%

FINDINGS: none — all targets conformant within policy.
```

When a runtime *does* diverge, `cvconform` tells you why — not "probably a
bug", but:

```
CRITICAL FINDING
Backend:       CoreML
Affected:      Conv+Activation fusion
Cause:         FP16 accumulation drift
Impact:        37 production images produce different classes
Confidence:    96%
Fixes:         1. Disable operator fusion
               2. Force FP32 accumulation
               3. Report upstream issue
```

## Extras

| Feature | How |
|---|---|
| **CI gate** | `--require-conformant` / `--pre-commit` exit non-zero on divergence |
| **Pre-commit hook** | one-line install — see `.pre-commit-hooks.yaml` |
| **GitHub Action** | composite action installs → verifies → comments a badge — see `.github/actions/verify` |
| **Regression memory** | every failure becomes a permanent, versioned corpus entry |
| **Failure discovery** | hunts for diverging inputs (noise, edges, illumination, blur, compression, extremes) |
| **Failure reduction** | shrinks a failing image to a minimal repro |

## The science behind it

Every result is **reproducible and evidence-driven**:

- seedable inputs → identical reports for identical seeds
- recorded environment + exact backend versions
- evidence objects: *observed X at node Y, magnitude Z, likely mechanism M,
  confidence C%*
- a permanent regression corpus so `cvconform` gets smarter about how vision
  models fail in the real world

## What "conformance" means (and doesn't)

`cvconform` verifies **deployment conformance**: that a compiled artifact
(ONNX, CoreML, …) still behaves like your reference model. It answers **"is
this still the same model?"**, not **"is this model accurate?"**.

- accurate-but-broken-by-export → **detected**
- inaccurate-but-faithfully-exported → conformant (that's a calibration
  problem, out of scope here)

## Architecture

```
loaders/    framework -> VisionGraph IR
ir/         VisionGraph, operators, precision, passes, registry
autodetect/ zero-config model + calibration discovery
config/     .cvconform.yaml read/write
runtimes/   versioned wrappers (onnxruntime, pytorch, coreml, ...)
engines/    differential, comparison, analyze (root cause)
discovery/  input generation + expedition
reduce/     failing-input minimization
corpus/     regression memory
research/   AI research agent
report/     human + JSON conformance reports
cli.py      init / verify / discover
```

## Development

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
python -m pytest
```

Typed Python with dataclasses, conventional commits, a green `pytest` run
before any PR. See `PHASE1_ZERO_CONFIG.md`, `ARCHITECTURE.md`, and
`TECHNICAL_ROADMAP.md` for the plan.

## License

MIT © David Nichols
