# cvconform

**The correctness and reliability layer for computer vision AI.**

Computer vision has a hidden reliability crisis. A model is trained once
(PyTorch, CUDA, FP32) and then silently transformed — ONNX, TensorRT, CoreML,
OpenVINO, TFLite, quantization, operator fusion. Every transformation creates
opportunities for the model to stop being itself.

The industry mostly asks *"Can this model run?"*

`cvconform` answers the question nobody was asking: **"Is this still the same
model?"**

```python
from cvconform import verify

result = verify(
    model="yolo11.pt",
    reference="pytorch",
    targets=["onnx", "coreml"],
    seed=0,
)
```

```
VISION CONFORM REPORT
Model:       yolo11.pt
Reference:   pytorch

CONFORMANCE SCORES
  onnx            100.00%
  coreml           96.36%

FINDINGS: none — all targets conformant within policy.
```

## Why

Every deployment path adds silent risk:

- **FP16 accumulation drift** → different classes on some images
- **Operator fusion** → changed execution order
- **Quantization scale mismatch** → post-quantization accuracy drop
- **NMS implementation differences** → different detections
- **Unsupported-operator fallback** → a graph that silently does less

`cvconform` executes the *same model* on *multiple runtimes* with the *same
seeded inputs*, compares the outputs, and — when they differ — tells you
*exactly why*: which backend, which op, which precision/fusion change, which
images.

## Install

```bash
pip install cvconform          # core: numpy, onnx, onnxruntime
pip install 'cvconform[pytorch]'   # torch reference
pip install 'cvconform[coreml]'    # CoreML target (macOS)
pip install 'cvconform[openvino]'  # OpenVINO target
```

> Requires Python 3.10+. On Apple Silicon you get CoreML + MLX paths for free.

## Quickstart

```bash
# CLI — human + JSON report
cvconform verify model.pt --reference pytorch --targets onnx,coreml
cvconform verify model.pt --json report.json

# Python API
from cvconform import verify
report = verify(model="model.pt", reference="pytorch",
                targets=["onnx", "coreml"])
```

## What it can do

| System | What it does |
|---|---|
| **Vision IR** | A framework-agnostic `VisionGraph` (LLVM-IR-for-vision): operators, tensors, shapes, precision, quantization. Every loader lowers into it. |
| **Differential execution** | Same inputs → reference + every target → normalized outputs → per-kind comparison (tensors, boxes, classes, scores). |
| **Root-cause analysis** | Detects *and explains*: fusion order, FP16/FP32 drift, quant scale, fallback, padding, NMS. Evidence objects, not vibes. |
| **Failure discovery** | Deterministic input generator (noise, edges, illumination, blur, compression, extremes) that hunts for diverging inputs. |
| **Failure reduction** | Minimizes a failing input to a small reproducing patch (compiler-minimization style). |
| **Regression memory** | Every discovered failure becomes a permanent, versioned corpus entry. The system gets smarter forever. |
| **AI research agent** | On failure, searches public sources for the mechanism + affected/fixed versions + a recommendation. |

## What "conformance" means (and doesn't)

`cvconform` verifies **deployment conformance**: that a compiled artifact (ONNX,
CoreML, ...) still behaves like your reference model — same inputs, same
outputs, within tolerance. It answers **"is this still the same model?"**, not
**"is this model accurate?"**.

- A model that is *accurate-but-broken-by-export* → detected (good).
- A model that is *inaccurate-but-faithfully-exported* → conformant, because the
  inaccurate behavior is preserved faithfully. That's a training/calibration
  problem, out of `cvconform`'s scope.
- The fault-injection suite (`tests/test_fault_injection.py`) proves the engine
  detects genuine clean-vs-corrupt divergence; the `verify` CLI proves the
  compiled artifact matches the reference.


## The engineering contract

Every result is **scientific and reproducible**:

- seedable inputs
- recorded environment + exact backend versions
- deterministic reproduction
- evidence objects with observed magnitude, location, and confidence

No "probably a bug." Always: *Observed X at node Y, magnitude Z, likely
mechanism M, confidence C%.*

## Architecture

```
loaders/    framework -> VisionGraph IR
ir/         VisionGraph, operators, precision, passes
runtimes/   versioned wrappers (onnxruntime, pytorch, coreml, ...)
engines/    differential, comparison, analyze (root cause)
discovery/  input generation + expedition
reduce/     failing-input minimization
corpus/     regression memory
research/   AI research agent
report/     human + JSON conformance reports
```

See `ARCHITECTURE.md`, `VISION.md`, `PRODUCT_REQUIREMENTS.md`,
`TECHNICAL_ROADMAP.md`, `CURRENT_STATE.md`.

## Contributing

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
python -m pytest
```

We prefer typed Python with dataclasses, conventional commits, and a green
`pytest` run before any PR.

## License

MIT.
