# CURRENT_STATE — cvconform

Snapshot of the machine + environment that `cvconform` is being built against.
This document is a living record: update it whenever the environment changes
in a way that matters.

## Identity

**Working name:** `cvconform` — the correctness and reliability layer for
computer vision AI systems. The product asks and answers one question that the
industry ignores:

> "Can this model run?" — everyone asks this.
> **"Is this still the same model?"** — this is what we answer.

Category: **AI Model Reliability Infrastructure**.

## Hardware

| Property | Value |
|---|---|
| Chip | Apple M4 |
| RAM | 16 GB |
| GPU | Apple M4 integrated, Metal 4 |
| ANN accelerator | Apple Neural Engine (16-core) |
| Arch | arm64 |
| Disk free | ~20 GB at `/` |

**Implications**
- No CUDA. The usable device backends are: CPU, Metal, and CoreML/ANE.
- Backends like `tensorrt` cannot run here; they are torchscript-parsed /
  config targets for now (a "compile" path to the runtime, verifiable on
  CI where a GPU exists).
- MLX is a first-class real runtime and a strong differential target.

## Software

| Tool | Version | Notes |
|---|---|---|
| macOS | (Darwin) | Metal 4 |
| Python (system) | 3.14.6 | Too new for native wheels; don't use. |
| Python (project) | 3.12 | via `uv venv --python 3.12` — the stable, wheel-rich target. |
| uv | 0.11.6 | package + env manager |
| Node | v22.23.1 | via mise (working npm). Do NOT use Node 26. |
| git | 2.54.0 | |
| numpy | 2.4.6 (system) | project venv has its own |

## Framework matrix (project venv)

| Framework | Status | Role in cvconform |
|---|---|---|
| numpy | installing | numerics core |
| onnx | installing | IR transport + graph model |
| onnxruntime | installing | CPU differential runtime |
| coremltools | TODO | CoreML compile + runtime (needs py3.12 wheels) |
| torch | TODO | reference backend + torchscript/onnx export |
| openvino | TODO | x86/CPU differential runtime (may run via folk build) |
| tensorflow | TODO (unlikely) | TF reference models; heavy — defer |

## Relevant existing work (in `~/Projects`)

- **local-operator** — distributed operator RL. Not directly used by
  cvconform, but validates the environment + MLX + Colab patterns and the
  "evidence over assertion" lesson the author already runs.
- **Gargantua** — Taichi GPU renderer. Confirms Taichi/graphics toolchain works
  on this M4; a source of synthetic image diversity later (Phase "Failure
  Discovery" can adopt its renderer to generate procedural scenes).
- **aafp / aafp-go** — agent infra; the user's conventional-commit, typed
  Python, pytest-before-commit workflow applies here unchanged.

## Known failure patterns (from lesson bank) that affect this build

1. **Python 3.14 / 2.13 torch / sklearn too new** → breaks coremltools.
   Mitigation: pin project venv to Python 3.12.
2. **libomp double-init** on torch import → set `KMP_DUPLICATE_LIB_OK=TRUE`
   in the dev shell if it bites.
3. **Don't overload the M4** — it is a daily-driver; keep heavy MLX/compute
   jobs short and sequenced, not parallel.
4. **npm devin_on_node_26 broken** — use Node 22. (Not needed for the Python
   core, but relevant if we add a TS MCP shim.)
5. **Verify against real data, never code-only reasoning** — every
   conformance claim in this repo must be backed by a reproducible run.

## Build order intent

Phase 2 (docs) → Phase 3 (vertical slice: torch→onnx→coreml, differential
runner, report) → Phase 4 (break it) → Phase 5 (expand) → Phase 6 (harden).
