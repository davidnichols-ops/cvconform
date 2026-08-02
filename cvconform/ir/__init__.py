"""Vision IR — the LLVM-IR-style intermediate representation for vision models.

A :class:`VisionGraph` is a DAG of :class:`Operator` nodes connected by named
tensor edges. Every node describes an *op*, its *attributes*, the *tensor specs*
it consumes and produces (shape, dtype, precision, quant info), and its
*provenance* (which loader made it, from what original string).

Frameworks do not get to be special here: every loader (onnx, torchscript,
coreml, tensorflow, tensorrt) lowers *into* :class:`VisionGraph`. Everything
above the loaders is framework-agnostic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class Precision(str, Enum):
    """Numerical precision of a tensor."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP64 = "fp64"
    INT8 = "int8"
    UINT8 = "uint8"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"
    UNKNOWN = "unknown"

    @classmethod
    def from_dtype(cls, dtype: str) -> "Precision":
        m = {
            "float32": cls.FP32,
            "float16": cls.FP16,
            "bfloat16": cls.BF16,
            "float64": cls.FP64,
            "int8": cls.INT8,
            "uint8": cls.UINT8,
            "int32": cls.INT32,
            "int64": cls.INT64,
            "bool": cls.BOOL,
        }
        return m.get(dtype.lower(), cls.UNKNOWN)


@dataclass
class Quantization:
    """Quantization metadata (for int8/uint8 tensors)."""

    scale: Optional[Any] = None
    zero_point: Optional[Any] = None
    per_channel: bool = False
    axis: int = 0


@dataclass
class TensorSpec:
    """Describes a tensor flowing between operators."""

    shape: Optional[Tuple[int, ...]] = None
    dtype: str = "float32"
    precision: Optional[Precision] = None
    quant: Optional[Quantization] = None
    layout: str = "NCHW"  # NCHW | NHWC | NC | ND

    def __post_init__(self) -> None:
        if self.precision is None:
            self.precision = Precision.from_dtype(self.dtype)

    def rank(self) -> Optional[int]:
        return None if self.shape is None else len(self.shape)

    def is_quantized(self) -> bool:
        return self.quant is not None and self.quant.scale is not None

    def desc(self) -> str:
        parts = []
        if self.shape:
            parts.append("x".join(str(s) if s is not None else "?" for s in self.shape))
        parts.append(self.dtype)
        if self.layout:
            parts.append(self.layout)
        if self.quant and self.quant.scale is not None:
            parts.append(f"q(scale={to_float(self.quant.scale)})")
        return "[" + ", ".join(parts) + "]"


def to_float(v: Any) -> Any:
    """Best-effort scalar conversion for rendering quant scales."""
    try:
        import numpy as np

        if hasattr(v, "item"):
            return v.item()
        return v
    except Exception:  # pragma: no cover - defensive
        return v


@dataclass
class Operator:
    """A single node in a VisionGraph."""

    op: str
    name: str
    inputs: List[Tuple[str, TensorSpec]] = field(default_factory=list)
    outputs: List[Tuple[str, TensorSpec]] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    provenance: str = ""  # e.g. "onnx: Conv" or "torchscript: aten::conv2d"

    def __post_init__(self) -> None:
        self._out_edge: Optional[str] = None

    def input_names(self) -> List[str]:
        return [n for n, _ in self.inputs]

    def output_names(self) -> List[str]:
        return [n for n, _ in self.outputs]

    def input_spec(self) -> List[TensorSpec]:
        return [s for _, s in self.inputs]

    def output_spec(self) -> List[TensorSpec]:
        return [s for _, s in self.outputs]

    def short(self) -> str:
        out = ", ".join(self.output_names()) or "?"
        ins = ", ".join(self.input_names()) or "?"
        return f"{self.op}({ins} -> {out})"

    def describe(self) -> str:
        lines = [f"{self.name}: {self.op}  [{self.provenance}]"]
        for n, s in self.inputs:
            lines.append(f"    in  {n} {s.desc()}")
        for n, s in self.outputs:
            lines.append(f"    out {n} {s.desc()}")
        if self.attrs:
            lines.append(f"    attrs: {self.attrs}")
        return "\n".join(lines)


class OperatorRegistry:
    """Canonical operator namespace."""

    IDENTITY = "cv.Identity"
    CONV2D = "cv.Conv2D"
    CONVTRANSPOSE2D = "cv.ConvTranspose2D"
    DEPTHWISECONV = "cv.DepthwiseConv"
    BATCHNORM = "cv.BatchNorm"
    GROUPNORM = "cv.GroupNorm"
    LAYERNORM = "cv.LayerNorm"
    INSTANCENORM = "cv.InstanceNorm"
    GEMM = "cv.Gemm"
    MATMUL = "cv.MatMul"
    ADD = "cv.Add"
    MUL = "cv.Mul"
    SUB = "cv.Sub"
    DIV = "cv.Div"
    MAXPOOL = "cv.MaxPool"
    AVGPOOL = "cv.AvgPool"
    GLOBALAVGPOOL = "cv.GlobalAvgPool"
    RELU = "cv.ReLU"
    LEAKY_RELU = "cv.LeakyReLU"
    SIGMOID = "cv.Sigmoid"
    SOFTMAX = "cv.Softmax"
    SILU = "cv.SiLU"
    HARDSWISH = "cv.HardSwish"
    RESIZE = "cv.Resize"
    TRANSPOSE = "cv.Transpose"
    RESHAPE = "cv.Reshape"
    FLATTEN = "cv.Flatten"
    CONCAT = "cv.Concat"
    SPLIT = "cv.Split"
    SLICE = "cv.Slice"
    GATHER = "cv.Gather"
    SQUEEZE = "cv.Squeeze"
    UNSQUEEZE = "cv.Unsqueeze"
    PAD = "cv.Pad"
    CONSTANT = "cv.Constant"
    INPUT = "cv.Input"
    OUTPUT = "cv.Output"
    NMS = "cv.NMS"
    REDUCEMEAN = "cv.ReduceMean"
    CLIP = "cv.Clip"
    # Non-tensor / control / head ops
    UNSUPPORTED = "cv.Unsupported"

    # Ops that delimit the model boundary / represent inputs and outputs.
    BOUNDARY = {INPUT, OUTPUT, CONSTANT}


@dataclass
class VisionGraph:
    """A DAG of operators describing a vision model.

    ``inputs`` and ``outputs`` are the named graph-level boundaries, mapping
    external names to a producer node name. Operators reference each other by
    name via their edge lists.
    """

    name: str = ""
    ops: Dict[str, Operator] = field(default_factory=dict)
    inputs: Dict[str, str] = field(default_factory=dict)  # ext name -> producer op
    outputs: Dict[str, str] = field(default_factory=dict)  # ext name -> producer op
    loader: str = ""
    loader_version: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_op(self, op: Operator) -> "VisionGraph":
        assert op.name not in self.ops, f"duplicate op name {op.name}"
        self.ops[op.name] = op
        # wire graph-level boundaries from Input/Output boundary nodes
        if op.op == OperatorRegistry.INPUT and op.outputs:
            self.inputs[op.outputs[0][0]] = op.name
        if op.op == OperatorRegistry.OUTPUT and op.inputs:
            self.outputs[op.inputs[0][0]] = op.name
        return self

    def refresh_boundaries(self) -> "VisionGraph":
        """Recompute graph-level inputs/outputs from boundary nodes."""
        self.inputs = {}
        self.outputs = {}
        for op in self.ops.values():
            if op.op == OperatorRegistry.INPUT:
                for n, _ in op.outputs:
                    self.inputs[n] = op.name
            elif op.op == OperatorRegistry.OUTPUT:
                for n, _ in op.inputs:
                    self.outputs[n] = op.name
        return self

    def topological_order(self) -> List[Operator]:
        """Return operators in topological (dependencies-before-consumer) order."""
        order: List[Operator] = []
        visited = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            op = self.ops.get(name)
            if op is None:
                return
            for in_name, _ in op.inputs:
                # in edges reference producer op names
                if in_name in self.ops:
                    visit(in_name)
            order.append(op)

        for name in self.ops:
            visit(name)
        return order

    def schema_hash(self) -> str:
        """Stable hash of the graph's operator schema (op names + attrs + specs).

        Two graphs with the same schema hash are structurally conformant at the
        IR level. Precision/shape changes are reflected here.
        """
        h = hashlib.sha256()
        h.update(self.name.encode())
        for op in self.topological_order():
            h.update(op.op.encode())
            for n, s in op.inputs + op.outputs:
                h.update(str(s.desc()).encode())
            h.update(repr(sorted(str(k) + "=" + str(v) for k, v in op.attrs.items())).encode())
        return h.hexdigest()

    def describe(self) -> str:
        lines = [f"VisionGraph: {self.name} (loader={self.loader} {self.loader_version})"]
        lines.append(f"  inputs: {list(self.inputs)}")
        lines.append(f"  outputs: {list(self.outputs)}")
        lines.append(f"  operators ({len(self.ops)}):")
        for op in self.topological_order():
            lines.append("    " + op.short())
        return "\n".join(lines)
