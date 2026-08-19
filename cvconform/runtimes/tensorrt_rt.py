"""Runtime: TensorRT via tensorrt Python API.

TensorRT is NVIDIA's inference optimization engine for NVIDIA GPU deployments.
Engines are compiled for specific GPU architectures and provide maximum throughput.

This runtime loads a TensorRT engine (.engine or .plan file) and executes it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from cvconform.runtimes import RuntimeInfo, env_fingerprint, normalize_tensor_outputs, to_numpy

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    _HAS = True
except Exception:  # pragma: no cover
    trt = None
    cuda = None
    _HAS = False


class TensorRTRuntime:
    runtime_name = "tensorrt"

    def __init__(
        self,
        engine_path: str,
        workspace_size: int = 1 << 30,  # 1GB default
        **kwargs
    ):
        if not _HAS:
            raise ImportError(
                "tensorrt and pycuda are required for tensorrt runtime. "
                "Install with: pip install tensorrt pycuda"
            )
        self.engine_path = engine_path
        self.workspace_size = workspace_size
        self._engine = None
        self._context = None
        self._bindings = None
        self._input_names = None
        self._output_names = None
        self._stream = None
        self._load_engine()

    def _load_engine(self):
        """Load TensorRT engine from file."""
        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self._engine = runtime.deserialize_cuda_engine(f.read())
        
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize engine from {self.engine_path}")
        
        self._context = self._engine.create_execution_context()
        self._stream = cuda.Stream()
        
        # Get binding names and shapes
        self._input_names = []
        self._output_names = []
        self._bindings = [None] * self._engine.num_bindings
        
        for i in range(self._engine.num_bindings):
            name = self._engine.get_binding_name(i)
            is_input = self._engine.binding_is_input(i)
            dtype = trt.nptype(self._engine.get_binding_dtype(i))
            shape = self._engine.get_binding_shape(i)
            
            if is_input:
                self._input_names.append(name)
            else:
                self._output_names.append(name)

    def info(self) -> RuntimeInfo:
        """Return runtime version and environment info."""
        import torch
        cuda_version = "unknown"
        try:
            cuda_version = torch.version.cuda or "unknown"
        except Exception:
            pass
        
        gpu_name = "unknown"
        try:
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        except Exception:
            pass
        
        return RuntimeInfo(
            runtime=self.runtime_name,
            runtime_version=trt.__version__ if trt else "unknown",
            framework="tensorrt",
            framework_version=trt.__version__ if trt else "unknown",
            provider="CUDA",
            device=gpu_name,
            env_fingerprint=env_fingerprint(),
            extra={
                "engine_path": self.engine_path,
                "workspace_size": self.workspace_size,
                "num_bindings": self._engine.num_bindings if self._engine else 0,
                "input_names": self._input_names,
                "output_names": self._output_names,
            },
        )

    def run(self, feeds: Dict[str, np.ndarray], output_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute the TensorRT engine on the given feeds."""
        if self._context is None or self._engine is None:
            raise RuntimeError("Engine not loaded")
        
        # Allocate device memory for inputs and outputs
        d_inputs = {}
        d_outputs = {}
        h_outputs = {}
        
        try:
            # Prepare input buffers
            for name in self._input_names:
                if name not in feeds:
                    raise ValueError(f"Missing input: {name}")
                h_input = np.ascontiguousarray(feeds[name])
                d_input = cuda.mem_alloc(h_input.nbytes)
                cuda.memcpy_htod_async(d_input, h_input, self._stream)
                d_inputs[name] = d_input
                self._bindings[self._engine.get_binding_index(name)] = int(d_input)
            
            # Prepare output buffers
            out_names = output_names or self._output_names
            for name in out_names:
                binding_idx = self._engine.get_binding_index(name)
                shape = self._context.get_binding_shape(binding_idx)
                dtype = trt.nptype(self._engine.get_binding_dtype(binding_idx))
                h_output = np.empty(shape, dtype=dtype)
                d_output = cuda.mem_alloc(h_output.nbytes)
                d_outputs[name] = d_output
                h_outputs[name] = h_output
                self._bindings[binding_idx] = int(d_output)
            
            # Execute
            self._context.execute_async_v2(
                bindings=self._bindings,
                stream_handle=self._stream.handle
            )
            self._stream.synchronize()
            
            # Copy outputs back
            results = {}
            for name in out_names:
                cuda.memcpy_dtoh_async(h_outputs[name], d_outputs[name], self._stream)
            self._stream.synchronize()
            
            for name in out_names:
                results[name] = h_outputs[name]
            
            return results
            
        finally:
            # Cleanup device memory
            for d in d_inputs.values():
                d.free()
            for d in d_outputs.values():
                d.free()

    def run_and_normalize(self, feeds: Dict[str, np.ndarray], output_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute and normalize outputs to cvconform schema."""
        raw = self.run(feeds, output_names)
        return normalize_tensor_outputs(raw, list(raw.keys()))


def load_tensorrt_engine(path: str) -> bytes:
    """Load TensorRT engine file as bytes (for artifact passing)."""
    with open(path, "rb") as f:
        return f.read()