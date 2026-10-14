"""Native TensorRT builder and synchronous, single-stream Torch buffer bridge."""
import pathlib
import numpy as np
import torch
import tensorrt as trt

LOGGER = trt.Logger(trt.Logger.WARNING)
DTYPES = {trt.float32: torch.float32, trt.float16: torch.float16, trt.bfloat16: torch.bfloat16,
          trt.int32: torch.int32, trt.int64: torch.int64, trt.bool: torch.bool, trt.int8: torch.int8}


def build(onnx_path, engine_path, workspace=2 * 1024**3):
    builder = trt.Builder(LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
    config.builder_optimization_level = 3
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError('TensorRT returned no serialized engine')
    serialized=bytes(plan)
    with pathlib.Path(engine_path).open('xb') as f:
        f.write(serialized)
    return {'inputs': [network.get_input(i).name for i in range(network.num_inputs)],
            'outputs': [network.get_output(i).name for i in range(network.num_outputs)],
            'layers': network.num_layers, 'engine_bytes': len(serialized)}


class Engine:
    def __init__(self, path):
        self.runtime = trt.Runtime(LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(pathlib.Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError('Engine deserialization failed')
        self.context = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.names if n not in self.inputs]

    def __call__(self, *values):
        assert len(values) == len(self.inputs), (self.inputs, len(values))
        buffers = {}
        for name, value in zip(self.inputs, values):
            assert tuple(value.shape) == tuple(self.engine.get_tensor_shape(name)), (name, value.shape)
            assert value.dtype == DTYPES[self.engine.get_tensor_dtype(name)], (name, value.dtype)
            buffers[name] = value.contiguous()
        for name in self.outputs:
            buffers[name] = torch.empty(tuple(self.engine.get_tensor_shape(name)),
                                        dtype=DTYPES[self.engine.get_tensor_dtype(name)], device='cuda')
        for name, value in buffers.items():
            self.context.set_tensor_address(name, value.data_ptr())
        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('TensorRT execution failed')
        stream.synchronize()
        return tuple(buffers[n] for n in self.outputs)


def metrics(a, b):
    d = (a.float() - b.float()).abs().detach().cpu().numpy().reshape(-1)
    return {'max_abs': float(d.max()), 'mae': float(d.mean()), 'p99_abs': float(np.quantile(d, .99)),
            'finite': bool(torch.isfinite(b).all())}
