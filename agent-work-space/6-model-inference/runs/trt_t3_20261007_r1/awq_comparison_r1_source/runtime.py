"""Single-policy native TensorRT runtime with serial shared enqueue memory."""
import pathlib

import tensorrt as trt
import torch

from common import tensor_inputs, time_table, sha

LOGGER = trt.Logger(trt.Logger.WARNING)
DTYPES = {trt.float32: torch.float32, trt.float16: torch.float16, trt.bfloat16: torch.bfloat16,
          trt.int32: torch.int32, trt.int64: torch.int64, trt.bool: torch.bool, trt.int8: torch.int8}


class LayerProfiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.times = {}
        self.enabled = True

    def report_layer_time(self, name, ms):
        if not self.enabled:
            return
        record = self.times.setdefault(name, {'calls': 0, 'total_ms': 0.0})
        record['calls'] += 1
        record['total_ms'] += ms


class Engine:
    def __init__(self, path, expected_sha):
        assert sha(path) == expected_sha, path
        self.runtime = trt.Runtime(LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(pathlib.Path(path).read_bytes())
        assert self.engine is not None, path
        self.context = self.engine.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
        assert self.context is not None
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.names if n not in self.inputs]
        self.buffers = {n: torch.empty(tuple(self.engine.get_tensor_shape(n)),
                       dtype=DTYPES[self.engine.get_tensor_dtype(n)], device='cuda') for n in self.outputs}
        self.profiler = None

    def bind_workspace(self, workspace):
        assert workspace.data_ptr() % 256 == 0
        assert workspace.numel() >= self.engine.device_memory_size_v2
        self.context.set_device_memory(workspace.data_ptr(), workspace.numel())

    def call(self, stream, *values):
        assert len(values) == len(self.inputs)
        held = {}
        for n, v in zip(self.inputs, values):
            assert tuple(v.shape) == tuple(self.engine.get_tensor_shape(n)), (n, v.shape)
            assert v.dtype == DTYPES[self.engine.get_tensor_dtype(n)], (n, v.dtype)
            held[n] = v.contiguous()
        for n, v in {**held, **self.buffers}.items():
            assert self.context.set_tensor_address(n, v.data_ptr())
        assert stream.cuda_stream != 0
        assert self.context.execute_async_v3(stream.cuda_stream)
        # Completion protects reuse of the shared workspace and input lifetimes.
        stream.synchronize()
        return tuple(self.buffers[n] for n in self.outputs)

    def profile(self, enabled):
        if enabled:
            self.profiler = LayerProfiler()
            self.context.profiler = self.profiler
            self.context.enqueue_emits_profile = True
        elif self.profiler is not None:
            # TensorRT 10.13 setProfiler rejects nullptr. Keep the owner alive
            # and disable automatic reporting; production never attaches it.
            self.profiler.enabled = False
            self.context.enqueue_emits_profile = False


class Policy:
    def __init__(self, specification):
        self.stream = torch.cuda.Stream()
        self.engines = {}
        with torch.cuda.stream(self.stream):
            for name in ['vision', 'prefix', 'action_step']:
                item = specification['engines'][name]
                self.engines[name] = Engine(item['path'], item['sha256'])
            sizes = {n: e.engine.device_memory_size_v2 for n, e in self.engines.items()}
            self.workspace = torch.empty(max(256, max(sizes.values())), dtype=torch.uint8, device='cuda')
            for engine in self.engines.values():
                engine.bind_workspace(self.workspace)
            self.times, self.table, self.dt = time_table()
        self.stream.synchronize()
        self.workspace_sizes = sizes
        self.specification = specification

    def infer(self, observation, noise, audit=False, trace=False):
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream), torch.inference_mode():
            images, pref, action = tensor_inputs(observation)
            # The vision output buffer is shared; preserve each slot before reuse.
            embeddings = torch.cat([self.engines['vision'].call(self.stream, x)[0].clone() for x in images], 1)
            cache = self.engines['prefix'].call(self.stream, embeddings, *pref)
            assert len(cache) == 37 and torch.equal(cache[0], pref[1])
            frozen = tuple(t.clone() for t in cache[1:]) if audit else ()
            x = noise.clone()
            velocity = []
            for i in range(10):
                v, = self.engines['action_step'].call(self.stream, x, self.table[i:i+1], *action, *cache[1:])
                if trace:
                    velocity.append(v.detach().cpu())
                x = x + self.dt * v
            immutable = all(torch.equal(a, b) for a, b in zip(frozen, cache[1:])) if audit else None
            assert not audit or immutable
            finite = bool(torch.isfinite(x).all())
            assert finite
            final = x.cpu()
            result = {'internal_actions': final, 'finite': finite, 'cache_immutable': immutable,
                'KV_shapes': [list(t.shape) for t in cache[1:]],
                'KV_dtypes': [str(t.dtype) for t in cache[1:]],
                'velocity_fields': torch.stack(velocity) if trace else None}
        self.stream.synchronize()
        return result

    def set_profiling(self, enabled):
        for engine in self.engines.values():
            engine.profile(enabled)

    def profiles(self):
        return {n: e.profiler.times for n, e in self.engines.items() if e.profiler is not None}

    def metadata(self):
        return {'single_model_resident': True, 'floating_source_model_instantiated': False,
                'nondefault_stream': True, 'serial_synchronized_workspace_reuse': True,
                'workspace_sizes_per_context': self.workspace_sizes,
                'shared_workspace_bytes': self.workspace.numel(),
                'saved_workspace_bytes': sum(self.workspace_sizes.values()) - self.workspace.numel(),
                'time_sequence': self.times.tolist(), 'CUDA_Graph': False,
                'engines': self.specification['engines'],
                'fixed_interfaces': {n: [{'name': k, 'shape': list(e.engine.get_tensor_shape(k)),
                    'dtype': str(e.engine.get_tensor_dtype(k))} for k in e.names] for n, e in self.engines.items()}}
