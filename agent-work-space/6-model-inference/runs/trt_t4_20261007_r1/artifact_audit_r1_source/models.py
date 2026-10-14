"""One resident original or TensorRT policy; timing does not alter arithmetic."""
import json
import time

import numpy as np
import torch

from common import ROOT, config_and_transforms, frozen_module, prepare, verified


class InferenceModel:
    def __init__(self, variant):
        self.variant = variant
        self.events = []
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.cfg, self.inputs, self.outputs = config_and_transforms()
        if variant == 'original_torch':
            h = frozen_module('t4_original_float_helpers', ROOT / 'versions/trt_t0_t2_20261007_r1/common.py')
            self.net, _ = h.load_model()
            self.net.paligemma_with_expert.embed_image = self.timed(self.net.paligemma_with_expert.embed_image, 'vision')
            self.net.denoise_step = self.timed(self.net.denoise_step, 'action_step')
            old = self.net.paligemma_with_expert.forward
            def forward(*args, **kwargs):
                if kwargs.get('inputs_embeds', [None])[0] is not None:
                    return self.timed(old, 'prefix')(*args, **kwargs)
                return old(*args, **kwargs)
            self.net.paligemma_with_expert.forward = forward
            self.stream = torch.cuda.current_stream()
            self.metadata = {'variant': variant, 'single_model_resident': True,
                'source_weight_sha256': h.SOURCE_SHA, 'floating_source_model_instantiated': True,
                'reference_backend': 'original same-source PyTorch', 'JAX_performance_claim': False}
        else:
            algorithm = 'float' if variant == 'trt_float' else variant
            path = ROOT / f'runs/trt_t3_20261007_r1/bundles/{algorithm}.json'
            verified(path)
            spec = json.loads(path.read_text())
            runtime = frozen_module('t4_frozen_t3_runtime', ROOT / 'versions/trt_t3_20261007_r1/runtime.py')
            self.policy = runtime.Policy(spec)
            for stage, engine in self.policy.engines.items():
                engine.call = self.timed(engine.call, stage, passed_stream=True)
            self.stream = self.policy.stream
            self.metadata = {'variant': variant, 'bundle': str(path), **self.policy.metadata()}

    def timed(self, function, stage, passed_stream=False):
        def wrapper(*args, **kwargs):
            stream = args[0] if passed_stream else torch.cuda.current_stream()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(stream)
            result = function(*args, **kwargs)
            end.record(stream)
            self.events.append((stage, start, end))
            return result
        return wrapper

    def infer(self, record, noise_seed):
        started = time.perf_counter()
        self.events = []
        ob = prepare(record, self.inputs)
        torch.manual_seed(int(noise_seed))
        noise = torch.randn(1,10,32,device='cuda',dtype=torch.float32)
        torch.cuda.synchronize()
        preprocessed = time.perf_counter()
        self.stream.wait_stream(torch.cuda.current_stream())
        first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        first.record(self.stream)
        with torch.inference_mode():
            if self.variant == 'original_torch':
                internal = self.net.sample_actions('cuda',ob,noise=noise,num_steps=10).cpu()
            else:
                result = self.policy.infer(ob,noise,audit=False,trace=False)
                internal = result['internal_actions']
        last.record(self.stream)
        last.synchronize()
        sampled = time.perf_counter()
        assert internal.shape == (1,10,32) and bool(torch.isfinite(internal).all())
        output = self.outputs({'state':ob.state[0].cpu().numpy(),'actions':internal[0].numpy()})['actions']
        assert output.shape == (10,7) and np.isfinite(output).all()
        stages, counts = {}, {}
        for name, start, end in self.events:
            end.synchronize()
            stages[name] = stages.get(name,0.) + start.elapsed_time(end)
            counts[name] = counts.get(name,0) + 1
        assert counts == {'vision':3,'prefix':1,'action_step':10}, counts
        ended = time.perf_counter()
        return output, {'preprocessing_and_noise_ms':(preprocessed-started)*1000,
            'sample_wall_ms':(sampled-preprocessed)*1000,
            'sample_stream_elapsed_ms':first.elapsed_time(last),
            'postprocessing_and_timing_collection_ms':(ended-sampled)*1000,
            'model_path_ms':(ended-started)*1000,'module_gpu_ms':stages,'module_calls':counts,
            'torch_allocated_bytes':torch.cuda.memory_allocated(),'torch_reserved_bytes':torch.cuda.memory_reserved(),
            'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'torch_peak_reserved_bytes':torch.cuda.max_memory_reserved()}
