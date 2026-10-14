"""Independent native-engine process; cold load, warmup and real 10-step requests."""
import argparse
import ctypes
import json
import os
import pathlib
import threading
import time
import traceback

import numpy as np
import torch

from common import config_and_transforms, prepare, sha, verified, write_json
from runtime import Policy


class MemoryMonitor:
    class Info(ctypes.Structure):
        _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]

    def __init__(self):
        self.lib = ctypes.CDLL('libnvidia-ml.so.1')
        assert self.lib.nvmlInit_v2() == 0
        self.handle = ctypes.c_void_p()
        assert self.lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(0), ctypes.byref(self.handle)) == 0
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def sample(self):
        value = self.Info()
        assert self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(value)) == 0
        return {'monotonic_seconds': time.monotonic(), 'device_used_bytes': value.used,
                'device_total_bytes': value.total}

    def loop(self):
        while not self.stop.is_set():
            self.samples.append(self.sample())
            self.stop.wait(.05)

    def close(self):
        self.stop.set()
        self.thread.join()
        self.samples.append(self.sample())
        self.lib.nvmlShutdown()
        return self.samples


def main(out, bundle, pilot):
    out.mkdir(parents=True, exist_ok=False)
    specification = json.loads(bundle.read_text())
    rows = json.loads(pilot.read_text())
    verified(pilot)
    write_json(out / 'registration.json', {'pid': os.getpid(), 'bundle_sha256': sha(bundle),
        'pilot_sha256': sha(pilot), 'algorithm': specification['algorithm'], 'observations': len(rows),
        'preprocessing': 'original openpi transforms and Torch preprocessing without a source model',
        'same_noise_and_ten_denoising_steps': True, 'full_float_model_loaded': False,
        'memory_test_is_development_4090_only': True})
    monitor = MemoryMonitor()
    baseline = monitor.sample()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg, inp, output = config_and_transforms()
    policy = Policy(specification)
    metadata = policy.metadata()
    write_json(out / 'runtime.json', metadata)
    first = rows[0]
    verified(pathlib.Path(first['path']))
    ob = prepare(dict(np.load(first['path'])), inp)
    torch.manual_seed(first['noise_seed'])
    noise = torch.randn(1,10,32,device='cuda')
    # A complete ten-step warmup exercises every registered time input.
    policy.set_profiling(True)
    warm = policy.infer(ob, noise, audit=True)
    write_json(out / 'layer_profile.json', {'scope': 'one warm request, diagnostic only',
        'profiles': policy.profiles(), 'target_device_latency_accepted': False})
    policy.set_profiling(False)
    actions, velocity, cases = [], [], []
    start = time.monotonic()
    for index, row in enumerate(rows):
        verified(pathlib.Path(row['path']))
        ob = prepare(dict(np.load(row['path'])), inp)
        torch.manual_seed(row['noise_seed'])
        result = policy.infer(ob, torch.randn(1,10,32,device='cuda'), audit=True, trace=True)
        state = ob.state[0].cpu().numpy()
        internal = result['internal_actions']
        env = output({'state': state.copy(), 'actions': internal[0].numpy()})['actions']
        assert env.shape == (10,7) and np.isfinite(env).all()
        actions.append(internal)
        velocity.append(result['velocity_fields'])
        cases.append({'task': row['task'], 'seed': row['seed'], 'step': row['step'],
            'file_sha256': row['file_sha256'], 'env_actions': env.tolist(),
            'finite': result['finite'], 'cache_immutable': result['cache_immutable']})
        if (index+1) % 16 == 0:
            print('NATIVE INFERENCE', specification['algorithm'], index+1, len(rows), flush=True)
    torch.save({'actions': torch.cat(actions), 'velocity_fields': torch.stack(velocity)}, out / 'outputs.pt')
    write_json(out / 'cases.json', cases)
    samples = monitor.close()
    write_json(out / 'memory_samples.json', samples)
    summary = {'completed': True, 'pid': os.getpid(), 'algorithm': specification['algorithm'],
        'observations': len(rows), 'all_finite': all(r['finite'] for r in cases),
        'all_cache_immutable': all(r['cache_immutable'] for r in cases),
        'elapsed_seconds_including_diagnostic_transfers': time.monotonic()-start,
        'device_memory_peak_bytes': max(s['device_used_bytes'] for s in samples),
        'device_memory_baseline_bytes': baseline['device_used_bytes'],
        'torch_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'torch_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'warmup_complete_ten_steps': True, 'full_float_source_loaded': False,
        'reload_comparison': 'pending independent process comparison',
        '4070_8GB_accepted': False, 'closed_loop_success_accepted': False}
    write_json(out / 'summary.json', summary)
    print('NATIVE PROCESS COMPLETE', summary, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=pathlib.Path, required=True)
    p.add_argument('--bundle', type=pathlib.Path, required=True)
    p.add_argument('--pilot', type=pathlib.Path, required=True)
    a = p.parse_args()
    try:
        main(a.out, a.bundle, a.pilot)
    except Exception:
        if a.out.exists() and not (a.out / 'failure.json').exists():
            write_json(a.out / 'failure.json', {'traceback': traceback.format_exc()})
        raise
