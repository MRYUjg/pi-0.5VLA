"""Loopback inference, frozen transforms, request-specific reproducible diffusion noise."""
import argparse
import base64
import hashlib
import json
import pathlib
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jax
import numpy as np
from openpi import transforms
from openpi.models import model
from openpi.policies.policy import Policy
from openpi.shared import normalize
from openpi.training import config

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'versions/lora_20261007_r1/scripts'))
from configuration import make, TRAIN_ID


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=['finetuned', 'base'], required=True)
    p.add_argument('--out', type=pathlib.Path, required=True)
    p.add_argument('--port', type=int, default=8872)
    a = p.parse_args()
    train = ROOT / 'runs/spatial_lora_20261007_r1'
    if a.kind == 'finetuned':
        assert json.loads((train / 'reload_result.json').read_text())['pass']
        assert json.loads((train / 'selected_checkpoint.json').read_text())['step'] == 10000
        cfg = make(train)
        checkpoint = train / 'checkpoint/10000'
        norm_path = checkpoint / 'assets' / TRAIN_ID
    else:
        cfg = config.get_config('pi05_libero')
        checkpoint = pathlib.Path('/openpi_assets/openpi-assets/checkpoints/pi05_libero')
        norm_path = checkpoint / 'assets/physical-intelligence/libero'
    net = cfg.model.load(model.restore_params(checkpoint / 'params'))
    net.eval()
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    stats = normalize.load(norm_path)
    policy = Policy(net,
        transforms=[*dc.data_transforms.inputs,
                    transforms.Normalize(stats, use_quantiles=dc.use_quantile_norm),
                    *dc.model_transforms.inputs],
        output_transforms=[*dc.model_transforms.outputs,
                           transforms.Unnormalize(stats, use_quantiles=dc.use_quantile_norm),
                           *dc.data_transforms.outputs], sample_kwargs={'num_steps': 10})
    lock = threading.Lock()
    meta = {'kind': a.kind, 'checkpoint': str(checkpoint), 'norm_path': str(norm_path),
            'norm_sha256': hashlib.sha256((norm_path / 'norm_stats.json').read_bytes()).hexdigest(),
            'action_horizon': 10, 'action_dim': 7, 'replan_steps': 5,
            'discrete_state_input': False, 'extra_delta_transform': False,
            'preserve_checkpoint_parameter_dtype': True,
            'diffusion_noise': 'numpy SeedSequence([episode_seed, query_index, 271828]); float32 (10,32)'}
    # Compile without using any reserved test scene or test observation.
    warm = {'observation/state': np.zeros(8, np.float32), 'prompt': 'pick up the bowl',
            'observation/image': np.zeros((256,256,3), np.uint8),
            'observation/wrist_image': np.zeros((256,256,3), np.uint8)}
    t = time.monotonic()
    action = policy.infer(warm, noise=np.zeros((10,32), np.float32))['actions']
    assert action.shape == (10,7) and np.isfinite(action).all()
    meta['warmup_seconds'] = time.monotonic() - t
    (a.out / ('server_' + a.kind + '_metadata.json')).write_text(json.dumps(meta, indent=2))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps({'ready': True, **meta}).encode())
        def do_POST(self):
            if self.path == '/shutdown':
                self.send_response(200); self.end_headers()
                threading.Thread(target=self.server.shutdown).start()
                return
            try:
                x = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                obs = {'observation/state': np.asarray(x['state'], dtype=np.float32), 'prompt': x['prompt']}
                for k in ('image', 'wrist_image'):
                    obs['observation/' + k] = np.frombuffer(base64.b64decode(x[k]), np.uint8).reshape(256,256,3)
                noise = np.random.default_rng(np.random.SeedSequence(
                    [int(x['seed']), int(x['query']), 271828])).standard_normal((10,32), dtype=np.float32)
                t = time.monotonic()
                with lock:
                    actions = policy.infer(obs, noise=noise)['actions']
                assert actions.shape == (10,7) and np.isfinite(actions).all()
                body = json.dumps({'actions': actions.tolist(), 'seconds': time.monotonic()-t,
                                   'noise_sha256': hashlib.sha256(noise.tobytes()).hexdigest(), 'kind': a.kind}).encode()
                self.send_response(200); self.end_headers(); self.wfile.write(body)
            except Exception as exc:
                self.send_response(500); self.end_headers(); self.wfile.write(repr(exc).encode())
                raise
    print('POLICY READY', a.kind, a.port, flush=True)
    ThreadingHTTPServer(('127.0.0.1', a.port), Handler).serve_forever()
    print('POLICY CLOSED', a.kind, flush=True)


if __name__ == '__main__':
    main()
