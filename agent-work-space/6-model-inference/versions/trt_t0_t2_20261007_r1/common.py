"""Version-local TensorRT preparation; the repository and frozen weights are read-only."""
import dataclasses
import hashlib
import json
import pathlib

import numpy as np
import torch

VERSION = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path('/work') if pathlib.Path('/work/weights').exists() else VERSION.parents[1]
SOURCE = ROOT / 'weights/pi05_libero_bf16_values_r1'
SOURCE_SHA = '69ee787ec12921613feabf69b01348628b3d6327a8865cbc02ecb9be788d9258'


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def load_model(device='cuda', verify=True):
    from openpi.training.config import get_config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from safetensors.torch import load_file
    cfg = get_config('pi05_libero')
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    assert cfg.model.pi05 and not cfg.model.discrete_state_input
    assert (cfg.model.action_dim, cfg.model.action_horizon) == (32, 10)
    if verify:
        assert sha(SOURCE / 'model.safetensors') == SOURCE_SHA
    with torch.device('meta'):
        net = PI0Pytorch(cfg.model)
    net.paligemma_with_expert.paligemma.lm_head = torch.nn.Identity()
    net.paligemma_with_expert.gemma_expert.lm_head = torch.nn.Identity()
    for m in net.modules():
        if type(m).__name__ == 'GemmaRotaryEmbedding':
            other = type(m)(config=m.config, device='cpu')
            m.inv_freq = other.inv_freq
            m.original_inv_freq = other.original_inv_freq
        for name, b in list(m._buffers.items()):
            if b is not None and b.device.type == 'meta':
                assert name == 'position_ids', (type(m).__name__, name)
                m._buffers[name] = torch.arange(b.shape[-1], dtype=b.dtype)[None, :]
    weights = load_file(str(SOURCE / 'model.safetensors'))
    net.load_state_dict(weights, strict=True, assign=True)
    del weights
    net.eval().to(device)
    # sample_actions sets these on every call; set them also for isolated prefill.
    net.paligemma_with_expert.paligemma.language_model.config._attn_implementation='eager'
    net.paligemma_with_expert.gemma_expert.model.config._attn_implementation='eager'
    return net, cfg


def transforms(cfg):
    from openpi import transforms as tr
    from openpi.shared import normalize
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    stats = normalize.load(SOURCE / 'assets' / dc.asset_id)
    inputs = tr.compose([*dc.data_transforms.inputs,
        tr.Normalize(stats, use_quantiles=dc.use_quantile_norm), *dc.model_transforms.inputs])
    outputs = tr.compose([*dc.model_transforms.outputs,
        tr.Unnormalize(stats, use_quantiles=dc.use_quantile_norm), *dc.data_transforms.outputs])
    return inputs, outputs


def prepare(record, cfg, device='cuda'):
    import jax
    from openpi.models.model import Observation
    raw = {'observation/image': record['image'], 'observation/wrist_image': record['wrist_image'],
           'observation/state': np.asarray(record['state'], dtype=np.float32), 'prompt': str(record['prompt'])}
    inputs, _ = transforms(cfg)
    x = inputs(raw)
    return Observation.from_dict(jax.tree.map(lambda a: torch.from_numpy(np.array(a))[None].to(device), x))


def time_table(device='cuda'):
    from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
    device = torch.device(device)
    t = torch.tensor(1., device=device, dtype=torch.float32)
    dt = torch.tensor(-.1, device=device, dtype=torch.float32)
    times, embs = [], []
    while t >= -dt / 2:
        times.append(t.clone())
        embs.append(create_sinusoidal_pos_embedding(t[None], 1024, 4e-3, 4., device).to(t.dtype))
        t += dt
    assert len(times) == 10
    return torch.stack(times), torch.cat(embs), dt


def selected_language(net):
    return {n: m for n, m in net.named_modules() if isinstance(m, torch.nn.Linear)
            and n.startswith('paligemma_with_expert.paligemma.model.language_model.layers.')
            and ('.self_attn.' in n or '.mlp.' in n)}
