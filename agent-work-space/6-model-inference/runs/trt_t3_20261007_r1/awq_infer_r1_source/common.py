"""T3 metadata/preprocessing only: no floating model or safetensor loading."""
import dataclasses
import hashlib
import json
import pathlib

import numpy as np
import torch

ROOT = pathlib.Path('/work')
PREVIOUS = ROOT / 'runs/trt_t0_t2_20261007_r1'
PREVIOUS_MANIFEST = ROOT / 'releases/trt_t0_t2_20261007_r1/manifest.json'
PREVIOUS_SHA = '4f359c7b939e10c7eebba4213f4156aef7dc02874101c9919f9ad0a3222c23ee'
SOURCE_SHA = '69ee787ec12921613feabf69b01348628b3d6327a8865cbc02ecb9be788d9258'


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    with pathlib.Path(path).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')


def verified(path):
    assert sha(PREVIOUS_MANIFEST) == PREVIOUS_SHA
    m = json.loads(PREVIOUS_MANIFEST.read_text())
    relative = str(pathlib.Path(path).relative_to(ROOT))
    entry = m['files_relative_to_6_model_inference'].get(relative)
    expected = entry['sha256'] if entry else m['unchanged_frozen_dependencies'][relative]
    actual = sha(path)
    assert actual == expected, relative
    return actual


def config_and_transforms():
    from openpi.training.config import get_config
    from openpi import transforms as tr
    from openpi.shared import normalize
    cfg = get_config('pi05_libero')
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    norm = ROOT / 'weights/pi05_libero_bf16_values_r1/assets' / dc.asset_id / 'norm_stats.json'
    verified(norm)
    stats = normalize.load(norm.parent)
    inp = tr.compose([*dc.data_transforms.inputs, tr.Normalize(stats, use_quantiles=dc.use_quantile_norm),
                      *dc.model_transforms.inputs])
    out = tr.compose([*dc.model_transforms.outputs, tr.Unnormalize(stats, use_quantiles=dc.use_quantile_norm),
                      *dc.data_transforms.outputs])
    return cfg, inp, out


def prepare(record, transform):
    import jax
    from openpi.models.model import Observation
    raw = {'observation/image': record['image'], 'observation/wrist_image': record['wrist_image'],
           'observation/state': np.asarray(record['state'], dtype=np.float32), 'prompt': str(record['prompt'])}
    return Observation.from_dict(jax.tree.map(lambda a: torch.from_numpy(np.array(a))[None].cuda(), transform(raw)))


def tensor_inputs(observation):
    from openpi.models_pytorch.preprocessing_pytorch import preprocess_observation_pytorch
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    ob = preprocess_observation_pytorch(observation, train=False)
    images = list(ob.images.values())
    masks = list(ob.image_masks.values())
    assert len(images) == 3
    pad = torch.cat([*(m[:, None].expand(1, 256) for m in masks), ob.tokenized_prompt_mask], 1)
    pos = torch.cumsum(pad, 1) - 1
    attn = make_att_2d_masks(pad, torch.zeros_like(pad))[:, None]
    mask = torch.where(attn, 0.0, -2.3819763e38)
    suffix = torch.cat([pad[:, None, :].expand(1, 10, pad.shape[1]),
                        torch.ones(1, 10, 10, dtype=torch.bool, device='cuda')], 2)
    action_pos = pad.sum(-1)[:, None] + torch.arange(10, device='cuda')[None, :]
    action_mask = torch.where(suffix[:, None], 0.0, -2.3819763e38)
    return images, (ob.tokenized_prompt, pad, pos, mask), (action_pos, action_mask)


def time_table():
    from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
    t = torch.tensor(1., device='cuda', dtype=torch.float32)
    dt = torch.tensor(-.1, device='cuda', dtype=torch.float32)
    times, table = [], []
    while t >= -dt / 2:
        times.append(t.clone())
        table.append(create_sinusoidal_pos_embedding(t[None], 1024, 4e-3, 4., torch.device('cuda')).float())
        t += dt
    assert len(times) == 10
    return torch.stack(times), torch.cat(table), dt
