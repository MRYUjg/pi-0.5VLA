import argparse
import importlib.metadata
import json
import pathlib
import platform
import subprocess
import traceback

import numpy as np
import torch
import onnx
from onnx import helper, TensorProto

from common import load_model, prepare, selected_language, time_table, write_json, SOURCE, SOURCE_SHA, sha
from subgraphs import Vision, Prefix, ActionStep, inputs
from runtime import Engine, build, metrics


def probe(out):
    import tensorrt
    # A deterministic codec/parser test, never substituted for task calibration.
    q = (np.arange(256 * 256).reshape(256, 256) % 16 - 8).astype(np.int8)
    packed = (q.reshape(-1)[::2].astype(np.uint8) & 15) | ((q.reshape(-1)[1::2].astype(np.uint8) & 15) << 4)
    w = TensorProto(name='weight', data_type=TensorProto.INT4, dims=q.shape, raw_data=packed.tobytes())
    scale = torch.full((2, 256), .01, dtype=torch.bfloat16)
    s = TensorProto(name='scale', data_type=TensorProto.BFLOAT16, dims=scale.shape,
                    raw_data=scale.view(torch.int16).numpy().tobytes())
    nodes = [helper.make_node('DequantizeLinear', ['weight', 'scale'], ['w'], axis=0, block_size=128),
             helper.make_node('MatMul', ['x', 'w'], ['y'])]
    model = helper.make_model(helper.make_graph(nodes, 'int4_parser_probe',
        [helper.make_tensor_value_info('x', TensorProto.BFLOAT16, [1, 16, 256])],
        [helper.make_tensor_value_info('y', TensorProto.BFLOAT16, [1, 16, 256])], [w, s]),
        opset_imports=[helper.make_opsetid('', 21)], ir_version=10)
    path = out / 'int4_probe.onnx'
    onnx.save(model, path)
    onnx.checker.check_model(str(path))
    b = build(path, out / 'int4_probe.engine')
    x = torch.linspace(-1, 1, 4096, device='cuda').reshape(1, 16, 256).bfloat16()
    ref = x @ (torch.from_numpy(q).to('cuda').bfloat16() * scale.repeat_interleave(128, 0).to('cuda'))
    got, = Engine(out / 'int4_probe.engine')(x)
    result = metrics(ref, got)
    assert result['finite'] and result['max_abs'] < .02, result
    return {'tensorrt': tensorrt.__version__, 'opset': 21, 'block_size': 128,
            'signed_int4': True, 'zero_point': 0, 'symmetry': True, 'build': b, 'accuracy': result,
            'task_calibration': False}


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    packages = {}
    for name in ['torch', 'transformers', 'onnx', 'onnxruntime-gpu', 'nvidia-modelopt',
                 'tensorrt-cu12', 'cupy-cuda12x', 'safetensors', 'numpy', 'ml-dtypes']:
        packages[name] = importlib.metadata.version(name)
    write_json(out / 'environment.json', {'python': platform.python_version(), 'packages': packages,
        'gpu': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
        'cuda': torch.version.cuda, 'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version',
             '--format=csv,noheader'], text=True).strip(), 'actual_4070_validation': False})
    write_json(out / 'int4_probe.json', probe(out))
    net, cfg = load_model()
    chosen = selected_language(net)
    assert len(chosen) == 126, len(chosen)
    write_json(out / 'layer_scope.json', {'source_sha256': SOURCE_SHA, 'layers': {
        n: {'shape': list(m.weight.shape), 'dtype': str(m.weight.dtype), 'parameters': m.weight.numel()}
        for n, m in chosen.items()}, 'parameters': sum(m.weight.numel() for m in chosen.values())})
    rec_path = pathlib.Path('/reference/runs/observations_r2/task00_i0/initial_observation.npz')
    observation = prepare(dict(np.load(rec_path)), cfg)
    times, table, dt = time_table()
    with torch.inference_mode():
        images, pref, act = inputs(net, observation)
        vision = Vision(net).eval()
        image_checks = [metrics(net.paligemma_with_expert.embed_image(x), vision(x)) for x in images]
        prefix = Prefix(net).eval()
        result = prefix(*pref)
        _, cache = net.paligemma_with_expert.forward(attention_mask=pref[4], position_ids=pref[3],
                    inputs_embeds=[net.embed_prefix(*net._preprocess_observation(observation, train=False)[:4])[0], None], use_cache=True)
        cache_checks = [metrics(cache[i][j], result[1 + 2*i+j]) for i in range(18) for j in range(2)]
        torch.manual_seed(910000)
        noise = torch.randn(1, 10, 32, device='cuda', dtype=torch.float32)
        ref = net.sample_actions('cuda', observation, noise=noise)
        x = noise.clone()
        step = ActionStep(net).eval()
        cache_copy = tuple(k.clone() for k in result[1:])
        step_checks = []; step_inputs = []; step_outputs = []
        for i in range(10):
            v = step(x, table[i:i+1], *act, *result[1:])
            orig = net.denoise_step(observation.state, pref[2], cache, x, times[i:i+1])
            step_inputs.append(x.cpu());step_outputs.append(orig.cpu())
            step_checks.append(metrics(orig, v))
            x = x + dt * v
        cache_immutable = all(torch.equal(a, b) for a, b in zip(cache_copy, result[1:]))
        final = metrics(ref, x)
        passed = all(c['max_abs'] == 0 for c in image_checks + cache_checks + step_checks) and final['max_abs'] == 0 and cache_immutable
        torch.save({'vision': images[0].cpu(), 'all_images': [v.cpu() for v in images],
            'prefix': tuple(v.cpu() for v in pref), 'action': (noise.cpu(), table[:1].cpu(), *(v.cpu() for v in act), *(v.cpu() for v in result[1:])),
            'times': times.cpu(), 'time_embeddings': table.cpu(), 'noise': noise.cpu(), 'float_actions': ref.cpu(),
            'step_inputs': step_inputs, 'step_outputs': step_outputs,
            'prefix_outputs': tuple(v.cpu() for v in result)}, out / 'reference_inputs.pt')
    write_json(out / 'contract.json', {'passed': passed, 'source_sha256': sha(SOURCE / 'model.safetensors'),
        'observation_sha256': sha(rec_path), 'image_slots': 3, 'real_cameras': 2, 'placeholder_mask': False,
        'prefix_length': pref[2].shape[1], 'token_length': pref[1].shape[1],
        'prefix_kv': [{'shape': list(t.shape), 'dtype': str(t.dtype)} for t in result[1:]],
        'state_interface_dim': 8, 'state_conditioning': False, 'noise_seed': 910000,
        'time_sequence': times.tolist(), 'time_embedding_source': 'original FP64 helper and repeated FP32 time update',
        'image_checks': image_checks, 'cache_checks': cache_checks, 'step_checks': step_checks,
        'final_actions': final, 'prefix_cache_immutable': cache_immutable})
    assert passed, 'Pure subgraph contract failed; see contract.json'
    print('T0 PASS', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=pathlib.Path, required=True); a = p.parse_args()
    try:
        main(a.out)
    except Exception:
        a.out.mkdir(parents=True, exist_ok=True)
        write_json(a.out / 'failure.json', {'status': 'failed', 'traceback': traceback.format_exc()})
        raise
