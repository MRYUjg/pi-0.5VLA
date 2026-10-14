"""Verify inherited dependencies without editing the original frozen release."""
import argparse
import importlib.metadata
import json
import pathlib
import platform
import subprocess

from common import PREVIOUS, sha, verified, write_json


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    old_path = PREVIOUS / 'environment_freeze_r1/runtime.json'
    verified(old_path)
    old = json.loads(old_path.read_text())
    packages = dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()))
    assert packages == old['packages'], 'runtime packages changed'
    sources = {}
    for name, expected in old['source_files'].items():
        path = pathlib.Path(name)
        actual = {'sha256': sha(path), 'bytes': path.stat().st_size}
        assert actual == expected, name
        sources[name] = actual
    import torch
    import tensorrt
    record = {'python': platform.python_version(), 'platform': platform.platform(),
        'packages': packages, 'torch_import_version': torch.__version__,
        'torch_cuda': torch.version.cuda, 'TensorRT_import_version': tensorrt.__version__,
        'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                       '--format=csv,noheader'], text=True).strip(),
        'source_files': sources, 'same_dependencies_as_T0_T2': True,
        'previous_environment_sha256': sha(old_path),
        'repository_mounted_read_only': True, 'tokenizer_in_image': True,
        'source_weight_loaded_for_T3_inference': False, 'actual_4070_acceptance': False}
    write_json(out / 'runtime.json', record)
    with (out / 'packages.txt').open('x') as f:
        f.write('\n'.join(f'{k}=={v}' for k, v in packages.items()) + '\n')
    print('ENVIRONMENT VERIFIED', len(sources), 'unchanged dependency files', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=pathlib.Path, required=True)
    main(p.parse_args().out)
