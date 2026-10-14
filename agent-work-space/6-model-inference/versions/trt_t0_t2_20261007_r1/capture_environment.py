"""Record the isolated runtime and read-only source dependencies before freezing."""
import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import subprocess


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    import torch
    import tensorrt
    import transformers
    import modelopt
    roots = [pathlib.Path('/app/src'), pathlib.Path('/app/packages/openpi-client/src'),
             pathlib.Path(transformers.__file__).parent / 'models/gemma',
             pathlib.Path(transformers.__file__).parent / 'models/siglip',
             pathlib.Path(transformers.__file__).parent / 'models/paligemma',
             pathlib.Path(modelopt.__file__).parent / 'torch/quantization']
    paths = {p for root in roots for p in root.rglob('*.py') if p.is_file()}
    tokenizer = pathlib.Path('/openpi_assets/big_vision/paligemma_tokenizer.model')
    paths.add(tokenizer)
    record = {
        'python': platform.python_version(), 'platform': platform.platform(),
        'packages': dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions())),
        'torch_import_version': torch.__version__, 'torch_cuda': torch.version.cuda,
        'TensorRT_import_version': tensorrt.__version__,
        'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                       '--format=csv,noheader'], text=True).strip(),
        'source_files': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(paths)},
        'tokenizer_in_image': True, 'repository_mounted_read_only': True,
        'calibration_teacher_shutdown': True,
        'ModelOpt_CUDA_extension': 'not loaded; official calibration fallback used',
        'actual_4070_acceptance': False,
    }
    (out / 'runtime.json').write_text(json.dumps(record, indent=2) + '\n')
    (out / 'packages.txt').write_text('\n'.join(f'{k}=={v}' for k, v in record['packages'].items()) + '\n')
    print('ENVIRONMENT RECORDED', len(paths), 'dependency files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=pathlib.Path, required=True)
    main(parser.parse_args().out)
