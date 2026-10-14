"""Frozen helper reuse; this version writes only its own run directory."""
import hashlib
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path('/work')
VERSION = 'trt_full4_20261007_r1'
RUN = ROOT / 'runs' / VERSION
SOURCE_SHA = '69ee787ec12921613feabf69b01348628b3d6327a8865cbc02ecb9be788d9258'
OLD = ROOT / 'runs/trt_t0_t2_20261007_r1'


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


def verified(path):
    manifest = ROOT / 'releases/trt_t3_20261007_r1/manifest.json'
    assert sha(manifest) == '016814fb9a9eaba7d398bb9fa7e2b3952de2995d8767b215c76ee7ed10bcbe17'
    d = json.loads(manifest.read_text())
    rel = str(pathlib.Path(path).relative_to(ROOT))
    entry = d['files_relative_to_6_model_inference'].get(rel)
    expected = entry['sha256'] if entry else d['unchanged_frozen_dependencies'][rel]
    assert sha(path) == expected, rel
    return expected


def frozen_module(name, path):
    verified(path)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


_helpers = frozen_module('full4_t3_helpers', ROOT / 'versions/trt_t3_20261007_r1/common.py')
config_and_transforms = _helpers.config_and_transforms
prepare = _helpers.prepare
tensor_inputs = _helpers.tensor_inputs
time_table = _helpers.time_table


def float_helpers():
    return frozen_module('full4_t0_helpers', ROOT / 'versions/trt_t0_t2_20261007_r1/common.py')


def wrappers(net):
    s = frozen_module('full4_subgraphs', ROOT / 'versions/trt_t0_t2_20261007_r1/subgraphs.py')
    return {'vision': s.Vision(net).eval(), 'action_step': s.ActionStep(net).eval()}


def runtime():
    return frozen_module('full4_t3_runtime', ROOT / 'versions/trt_t3_20261007_r1/runtime.py')
