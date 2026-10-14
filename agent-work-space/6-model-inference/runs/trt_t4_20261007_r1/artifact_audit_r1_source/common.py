"""T4 reuses the frozen T3 transforms, runtime and T0 floating reference."""
import hashlib
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path('/work')
VERSION = 'trt_t4_20261007_r1'
PREVIOUS = ROOT / 'releases/trt_t3_20261007_r1/manifest.json'
PREVIOUS_SHA = '016814fb9a9eaba7d398bb9fa7e2b3952de2995d8767b215c76ee7ed10bcbe17'


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def verified(path):
    assert sha(PREVIOUS) == PREVIOUS_SHA
    d = json.loads(PREVIOUS.read_text())
    name = str(pathlib.Path(path).relative_to(ROOT))
    entry = d['files_relative_to_6_model_inference'].get(name)
    expected = entry['sha256'] if entry else d['unchanged_frozen_dependencies'][name]
    assert sha(path) == expected, name
    return expected


def frozen_module(name, path):
    verified(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def transforms_and_prepare():
    h = frozen_module('t4_frozen_t3_helpers', ROOT / 'versions/trt_t3_20261007_r1/common.py')
    return h


# The frozen runtime imports these names from common. Only helpers are loaded;
# this does not construct a policy, instantiate a source model or load weights.
_helpers = transforms_and_prepare()
config_and_transforms = _helpers.config_and_transforms
prepare = _helpers.prepare
tensor_inputs = _helpers.tensor_inputs
time_table = _helpers.time_table
