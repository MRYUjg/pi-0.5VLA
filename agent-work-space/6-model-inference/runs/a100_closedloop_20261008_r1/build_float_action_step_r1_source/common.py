"""Reuse frozen sampling, transforms and audits with new A100 engines only."""
import hashlib
import importlib.util
import json
import pathlib
import sys

ROOT=pathlib.Path('/work');VERSION='a100_closedloop_20261008_r1';RUN=ROOT/'runs'/VERSION
SOURCE_SHA='69ee787ec12921613feabf69b01348628b3d6327a8865cbc02ecb9be788d9258'
_entries=None


def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write_json(p,d):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')


def verified(p):
    global _entries
    if _entries is None:
        _entries={}
        for version,expected in [('trt_full4_20261007_r1','63b7561e7be672d83069c3c883a0142efee583f2a26088d0e4e94768ad325cfc'),
                                 ('trt_t4_20261007_r1','b9875e01fb7820d4c1e49b289421e3daa130ebe3b309001d374198232375c1c0')]:
            m=ROOT/'releases'/version/'manifest.json';assert sha(m)==expected
            d=json.loads(m.read_text());e=dict(d['unchanged_frozen_dependencies'])
            e.update({k:v['sha256'] for k,v in d['files_relative_to_6_model_inference'].items()})
            for k,v in e.items():
                assert k not in _entries or _entries[k]==v,k
                _entries[k]=v
    p=pathlib.Path(p);rel=str(p.relative_to(ROOT));expected=_entries[rel]
    assert sha(p)==expected,rel
    return expected


def frozen_module(name,p):
    verified(p);s=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m


_helper=frozen_module('a100_frozen_t3_helpers',ROOT/'versions/trt_t3_20261007_r1/common.py')
config_and_transforms=_helper.config_and_transforms
prepare=_helper.prepare
tensor_inputs=_helper.tensor_inputs
time_table=_helper.time_table
