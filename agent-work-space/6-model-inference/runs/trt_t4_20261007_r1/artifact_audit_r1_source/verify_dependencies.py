"""CPU-only inherited package/code audit before closed-loop inference."""
import argparse
import hashlib
import importlib.metadata
import json
import pathlib


def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    root=pathlib.Path('/work');manifest=root/'releases/trt_t3_20261007_r1/manifest.json'
    assert sha(manifest)=='016814fb9a9eaba7d398bb9fa7e2b3952de2995d8767b215c76ee7ed10bcbe17'
    spec=json.loads(manifest.read_text())
    path=root/'runs/trt_t3_20261007_r1/environment_r1/runtime.json'
    assert sha(path)==spec['files_relative_to_6_model_inference'][str(path.relative_to(root))]['sha256']
    old=json.loads(path.read_text());packages=dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions()))
    assert packages==old['packages']
    for name,entry in old['source_files'].items():
        p=pathlib.Path(name);assert sha(p)==entry['sha256'] and p.stat().st_size==entry['bytes'],name
    with (out/'verified.json').open('x') as f:json.dump({'passed':True,'same_packages_and_sources_as_T3':True,
        'source_files':old['source_files'],'packages':packages,'source_files_count':len(old['source_files']),
        'verification_process_did_not_load_GPU_model':True},f,indent=2)
    print('T4 INHERITED DEPENDENCIES VERIFIED',len(old['source_files']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);main(p.parse_args().out)
