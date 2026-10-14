"""Freeze only this new version; verify referenced old release unchanged."""
import datetime
import hashlib
import json
import pathlib

ROOT=pathlib.Path('/mnt/data/vla-policy/openpi/agent-work-space/6-model-inference')
VERSION='trt_full4_20261007_r1'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(p,d):
    with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    release=ROOT/'releases'/VERSION;release.mkdir(exist_ok=False)
    old=ROOT/'releases/trt_t3_20261007_r1/manifest.json'
    assert sha(old)=='016814fb9a9eaba7d398bb9fa7e2b3952de2995d8767b215c76ee7ed10bcbe17'
    d=json.loads(old.read_text());deps=dict(d['unchanged_frozen_dependencies'])
    deps.update({k:v['sha256'] for k,v in d['files_relative_to_6_model_inference'].items()})
    deps[str(old.relative_to(ROOT))]=sha(old)
    t4=ROOT/'releases/trt_t4_20261007_r1/manifest.json'
    assert sha(t4)=='b9875e01fb7820d4c1e49b289421e3daa130ebe3b309001d374198232375c1c0'
    deps[str(t4.relative_to(ROOT))]=sha(t4)
    k='versions/trt_t4_20261007_r1/memory.py'
    deps[k]=json.loads(t4.read_text())['files_relative_to_6_model_inference'][k]['sha256']
    for i,(k,expected) in enumerate(deps.items()):
        assert sha(ROOT/k)==expected,('changed old dependency',k)
        if (i+1)%1000==0:print('OLD VERIFIED',i+1,flush=True)
    paths=[]
    for folder in [ROOT/'versions'/VERSION,ROOT/'runs'/VERSION]:
        paths.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.append(ROOT/'当前实施索引_20261007_r6.md')
    files={str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(paths)}
    comparisons=json.loads((ROOT/'runs'/VERSION/'comparison.json').read_text())
    acceptance={'451_reachable_Linear_INT4_export_and_execution':True,
        'all_parameters_4bit':False,'quality_vs_TRT_float':{v:r['benchmark']['quality_vs_frozen_TRT_float']['original_gates_passed'] for v,r in comparisons['variants'].items()},
        'performance_isolated':{v:r['performance_measurement_isolated'] for v,r in comparisons['variants'].items()},
        '20Hz':False,'LIBERO_closedloop_success_rate':False,'two_hour':False,'formal_200':False,
        'actual_8GB_hardware':False,'frozen_files_do_not_imply_deployment_acceptance':True}
    manifest={'version':VERSION,'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status':'quick_offline_comparison_frozen','files_relative_to_6_model_inference':files,
        'unchanged_frozen_dependencies':deps,'previous_release_manifest_sha256':sha(old),
        'acceptance':acceptance,'manifest_excludes_itself_and_verification_receipts':True}
    write(release/'manifest.json',manifest)
    bad=[]
    for k,e in files.items():
        if sha(ROOT/k)!=e['sha256']:bad.append(k)
    for k,e in deps.items():
        if sha(ROOT/k)!=e:bad.append(k)
    write(release/'server_verification.json',{'manifest_sha256':sha(release/'manifest.json'),
        'new_files_checked':len(files),'unchanged_dependencies_checked':len(deps),
        'mismatches':bad,'scope':'all manifest new files and unchanged dependencies on server'})
    assert not bad,bad
    for p in paths:p.chmod(0o444)
    for p in release.iterdir():p.chmod(0o444)
    print('FROZEN',len(files),len(deps),sha(release/'manifest.json'),flush=True)


if __name__=='__main__':main()
