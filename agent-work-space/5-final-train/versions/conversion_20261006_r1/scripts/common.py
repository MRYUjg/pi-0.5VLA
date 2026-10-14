"""Only admitted formal demonstrations; deterministic split and immutable provenance."""
import hashlib
import json
import pathlib
import h5py
import numpy as np
VERSION=pathlib.Path(__file__).resolve().parents[1]
COLLECTION=VERSION.parents[1]
BATCH=COLLECTION/'runs/formal_merged_20261006_r1'
COMMIT='0cf864870cf29f4738d3ade893e6fd13fbd7cdb5'
REPO_IDS={s:f'local/eco65_zhixing_spatial_formal_r2_{s}' for s in ('train','validation')}
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,v):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def require(ok,msg):
    if not ok:raise ValueError(msg)
def check_records(records,split,seen):
    for r in records:
        require(r['accepted'],'source not accepted')
        scene=r['stats']['scene'];require(scene['split']==split,'split mismatch')
        require(scene['seed']==r['seed'],'scene seed mismatch')
        require(r['seed'] not in seen,'duplicate or cross-split seed');seen.add(r['seed'])
def inspect_sources():
    summary=read(BATCH/'summary.json');protocol=read(BATCH/'protocol.json')
    require(summary['pass'],'formal source release gate failed')
    require(not summary['duplicate_initializations'],'duplicate starting scenes')
    registered={s['seed']:s for s in protocol['jobs']}
    test=read(COLLECTION/'versions/collection_20261006_r1/configs/test_seeds.json')
    held_out={s for seeds in test['tasks'].values() for s in seeds}
    selected={s:[] for s in REPO_IDS};fingerprints={};seen=set();counts={}
    require(len(summary['records'])==len(registered),'attempt ledger incomplete')
    for rel in summary['records']:
        ep=COLLECTION/rel;attempt=read(ep.with_suffix('.summary.json'))
        require(attempt['scene']==registered[attempt['scene']['seed']],'attempt not registered')
        if not attempt['accepted']:continue
        a=read(ep/'audit.json');s=a['scene'];split=s['split']
        require(a['accepted'] and a['integrity_pass'] and a['initialization_pass'] and a['teacher_success'],'source admission failed')
        require(a['physical_audit']['audit_pass'] and not a['errors'],'physical/structural source gate failed')
        require(s==registered[a['seed']] and s['seed'] not in held_out,'source split or reserved test leakage')
        require(s['task_id'] in summary['eligible_tasks'],'unqualified task')
        require(split in selected,'non-training split')
        require(read(ep.with_suffix('.status.json'))['exit_code']==0,'capture process exit')
        require(read(ep/'audit.status.json')['exit_code']==0,'audit process exit')
        require(read(ep/'replay.status.json')['exit_code']==0 and read(ep/'replay.json')['pass'],'replay process/result')
        for name,want in a['files'].items():
            require(sha(ep/name)==want,'source file mismatch: '+str(ep/name))
            fingerprints[str((ep/name).relative_to(COLLECTION))]=want
        for p in (ep/'audit.json',ep/'audit.status.json',ep.with_suffix('.status.json'),ep.with_suffix('.summary.json')):
            fingerprints[str(p.relative_to(COLLECTION))]=sha(p)
        with h5py.File(ep/'trajectory.h5','r') as f:
            n=len(f['transitions/actions'])
            require(n==a['frames'] and f['observations/state'].shape==(n+1,8),'source T+1 state contract')
            require(f['transitions/actions'].shape==(n,7),'source action contract')
            require(json.loads(f.attrs['coverage_scene'])['scene']==s,'source scene metadata')
            require(np.allclose(np.diff(f['observations/time']),.05,rtol=0,atol=1e-9),'source 20Hz contract')
        record={'accepted':True,'seed':s['seed'],'source_relative':rel,'stats':{'scene':s,'frames':n}}
        check_records([record],split,seen);selected[split].append(record)
        key=(s['task_id'],split);counts[key]=counts.get(key,0)+1
    for tid in summary['eligible_tasks']:
        require(counts.get((tid,'train'),0)>=20 and counts.get((tid,'validation'),0)>=4,'insufficient per-task admitted data')
    for split in selected:selected[split].sort(key=lambda r:(r['stats']['scene']['task_id'],r['seed']))
    for p in (BATCH/'summary.json',BATCH/'protocol.json'):
        fingerprints[str(p.relative_to(COLLECTION))]=sha(p)
    return selected,fingerprints
