"""Read-only source checks and immutable output helpers."""
import hashlib,json,pathlib
import h5py,numpy as np
VERSION=pathlib.Path(__file__).resolve().parents[1]
COLLECTION=VERSION.parents[1]
BATCH=COLLECTION/'runs/expansion_train_val_r1'
COMMIT='0cf864870cf29f4738d3ade893e6fd13fbd7cdb5'
REPO_IDS={s:f'local/eco65_zhixing_can_expansion_r1_{s}' for s in ('train','validation')}
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,v):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
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
    master=read(BATCH/'dataset_manifest.json');protocol=read(BATCH/'protocol.json')
    require(master['attempts']==master['accepted']==16,'expected complete accepted source batch')
    selected={};seen=set();fingerprints={}
    for split in REPO_IDS:
        p=BATCH/(split+'_manifest.json');fingerprints[str(p.relative_to(COLLECTION))]=sha(p)
        records=read(p)['episodes'];check_records(records,split,seen);selected[split]=records
        for r in records:
            ep=BATCH/('can_s'+str(r['seed']));audit=read(ep/'audit.json')
            require(audit['accepted'] and audit['stats']==r['stats'],'audit/manifest mismatch')
            require(read(ep.with_suffix('.status.json'))['exit_code']==0,'capture exit')
            require(read(ep/'replay.status.json')['exit_code']==0 and read(ep/'replay.json')['pass'],'replay exit')
            for name in ('trajectory.h5','capture.json','replay.json','teacher/result.json'):
                actual=sha(ep/name);require(actual==audit['files'][name],'source hash mismatch: '+str(ep/name));fingerprints[str((ep/name).relative_to(COLLECTION))]=actual
            fingerprints[str((ep/'audit.json').relative_to(COLLECTION))]=sha(ep/'audit.json')
            with h5py.File(ep/'trajectory.h5','r') as f:
                n=len(f['transitions/actions']);require(n==r['stats']['frames'],'source frame count')
                require(f['observations/state'].shape==(n+1,8),'source state shape')
                require(f['transitions/actions'].shape==(n,7),'source action shape')
                require(json.loads(f.attrs['coverage_scene'])['scene']==r['stats']['scene'],'source scene mismatch')
                require(np.allclose(np.diff(f['observations/time']),.05,rtol=0,atol=1e-9),'source not 20Hz')
    require(seen==set(protocol['seeds']),'split coverage mismatch')
    require(len(selected['train'])==12 and len(selected['validation'])==4,'split count')
    for n in ('dataset_manifest.json','protocol.json'):fingerprints[str((BATCH/n).relative_to(COLLECTION))]=sha(BATCH/n)
    return selected,fingerprints
