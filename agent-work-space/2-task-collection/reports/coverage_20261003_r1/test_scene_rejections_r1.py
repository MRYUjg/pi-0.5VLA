"""Verify coverage audit rejects forged initialization metadata; frozen audit is unchanged."""
import argparse,json,pathlib,shutil,sys,tempfile
import h5py
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'versions/coverage_20261003_r1/scripts'))
from audit import audit

def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    src=pathlib.Path(args.episode)
    baseline=json.loads((src/'audit.json').read_text())
    assert baseline['accepted'], 'Need accepted positive control'
    cases={'scene':'registered scene mismatch','manifest':'scene manifest mismatch','penetration':'initialization non-support penetration >2mm','drift':'target drift before teacher >5mm'}
    results=[]
    for case,expected in cases.items():
        with tempfile.TemporaryDirectory(prefix='coverage_audit_') as d:
            ep=pathlib.Path(d)/src.name;ep.mkdir()
            shutil.copy2(src/'trajectory.h5',ep/'trajectory.h5')
            for name in ('teacher','teacher.status.json','replay.status.json','replay.json'):(ep/name).symlink_to(src/name)
            shutil.copy2(src.with_suffix('.status.json'),ep.with_suffix('.status.json'))
            with h5py.File(ep/'trajectory.h5','r+') as f:
                if case=='scene':
                    v=json.loads(f.attrs['coverage_scene']);v['scene']['split']='test';f.attrs['coverage_scene']=json.dumps(v)
                elif case=='manifest':f.attrs['scene_manifest_sha256']='0'*64
                elif case=='penetration':f.attrs['preparation_non_support_penetration_mm']=3.
                else:
                    v=json.loads(f.attrs['settled_object_states']);v['alphabet_soup_1']['position'][0]+=.02;f.attrs['settled_object_states']=json.dumps(v)
            r=audit(ep)
            assert not r['accepted'] and expected in r['errors'],(case,r['errors'])
            results.append({'case':case,'expected_error':expected,'errors':r['errors'],'rejected':True})
    with pathlib.Path(args.out).open('x') as f:json.dump({'pass':True,'positive_control':str(src),'tests':results},f,indent=2)
    print('Scene negative tests passed:',len(results))
if __name__=='__main__':main()
