"""Negative tests against an accepted real trajectory, isolated temporary copies."""
import argparse,json,pathlib,shutil,tempfile
import h5py
from audit import audit

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--episode',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();source=pathlib.Path(a.episode);results=[]
    cases=('stale_observation','wrong_action','wrong_gripper_state','blank_rgb','crashed_capture','replay_failure')
    for case in cases:
        with tempfile.TemporaryDirectory(prefix='collection_audit_') as d:
            ep=pathlib.Path(d)/source.name;ep.mkdir()
            shutil.copy2(source/'trajectory.h5',ep/'trajectory.h5')
            for name in ('teacher','teacher.status.json','replay.status.json'):(ep/name).symlink_to(source/name)
            shutil.copy2(source/'replay.json',ep/'replay.json')
            shutil.copy2(source.with_suffix('.status.json'),ep.with_suffix('.status.json'))
            if case=='crashed_capture':
                p=ep.with_suffix('.status.json');r=json.loads(p.read_text());r['normal_exit']=False;r['exit_code']=139;p.write_text(json.dumps(r))
            elif case=='replay_failure':
                p=ep/'replay.json';r=json.loads(p.read_text());r['pass']=False;p.write_text(json.dumps(r))
            else:
                with h5py.File(ep/'trajectory.h5','r+') as f:
                    if case=='stale_observation':f['observations/qpos'][10]=f['observations/qpos'][0]
                    if case=='wrong_action':f['transitions/actions'][10,0]=.99
                    if case=='wrong_gripper_state':f['observations/state'][10,6]=.25
                    if case=='blank_rgb':f['observations/image'][0]=0
            r=audit(ep);assert not r['accepted'],case
            results.append({'case':case,'rejected':True,'errors':r['errors']})
    with pathlib.Path(a.out).open('x') as f:json.dump({'pass':True,'tests':results},f,indent=2)
    print('Negative tests passed:',len(results),flush=True)
if __name__=='__main__':main()
