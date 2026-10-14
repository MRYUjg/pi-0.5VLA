"""Independent accounting, trace/action checks, video decoding and paired scene audit."""
import argparse
import collections
import hashlib
import json
import math
import pathlib
import subprocess

import h5py
import imageio_ffmpeg
import numpy as np

V = pathlib.Path(__file__).resolve().parents[1]
ROOT = V.parents[1]


def wilson(k,n):
    if not n: return None
    z=1.959963984540054
    p=k/n; den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [mid-radius,mid+radius]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
    out=ROOT/'runs'/a.run
    assert pathlib.Path(a.run).name==a.run
    protocol=json.loads((V/'configs/protocol.json').read_text())
    seeds=json.loads((V/'configs/test_seeds.json').read_text())['tasks']
    errors=[]; videos=[]; allrows={}; conditions={}
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    for condition in protocol['conditions']:
        task_results={}; rows=[]; latencies=[]
        for task,ss in seeds.items():
            rr=[]
            for i,seed in enumerate(ss):
                episode=out/condition/f'task{int(task):02d}_s{seed}'
                if not (episode/'execution.json').exists():
                    errors.append('missing execution: '+str(episode));continue
                execution=json.loads((episode/'execution.json').read_text())
                result=execution.get('result',{})
                r={'condition':condition,'task':int(task),'seed':seed,'process_exit':execution['process_exit'],**result}
                rr.append(r);rows.append(r);allrows[(condition,int(task),seed)]=r
                if not result:continue
                if result['seed']!=seed or result['task_id']!=int(task) or result['condition']!=condition:
                    errors.append('identity mismatch: '+str(episode))
                if execution['process_exit']!=0:continue
                steps=[json.loads(line) for line in (episode/'steps.jsonl').read_text().splitlines()]
                native=[x['step'] for x in steps if x['native_success'] and x['step']<=220]
                valid=bool(result['initialization_pass'] and native and all(x['finite'] and not any(x['physics_warnings']) for x in steps if x['step']<=native[0]))
                if valid!=result['primary_success']:errors.append('primary success accounting: '+str(episode))
                if len(steps)!=result['steps']:errors.append('step count mismatch: '+str(episode))
                if len(steps)>220 and i!=0:errors.append('undeclared diagnostic extension: '+str(episode))
                with h5py.File(episode/'trajectory.h5','r') as f:
                    if f['preparation_actions'].shape!=(120,7):errors.append('preparation count: '+str(episode))
                    queries=f.get('queries',{})
                    if len(queries)!=result.get('queries',0):errors.append('query count: '+str(episode))
                    for q,g in queries.items():
                        actions=g['actions'][...];start=int(g.attrs['step'])
                        if start<220:latencies.append(float(g.attrs['latency_seconds']))
                        if actions.shape!=(10,7) or not np.isfinite(actions).all():errors.append('invalid action: '+str(episode))
                        for offset in range(min(5,len(steps)-start)):
                            x=steps[start+offset]
                            if not np.array_equal(actions[offset],x['raw_action']) or not np.array_equal(np.clip(actions[offset],-1,1),x['executed_action']):
                                errors.append('action execution mismatch: '+str(episode))
                        if g['image'].shape!=(256,256,3) or g['wrist_image'].shape!=(256,256,3):errors.append('image shape: '+str(episode))
                if i==0:
                    mp4=episode/'rollout.mp4'
                    if not mp4.exists():errors.append('missing video: '+str(mp4));continue
                    frames,size=imageio_ffmpeg.count_frames_and_secs(str(mp4))
                    decode=subprocess.run([ffmpeg,'-v','error','-i',str(mp4),'-f','null','-'],capture_output=True)
                    sha=hashlib.sha256(mp4.read_bytes()).hexdigest()
                    ok=decode.returncode==0 and frames==result['video_frames'] and sha==result['video_sha256']
                    if not ok:errors.append('video decode/frame/hash: '+str(mp4))
                    videos.append({'condition':condition,'task':int(task),'seed':seed,'path':str(mp4.relative_to(ROOT)),
                                   'primary_success':result['primary_success'],'extended_success':result['native_success_extended'],
                                   'initialization_pass':result['initialization_pass'],'failure':result['failure'],
                                   'frames':frames,'seconds':size,'sha256':sha,'decode_pass':ok})
            k=sum(r['process_exit']==0 and r.get('primary_success',False) for r in rr)
            task_results[task]={'completed':len(rr),'planned':20,'successes':k,'success_rate':k/20,'wilson_95':wilson(k,20),
                                'failure_categories':dict(collections.Counter(r.get('failure') or ('none' if r.get('primary_success') else 'process_failure') for r in rr)),
                                'initialization_pass_count':sum(r.get('initialization_pass',False) for r in rr)}
        k=sum(r['process_exit']==0 and r.get('primary_success',False) for r in rows)
        groups={}
        for label,tasks in [('trained',[0,2,3,5,6]),('untrained',[1,4,7,8,9])]:
            group=[r for r in rows if r['task'] in tasks]
            num=sum(r['process_exit']==0 and r.get('primary_success',False) for r in group)
            groups[label]={'tasks':tasks,'successes':num,'planned':100,'rate':num/100}
        conditions[condition]={'completed':len(rows),'planned':200,'successes':k,'success_rate':k/200,'macro_average':sum(t['success_rate'] for t in task_results.values())/10,
                               'wilson_95':wilson(k,200),'acceptance_pass':len(rows)==200 and k>=180,'tasks':task_results,'groups':groups,
                               'process_failures':sum(r['process_exit']!=0 for r in rows),
                               'inference_request_latency_seconds':{'count':len(latencies),'mean':float(np.mean(latencies)) if latencies else None,
                                                                    'p95':float(np.quantile(latencies,.95)) if latencies else None,
                                                                    'includes_queue_wait':True},
                               'successful_completion_steps':[r['first_success_step'] for r in rows if r.get('primary_success')],
                               'clipped_action_frames':sum(r.get('clipped_action_frames',0) for r in rows),
                               'robot_object_penetration_over_2mm_episodes':sum(r.get('execution_contacts',{}).get('max_robot_object_penetration_mm',0)>2 for r in rows)}
    paired={}
    for condition in ['panda_base','eco65_zhixing_base','eco65_panda_base']:
        common=[];same=[];lost=[]
        for t,ss in seeds.items():
            for s in ss:
                left=allrows.get(('eco65_zhixing_lora',int(t),s),{})
                right=allrows.get((condition,int(t),s),{})
                if left.get('reset_object_hash') and right.get('reset_object_hash'):
                    common.append((int(t),s))
                    (same if left['reset_object_hash']==right['reset_object_hash'] else lost).append((int(t),s))
        paired[condition]={'paired_reset_available':len(common),'object_initial_hash_equal':len(same),'mismatches':lost,
                           'note':'same native reset(seed), no scene state injection; geometric camera/robot differs intentionally'}
    report={'verification_pass':not errors,'errors':errors,'complete':all(c['completed']==200 for c in conditions.values()),
            'conditions':conditions,'paired_native_scenes':paired,'videos':videos,
            'scope':'220-step primary; preparation120; first trial650 secondary; not official LIBERO init-state .pruned benchmark; not full teacher r6 physical certification',
            'test_set_consumed':True,'test_outcomes_must_not_be_used_to_tune_this_candidate':True}
    with (out/'verified_results.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('VERIFICATION',report['verification_pass'],'COMPLETE',report['complete'],'VIDEOS',len(videos),flush=True)
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
