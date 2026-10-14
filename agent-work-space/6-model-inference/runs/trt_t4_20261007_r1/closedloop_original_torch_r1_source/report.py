"""Report matched development trials without relaxing earlier failed gates."""
import argparse
import hashlib
import json
import pathlib

import numpy as np

VARIANTS=['original_torch','trt_float','awq','gptq']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    x=np.asarray(values,float)
    return {'count':len(x),'mean':float(x.mean()),'p50':float(np.quantile(x,.5)),
        'p95':float(np.quantile(x,.95)),'p99':float(np.quantile(x,.99)),'max':float(x.max())}


def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    margin=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(center-margin),float(center+margin)]


def quality(a,b):
    # These definitions and limits are identical to the frozen T3 compare.py.
    d=np.abs(a[:,:,:6]-b[:,:,:6])/2
    confident=np.abs(a[:,:,6])>.2
    disagree=int((np.sign(a[:,:,6][confident])!=np.sign(b[:,:,6][confident])).sum())
    clip_a=(np.abs(a)>1).sum(axis=(0,1));clip_b=(np.abs(b)>1).sum(axis=(0,1))
    obj={'motion_mae':float(d.mean()),'motion_p99_abs':float(np.quantile(d,.99)),
        'motion_max_abs':float(d.max()),'gripper_disagreements':disagree,
        'confident_gripper_count':int(confident.sum()),'gripper_disagreement_fraction':disagree/max(1,int(confident.sum())),
        'clipping_rate_increase':float((clip_b.sum()-clip_a.sum())/a.size),
        'reference_clips_per_dimension':clip_a.tolist(),'candidate_clips_per_dimension':clip_b.tolist()}
    obj['original_gates_passed']=obj['motion_mae']<=.01 and obj['motion_p99_abs']<=.05 and \
        obj['gripper_disagreement_fraction']<=.005 and obj['clipping_rate_increase']<=.01
    return obj


def main(root,out):
    out.mkdir(exist_ok=False)
    summaries={};cases={};shared={}
    for variant in VARIANTS:
        resource=root/f'resource_{variant}_r1'
        benchmark=json.loads((resource/'benchmark/summary.json').read_text())
        memory=json.loads((resource/'memory_summary.json').read_text())
        assert benchmark['completed'] and benchmark['replay_passed'] and benchmark['repeat_passed']
        assert memory['isolated_compute_process'] and memory['no_sustained_growth_screen_passed'] and not memory['OOM']
        rows=[json.loads(line) for line in (resource/'benchmark/requests.jsonl').read_text().splitlines()[:128]]
        shared[variant]=np.asarray([r['actions'] for r in rows])
        folder=root/f'closedloop_{variant}_r1'
        exit_report=json.loads((folder/'exit_summary.json').read_text());assert exit_report['all_tasks_exit_zero']
        videos=[];eps=[];timings=[];clip=[];per_task=[]
        for task in range(10):
            n=0
            for index in [0,1,2]:
                path=folder/f'task{task:02d}/init{index:02d}.json'
                body=json.loads(path.read_text());ep=body['episode']
                assert (ep['variant'],ep['task'],ep['init_index'],ep['seed'])==(variant,task,index,940000+task*100+index)
                video=path.parent/ep['video'];assert sha(video)==ep['video_sha256']
                assert ep['video_frames']==ep['steps']+1
                assert len(body['steps'])==ep['steps'] and len(body['chunks'])==ep['requests']
                for step in body['steps']:
                    raw=np.asarray(step['raw_action']);executed=np.asarray(step['executed_action'])
                    assert raw.shape==(7,) and np.isfinite(raw).all()
                    assert np.array_equal(np.clip(raw,-1,1),executed)
                key=(task,index);cases[(variant,*key)]=ep;eps.append(ep);n+=int(ep['success'])
                timings += [r['timing'] for r in body['chunks']]
                clip += [r['clipped_dimensions'] for r in body['steps']]
                videos.append({'task':task,'init_index':index,'path':str(video),'bytes':video.stat().st_size,'sha256':ep['video_sha256']})
            per_task.append({'task':task,'successes':n,'trials':3,'fraction':n/3})
        k=sum(int(ep['success']) for ep in eps)
        summaries[variant]={'successes':k,'trials':30,'success_fraction':k/30,
            'Wilson_95_percent_CI_descriptive':wilson(k,30),'per_task':per_task,'episodes':eps,
            'resource_benchmark':benchmark,'resource_memory':memory,
            'closedloop_memory':json.loads((folder/'memory_summary.json').read_text()),
            'closedloop_latency_ms':{key:stats([r[key] for r in timings]) for key in [
                'image_state_preparation_and_encoding_ms','client_round_trip_ms',
                'observation_ready_to_action_ms','camera_step_to_action_ms']},
            'executed_raw_clip_fraction':float(np.asarray(clip,bool).mean()),
            'clipping_definition':'actual executed controls, all seven dimensions before np.clip; different trajectory inputs',
            'videos':videos,'all_video_first_frames_decoded_by_collectors':True}
    mismatches=[]
    for task in range(10):
        for index in [0,1,2]:
            baseline=cases[('original_torch',task,index)]
            for variant in VARIANTS[1:]:
                ep=cases[(variant,task,index)]
                for key in ['simulation_state_sha256','initial_observation_sha256','xml_sha256','controller_sha256']:
                    if ep['initialization'][key]!=baseline['initialization'][key]:
                        mismatches.append({'task':task,'init_index':index,'variant':variant,'field':key})
    for variant in VARIANTS:
        lost=[];recovered=[]
        for task in range(10):
            for index in [0,1,2]:
                a=cases[('trt_float',task,index)]['success'];b=cases[(variant,task,index)]['success']
                if a and not b:lost.append({'task':task,'init_index':index})
                if b and not a:recovered.append({'task':task,'init_index':index})
        s=summaries[variant]
        s['paired_vs_TensorRT_float']={'lost':lost,'recovered':recovered,'net_loss':len(lost)-len(recovered)}
        s['closedloop_original_gate_passed']=s['successes']>=27 and len(lost)-len(recovered)<=1 and not mismatches
        s['same_observation_offline_vs_TensorRT_float']=quality(shared['trt_float'],shared[variant])
        s['same_observation_cumulative_vs_original_Torch']=quality(shared['original_torch'],shared[variant])
        s['provisional_camera_to_dispatch_p99_200ms_passed']=s['closedloop_latency_ms']['camera_step_to_action_ms']['p99']<=200
    result={'version':'trt_t4_20261007_r1','all_120_episodes_complete':True,'initial_condition_alignment_passed':not mismatches,
        'initial_condition_mismatches':mismatches,'variants':summaries,'T1_original_gate_failed_temporarily_allowed':True,
        'GPTQ_T3_quality_failure_not_waived':True,'actual_4070_accepted':False,'formal_200_accepted':False,
        'two_hour_continuous_test_completed':False,'robot_accepted':False,
        'same_backend_isolates_quantization_error':True,'different_closedloop_trajectories_not_scored_as_same_input_quantization_error':True}
    for variant in ['awq','gptq']:
        s=summaries[variant]
        s['T4_quality_candidate_qualified']=s['closedloop_original_gate_passed'] and \
            s['same_observation_offline_vs_TensorRT_float']['original_gates_passed'] and \
            s['same_observation_cumulative_vs_original_Torch']['original_gates_passed']
    with (out/'summary.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    lines=['variant,task,init_index,seed,success,steps,requests,video']
    for variant in VARIANTS:
        for ep in summaries[variant]['episodes']:
            lines.append(','.join(str(x) for x in [variant,ep['task'],ep['init_index'],ep['seed'],
                int(ep['success']),ep['steps'],ep['requests'],ep['video']]))
    with (out/'episodes.csv').open('x') as f:f.write('\n'.join(lines)+'\n')
    print('T4 REPORT', {v:(summaries[v]['successes'],summaries[v]['closedloop_original_gate_passed']) for v in VARIANTS},flush=True)
    assert not mismatches,mismatches


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True)
    a=p.parse_args();main(a.root,a.out)
