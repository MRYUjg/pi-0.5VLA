"""Actual A100 success counts with matched initial conditions and original gates."""
import argparse,json,pathlib
import numpy as np
from common import ROOT,sha,write_json,frozen_module

VARIANTS=['trt_float','awq','gptq','full_awq','full_gptq']


def stats(values):
    a=np.asarray(values,float)
    return {'count':len(a),'mean':float(a.mean()),'p50':float(np.quantile(a,.5)),
            'p95':float(np.quantile(a,.95)),'p99':float(np.quantile(a,.99)),'max':float(a.max())}


def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;delta=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(mid-delta),float(mid+delta)]


def main(root,out):
    out.mkdir(exist_ok=False)
    assert json.loads((root/'simulation_alignment_r1.json').read_text())['passed']
    assert json.loads((root/'dependencies_r1/verified.json').read_text())['passed']
    assert json.loads((root/'artifact_audit_r1/verified.json').read_text())['passed']
    metrics=frozen_module('a100_frozen_quality',ROOT/'versions/trt_t3_20261007_r1/compare.py')
    scenes=json.loads((root/'preflight_r1/scenes.json').read_text())
    expected={(r['task'],r['init_index']):r for r in scenes};assert len(expected)==30
    variants={};cases={};pilot={};first={};mismatch=[]
    for variant in VARIANTS:
        folder=root/('closedloop_'+variant+'_r1')
        assert json.loads((folder/'exit_summary.json').read_text())['all_tasks_exit_zero']
        state=json.loads((folder/'container_exit.json').read_text());assert state['ExitCode']==0 and not state['OOMKilled']
        memory=json.loads((folder/'memory_summary.json').read_text());assert memory['single_model_resident']
        assert json.loads((folder/'pilot/summary.json').read_text())['repeat_passed']
        pilotrows=json.loads((folder/'pilot/cases.json').read_text());assert len(pilotrows)==128
        pilot[variant]=np.asarray([r['env_actions'] for r in pilotrows]);episodes=[];task_counts=[];timings=[];initial_actions=[]
        for task in range(10):
            task_summary=json.loads((folder/f'task{task:02d}/summary.json').read_text())
            assert task_summary['trials']==3 and task_summary['all_videos_decoded']
            for index in [0,1,2]:
                p=folder/f'task{task:02d}/init{index:02d}.json';body=json.loads(p.read_text());ep=body['episode'];e=expected[(task,index)]
                assert (ep['variant'],ep['task'],ep['init_index'],ep['seed'],ep['initial_state_sha256'])==(variant,task,index,e['seed'],e['initial_state_sha256'])
                assert len(body['chunks'])==ep['requests'] and len(body['steps'])==ep['steps']
                assert ep['video_frames']==ep['steps']+1 and sha(p.parent/ep['video'])==ep['video_sha256']
                assert body['chunks'][0]['step']==0 and body['chunks'][0]['noise_seed']==ep['seed']*1000
                for step in body['steps']:
                    raw=np.asarray(step['raw_action']);actual=np.asarray(step['executed_action'])
                    assert raw.shape==(7,) and np.isfinite(raw).all() and np.array_equal(np.clip(raw,-1,1),actual)
                assert body['steps'][-1]['success']==ep['success']
                if not ep['success']:assert len(body['steps'])==220
                cases[(variant,task,index)]=ep;episodes.append(ep)
                initial_actions.append(body['chunks'][0]['raw_actions'])
                timings.extend(c['timing'] for c in body['chunks'])
            assert sum(int(cases[(variant,task,i)]['success']) for i in [0,1,2])==task_summary['successes']
            task_counts.append({'task':task,'successes':task_summary['successes'],'trials':3})
        first[variant]=np.asarray(initial_actions);k=sum(int(ep['success']) for ep in episodes)
        variants[variant]={'successes':k,'trials':30,'success_fraction':k/30,'Wilson_95_percent_CI_descriptive':wilson(k,30),
            'per_task':task_counts,'episodes':episodes,'closedloop_memory':memory,
            'closedloop_latency_ms':{name:stats([t[name] for t in timings]) for name in ['client_round_trip_ms','observation_ready_to_action_ms','camera_step_to_action_ms']},
            'pilot_summary':json.loads((folder/'pilot/summary.json').read_text())}
    for task,index in expected:
        a=cases[('trt_float',task,index)]
        for variant in VARIANTS[1:]:
            b=cases[(variant,task,index)]
            for field in ['simulation_state_sha256','initial_observation_sha256','xml_sha256','controller_sha256']:
                if a['initialization'][field]!=b['initialization'][field]:mismatch.append({'variant':variant,'task':task,'init_index':index,'field':field})
    for variant in VARIANTS:
        lost=[];restored=[]
        for task,index in expected:
            a=cases[('trt_float',task,index)]['success'];b=cases[(variant,task,index)]['success']
            if a and not b:lost.append({'task':task,'init_index':index})
            if b and not a:restored.append({'task':task,'init_index':index})
        v=variants[variant];net=len(lost)-len(restored)
        v['paired_vs_A100_TRT_float']={'lost':lost,'recovered':restored,'net_loss':net}
        v['closedloop_original_gate_passed']=v['successes']>=27 and net<=1 and not mismatch
        v['pilot_128_vs_A100_TRT_float']=metrics.quality(pilot['trt_float'],pilot[variant])
        v['heldout_30_initial_observations_vs_A100_TRT_float']=metrics.quality(first['trt_float'],first[variant])
        v['current_A100_closedloop_and_numerical_gates_passed']=v['closedloop_original_gate_passed'] and v['pilot_128_vs_A100_TRT_float']['original_gates_passed'] and v['heldout_30_initial_observations_vs_A100_TRT_float']['original_gates_passed']
    for variant,baseline in [('full_awq','awq'),('full_gptq','gptq')]:
        lost=[];restored=[]
        for task,index in expected:
            a=cases[(baseline,task,index)]['success'];b=cases[(variant,task,index)]['success']
            if a and not b:lost.append({'task':task,'init_index':index})
            if b and not a:restored.append({'task':task,'init_index':index})
        variants[variant]['paired_vs_same_algorithm_prefix_only']={'reference':baseline,'lost':lost,'recovered':restored,'net_loss':len(lost)-len(restored)}
    data={'version':'a100_closedloop_20261008_r1','device':'NVIDIA A100-SXM4-40GB','all_150_episodes_complete':True,
        'initial_condition_alignment_passed':not mismatch,'initial_condition_mismatches':mismatch,
        'variants':variants,'all_150_videos_fully_decoded':True,'prior_4090_quality_failures_not_waived':True,
        'A100_results_do_not_replace_4090_performance':True,'formal_200_tested':False,'two_hour_tested':False,
        'new_1000_serial_requests_tested':False,'actual_8GB_device_tested':False,'real_robot_tested':False,
        'production_release_accepted':False,'no_calibration_training_or_controller_change':True,
        'same_input_quality_compared_only_on_128_pilot_and_30_initial_observations':True}
    write_json(out/'summary.json',data)
    csv=['variant,task,init_index,seed,success,steps,requests,video']
    for variant in VARIANTS:
        for ep in variants[variant]['episodes']:
            csv.append(','.join(str(x) for x in [variant,ep['task'],ep['init_index'],ep['seed'],int(ep['success']),ep['steps'],ep['requests'],ep['video']]))
    with (out/'episodes.csv').open('x') as f:f.write('\n'.join(csv)+'\n')
    assert not mismatch,mismatch
    print('A100 UNIFIED CLOSEDLOOP', {v:(r['successes'],r['closedloop_original_gate_passed']) for v,r in variants.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True)
    a=p.parse_args();main(a.root,a.out)
