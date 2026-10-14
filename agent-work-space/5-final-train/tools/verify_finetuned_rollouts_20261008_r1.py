"""Completed fine-tuned cohort verification; baseline comparison stays explicitly pending."""
import hashlib
import json
import pathlib
import subprocess

import imageio_ffmpeg
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
SOURCE='closed_loop_20261008_r1'
OUTPUT='closed_loop_finetuned_20261008_r1'
CONDITION='eco65_zhixing_lora'


def main():
    source=ROOT/'runs'/SOURCE;out=ROOT/'runs'/OUTPUT
    out.mkdir(exist_ok=False)
    (out/CONDITION).symlink_to(source/CONDITION,target_is_directory=True)
    seeds=json.loads((ROOT/'versions/rollout_20261008_r1/configs/test_seeds.json').read_text())['tasks']
    videos=[];counts={};failures={};allrows=[]
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    for task,ss in seeds.items():
        taskrows=[]
        for index,seed in enumerate(ss):
            ep=source/CONDITION/f'task{int(task):02d}_s{seed}'
            execution=json.loads((ep/'execution.json').read_text());r=execution['result']
            assert execution['process_exit']==0
            assert r['task_id']==int(task) and r['seed']==seed and r['condition']==CONDITION
            steps=[json.loads(line) for line in (ep/'steps.jsonl').read_text().splitlines()]
            assert len(steps)==r['steps']
            assert all(np.isfinite(x['qpos_after']).all() and np.isfinite(x['qvel_after']).all() and x['finite'] and not any(x['physics_warnings']) for x in steps)
            native=any(x['native_success'] and x['step']<=220 for x in steps)
            assert bool(r['initialization_pass'] and native)==r['primary_success']
            assert len(steps)<=220 or index==0
            taskrows.append(r);allrows.append(r)
            failures[r['failure']]=failures.get(r['failure'],0)+1
            if index==0:
                path=ep/'rollout.mp4'
                sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==r['video_sha256']
                frames,seconds=imageio_ffmpeg.count_frames_and_secs(str(path));assert frames==r['video_frames']
                decoded=subprocess.run([ffmpeg,'-v','error','-i',str(path),'-f','null','-'],capture_output=True)
                assert decoded.returncode==0
                videos.append({'condition':CONDITION,'task':int(task),'seed':seed,'path':str(path.relative_to(ROOT)),
                    'primary_success':r['primary_success'],'extended_success':r['native_success_extended'],
                    'initialization_pass':r['initialization_pass'],'failure':r['failure'],
                    'frames':frames,'seconds':seconds,'sha256':sha,'decode_pass':True})
        counts[task]={'completed':len(taskrows),'planned':20,'successes':sum(r['primary_success'] for r in taskrows),
                      'invalid_initializations':sum(not r['initialization_pass'] for r in taskrows)}
    assert len(allrows)==200 and len(videos)==10
    summary={'complete':True,'verification_pass':True,'scope':'fine-tuned cohort ONLY: 200 attempts, identity/native-goal/finite-state accounting, first10 MP4 full decode/frame/hash. Full action-chunk crosscheck and 3 baseline cohorts remain under frozen main evaluation.',
             'source_run':SOURCE,'condition':CONDITION,'successes':sum(r['primary_success'] for r in allrows),
             'attempts':200,'primary_budget':220,'initialization_pass_count':sum(r['initialization_pass'] for r in allrows),
             'tasks':counts,'failure_categories':failures,'videos':videos,
             'baseline_comparison_complete':False,'adaptation_acceptance_pass':sum(r['primary_success'] for r in allrows)>=180}
    with (out/'verified_results.json').open('x') as f:json.dump(summary,f,indent=2)
    print('FINETUNED VERIFIED',summary['successes'],'/200',flush=True)


if __name__=='__main__':main()
