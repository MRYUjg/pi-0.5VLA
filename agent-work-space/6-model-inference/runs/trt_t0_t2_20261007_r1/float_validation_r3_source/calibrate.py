"""Capture exact layer inputs from complete ten-step pi05 forwards on real Panda data."""
import argparse
import collections
import hashlib
import json
import pathlib
import traceback

import numpy as np
import torch

from common import load_model, prepare, selected_language, write_json, sha, SOURCE_SHA


def select(dataset, per_task=200):
    chosen=[];seen=set()
    for task in range(10):
        folder=dataset/f'task{task:02d}'
        rows=json.loads((folder/'observations.json').read_text())
        buckets=collections.defaultdict(list);unique=[]
        for r in rows:
            if r['image_sha256'] in seen:continue
            seen.add(r['image_sha256'])
            buckets[r['phase']].append(r)
            unique.append(r)
        task_chosen=[]
        for phase in ['early','middle','late']:
            bucket=buckets[phase]
            n=min(len(bucket),per_task//3)
            if n:
                for i in np.linspace(0,len(bucket)-1,n,dtype=int):task_chosen.append(bucket[i])
        remaining=[r for r in unique if r not in task_chosen]
        for r in remaining:
            if len(task_chosen)>=per_task:break
            task_chosen.append(r)
        assert len(task_chosen)==per_task,(task,len(task_chosen))
        for r in task_chosen:chosen.append({**r,'path':str(folder/r['file']),'file_sha256':sha(folder/r['file'])})
    return chosen


def canonical(name):
    if '.k_proj' in name or '.v_proj' in name:return name.replace('.k_proj','.q_proj').replace('.v_proj','.q_proj')
    if '.up_proj' in name:return name.replace('.up_proj','.gate_proj')
    return name


def main(out,dataset):
    out.mkdir(parents=True,exist_ok=False)
    rows=select(dataset)
    pilot=[]
    for task in range(10):
        group=[r for r in rows if r['task']==task]
        n=13 if task<8 else 12
        pilot.extend(group[i] for i in np.linspace(0,len(group)-1,n,dtype=int))
    assert len(pilot)==128 and len(rows)==2000
    write_json(out/'pilot.json',pilot)
    write_json(out/'dataset.json',{'observations':rows,'total':len(rows),'source_sha256':SOURCE_SHA,
        'temporal_correlation':True,'observations_are_distinct_environment_frames':True,
        'not_independent_trials':True,'formal_test_used':False})
    net,cfg=load_model()
    modules=selected_language(net)
    mapping={name:canonical(name) for name in modules}
    buffers={n:[] for n in set(mapping.values())}
    indices=[];calls=collections.Counter()
    handles=[]
    for n in buffers:
        def hook(m,args,name=n):
            x=args[0].detach()[0,indices].contiguous().cpu()
            assert x.dtype==torch.bfloat16,(name,x.dtype)
            buffers[name].append(x)
            calls[name]+=1
        handles.append(modules[n].register_forward_pre_hook(hook))
    for number,r in enumerate(rows):
        observation=prepare(dict(np.load(r['path'])),cfg)
        # Select valid real-camera/text tokens; mask-zero placeholder/padding never
        # contributes to activation statistics. The full forward still retains it.
        valid=torch.cat([torch.ones(512,dtype=torch.bool),torch.zeros(256,dtype=torch.bool),
                         observation.tokenized_prompt_mask[0].cpu()])
        ix=torch.nonzero(valid).flatten()
        indices=ix[torch.linspace(0,len(ix)-1,4).long()].to('cuda')
        torch.manual_seed(r['noise_seed'])
        with torch.inference_mode():
            actions=net.sample_actions('cuda',observation,noise=torch.randn(1,10,32,device='cuda'),num_steps=10)
        assert actions.shape==(1,10,32) and torch.isfinite(actions).all()
        if (number+1)%100==0:print('CAPTURE',number+1,flush=True)
    for h in handles:h.remove()
    stats={}
    folder=out/'activations';folder.mkdir()
    for i,(name,parts) in enumerate(sorted(buffers.items())):
        assert len(parts)==2000,(name,len(parts))
        x=torch.cat(parts)
        path=folder/f'layer_{i:03d}.pt';torch.save(x,path)
        stats[name]={'path':str(path),'shape':list(x.shape),'dtype':str(x.dtype),'sha256':sha(path),'forwards':calls[name]}
        buffers[name].clear()
    write_json(out/'activation_manifest.json',{'source_sha256':SOURCE_SHA,'dataset_sha256':sha(out/'dataset.json'),
        'modules':mapping,'activation_groups':stats,'full_sample_actions_forwards':2000,
        'denoise_steps_each':10,'tokens_per_observation':4,'all_phases_from_real_policy_trajectories':True})
    print('CALIBRATION PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--dataset',type=pathlib.Path,required=True);a=p.parse_args()
    try:main(a.out,a.dataset)
    except Exception:
        a.out.mkdir(parents=True,exist_ok=True);write_json(a.out/'failure.json',{'traceback':traceback.format_exc()});raise
