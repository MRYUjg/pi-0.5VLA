"""200 registered observations, real vision tokens and all ten expert steps."""
import argparse
import collections
import json
import pathlib

import numpy as np
import torch

from common import RUN, OLD, float_helpers, wrappers, verified, sha, write_json


def main(out):
    out.mkdir(exist_ok=False)
    dataset=OLD/'calibration_r1/dataset.json';verified(dataset)
    all_rows=json.loads(dataset.read_text())['observations'];rows=[]
    for task in range(10):
        group=[r for r in all_rows if r['task']==task]
        rows.extend(group[i] for i in np.linspace(0,len(group)-1,20,dtype=int))
    assert len(rows)==200 and len({r['file_sha256'] for r in rows})==200
    h=float_helpers();net,cfg=h.load_model()
    specs=wrappers(net);parts=collections.defaultdict(list);calls=collections.Counter();handles=[]
    modules={}
    for stage,model in specs.items():
        for name,mod in model.named_modules():
            if not isinstance(mod,torch.nn.Linear):continue
            key=stage+'/'+name;modules[key]=mod
            def hook(m,args,key=key,stage=stage):
                calls[key]+=1
                # Original complete forward still computes the masked third view.
                if stage=='vision' and (calls[key]-1)%3==2:return
                x=args[0].detach().reshape(-1,m.in_features)
                n=min(8 if stage=='vision' else 10,len(x))
                ix=torch.linspace(0,len(x)-1,n,device=x.device).long()
                sample=x[ix].contiguous().cpu()
                assert torch.isfinite(sample).all(),key
                parts[key].append(sample)
            handles.append(mod.register_forward_pre_hook(hook))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for i,row in enumerate(rows):
        verified(pathlib.Path(row['path']))
        observation=h.prepare(dict(np.load(row['path'])),cfg)
        torch.manual_seed(row['noise_seed'])
        with torch.inference_mode():
            x=net.sample_actions('cuda',observation,noise=torch.randn(1,10,32,device='cuda'),num_steps=10)
        assert x.shape==(1,10,32) and torch.isfinite(x).all()
        if (i+1)%20==0:print('CAPTURE',i+1,'/200',flush=True)
    for handle in handles:handle.remove()
    folder=out/'activations';folder.mkdir();report=[]
    for i,(key,mod) in enumerate(modules.items()):
        assert calls[key]==(600 if key.startswith('vision/') else 2000),(key,calls[key])
        sample=torch.cat(parts.pop(key));file=folder/f'layer_{i:03d}.pt'
        torch.save(sample,file)
        report.append({'key':key,'stage':key.split('/')[0],'module':key.split('/',1)[1],
            'path':str(file),'sha256':sha(file),'shape':list(sample.shape),'dtype':str(sample.dtype),
            'weight_dtype':str(mod.weight.dtype),'weight_shape':list(mod.weight.shape),
            'full_forward_calls':calls[key]})
        print('SAVED',key,list(sample.shape),flush=True)
    write_json(out/'manifest.json',{'observations':rows,'source_dataset_sha256':sha(dataset),
        'count':200,'noise_per_observation':1,'all_ten_expert_steps':True,
        'real_vision_views_captured':2,'masked_view_forward_preserved':True,'layers':report,
        'calibration_not_independent_quality_test':True,'formal_indices_used':False})
    print('CAPTURE COMPLETE',len(report),'Linear layers',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();main(a.out)
