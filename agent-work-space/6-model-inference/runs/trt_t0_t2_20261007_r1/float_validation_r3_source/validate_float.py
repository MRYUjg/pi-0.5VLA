"""128 real observations, same noise and all ten steps, across all Spatial tasks."""
import argparse
import json
import pathlib
import traceback

import numpy as np
import torch

from common import load_model,prepare,time_table,transforms,write_json,sha
from subgraphs import inputs
from runtime import Engine,metrics


def main(out,pilot,engines):
    out.mkdir(exist_ok=False,parents=True)
    gates={'internal_action_mae_max':.01,'internal_action_p99_abs_max':.05,
        'contract_range_scaled_motion_mae_max':.01,'contract_range_scaled_motion_p99_abs_max':.05,
        'confident_gripper_disagreement_max':.005,'confident_gripper_abs_reference_min':.2,
        'clipping_rate_increase_max':.01,'requires_finite':True}
    write_json(out/'gates.json',gates)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    net,cfg=load_model();_,output=transforms(cfg)
    vision=Engine(engines/'vision.engine');prefix=Engine(engines/'prefix.engine');step=Engine(engines/'action_step.engine')
    rows=json.loads(pilot.read_text());times,table,dt=time_table()
    records=[];errors=[];motion_errors=[];disagree=confident=clip_ref=clip_got=0
    clip_ref_dims=np.zeros(7,dtype=np.int64);clip_got_dims=np.zeros(7,dtype=np.int64)
    with torch.inference_mode():
        for index,r in enumerate(rows):
            observation=prepare(dict(np.load(r['path'])),cfg)
            images,pref,act=inputs(net,observation)
            actual_images=torch.cat([vision(image)[0] for image in images],1)
            image_check=metrics(pref[0],actual_images)
            actual_prefix=prefix(actual_images,*pref[1:])
            original_embs=net.embed_prefix(*net._preprocess_observation(observation,train=False)[:4])[0]
            _,original_cache=net.paligemma_with_expert.forward(attention_mask=pref[4],position_ids=pref[3],
                inputs_embeds=[original_embs,None],use_cache=True)
            cache_check=[metrics(original_cache[i][j],actual_prefix[1+2*i+j]) for i in range(18) for j in range(2)]
            frozen=tuple(t.clone() for t in actual_prefix[1:])
            torch.manual_seed(r['noise_seed']);noise=torch.randn(1,10,32,device='cuda')
            x_ref=noise.clone();x_got=noise.clone();step_check=[]
            for j in range(10):
                ref=net.denoise_step(observation.state,pref[2],original_cache,x_ref,times[j:j+1])
                forced,=step(x_ref,table[j:j+1],*act,*actual_prefix[1:])
                got,=step(x_got,table[j:j+1],*act,*actual_prefix[1:])
                step_check.append(metrics(ref,forced))
                x_ref=x_ref+dt*ref;x_got=x_got+dt*got
            assert all(torch.equal(a,b) for a,b in zip(frozen,actual_prefix[1:]))
            error=(x_ref-x_got).abs().cpu().numpy();errors.append(error.reshape(-1))
            state=observation.state[0].cpu().numpy()
            a=output({'state':state.copy(),'actions':x_ref[0].cpu().numpy()})['actions']
            b=output({'state':state.copy(),'actions':x_got[0].cpu().numpy()})['actions']
            motion_errors.append((np.abs(a[:,:6]-b[:,:6])/2).reshape(-1))
            c=np.abs(a[:,6])>.2;confident+=int(c.sum());disagree+=int((np.sign(a[:,6][c])!=np.sign(b[:,6][c])).sum())
            clip_ref+=int((np.abs(a)>1).sum());clip_got+=int((np.abs(b)>1).sum())
            clip_ref_dims+=(np.abs(a)>1).sum(axis=0);clip_got_dims+=(np.abs(b)>1).sum(axis=0)
            records.append({'task':r['task'],'seed':r['seed'],'step':r['step'],'file_sha256':r['file_sha256'],
                'image_embeddings':image_check,'prefix_cache':cache_check,'denoise_v_t':step_check,
                'actions':metrics(x_ref,x_got),'finite':bool(torch.isfinite(x_got).all())})
            records[-1].update(reference_env_actions=a.tolist(),tensorrt_env_actions=b.tolist())
            if (index+1)%16==0:print('FLOAT VALIDATE',index+1,len(rows),flush=True)
    e=np.concatenate(errors);m=np.concatenate(motion_errors)
    summary={'observations':len(rows),'tasks':sorted({r['task'] for r in rows}),
        'internal_action_mae':float(e.mean()),'internal_action_p99_abs':float(np.quantile(e,.99)),
        'internal_action_max_abs':float(e.max()),'motion_mae':float(m.mean()),'motion_p99_abs':float(np.quantile(m,.99)),
        'confident_gripper_count':confident,'gripper_disagreements':disagree,'gripper_disagreement_fraction':disagree/max(1,confident),
        'clipping_rate_increase':(clip_got-clip_ref)/(len(rows)*10*7),'finite':all(r['finite'] for r in records),
        'reference_clipped_values_per_dimension':clip_ref_dims.tolist(),'tensorrt_clipped_values_per_dimension':clip_got_dims.tolist(),
        'prefix_cache_immutable':True,'pilot_sha256':sha(pilot),'4070_performance_accepted':False}
    summary['passed']=summary['internal_action_mae']<=.01 and summary['internal_action_p99_abs']<=.05 and \
        summary['motion_mae']<=.01 and summary['motion_p99_abs']<=.05 and \
        summary['gripper_disagreement_fraction']<=.005 and summary['clipping_rate_increase']<=.01 and summary['finite']
    write_json(out/'cases.json',records);write_json(out/'summary.json',summary)
    assert summary['passed'],summary
    print('FLOAT VALIDATION PASS',summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--pilot',type=pathlib.Path,required=True);p.add_argument('--engines',type=pathlib.Path,required=True);a=p.parse_args()
    try:main(a.out,a.pilot,a.engines)
    except Exception:
        a.out.mkdir(exist_ok=True,parents=True);write_json(a.out/'failure.json',{'traceback':traceback.format_exc()});raise
