"""Offline dequantized policy comparison; this is not a native INT4 speed test."""
import argparse
import json
import pathlib
import traceback
import gc

import numpy as np
import torch
from torch import nn

from common import load_model,prepare,transforms,selected_language,write_json,sha
from codec import dequantize


class DequantizedLinear(nn.Module):
    def __init__(self,b):
        super().__init__()
        self.weight=nn.Parameter(dequantize(b),requires_grad=False)
        self.bias=None
        self.register_buffer('input_multiplier',b['input_multiplier'])

    def forward(self,x):
        return torch.nn.functional.linear(x*self.input_multiplier,self.weight,self.bias)


def main(out,pilot,candidate):
    out.mkdir(exist_ok=False,parents=True)
    gates={'motion_mae_max':.01,'motion_p99_abs_max':.05,'gripper_disagreement_max':.005,
        'confident_reference_gripper_abs_min':.2,'clipping_rate_increase_max':.01,
        'native_int4_runtime_acceptance':False}
    write_json(out/'gates.json',gates)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=json.loads(pilot.read_text())
    manifest=json.loads((candidate/'packed_manifest.json').read_text())
    net,cfg=load_model();_,output=transforms(cfg)
    originals=[]
    with torch.inference_mode():
        for r in rows:
            ob=prepare(dict(np.load(r['path'])),cfg);torch.manual_seed(r['noise_seed'])
            noise=torch.randn(1,10,32,device='cuda')
            x=net.sample_actions('cuda',ob,noise=noise)
            originals.append(output({'state':ob.state[0].cpu().numpy(),'actions':x[0].cpu().numpy()})['actions'])
    mapping=selected_language(net)
    for entry in manifest['layers']:
        assert sha(entry['file'])==entry['sha256']
        b=torch.load(entry['file'],weights_only=True)
        replacement=DequantizedLinear(b).cuda()
        parent,leaf=entry['module'].rsplit('.',1)
        setattr(net.get_submodule(parent),leaf,replacement)
    del mapping;gc.collect();torch.cuda.empty_cache()
    records=[];errors=[];disagreements=confident=clips_ref=clips_got=0
    with torch.inference_mode():
        for index,(r,ref) in enumerate(zip(rows,originals)):
            ob=prepare(dict(np.load(r['path'])),cfg);torch.manual_seed(r['noise_seed'])
            x=net.sample_actions('cuda',ob,noise=torch.randn(1,10,32,device='cuda'))
            got=output({'state':ob.state[0].cpu().numpy(),'actions':x[0].cpu().numpy()})['actions']
            assert np.isfinite(got).all()
            d=np.abs(ref[:,:6]-got[:,:6])/2;errors.append(d.reshape(-1))
            c=np.abs(ref[:,6])>.2;confident+=int(c.sum())
            disagreements+=int((np.sign(ref[:,6][c])!=np.sign(got[:,6][c])).sum())
            clips_ref+=int((np.abs(ref)>1).sum());clips_got+=int((np.abs(got)>1).sum())
            records.append({'task':r['task'],'seed':r['seed'],'step':r['step'],'motion_mae':float(d.mean()),
                'motion_max_abs':float(d.max()),'reference_actions':ref.tolist(),'candidate_actions':got.tolist()})
            if (index+1)%32==0:print('OFFLINE QUANT VALIDATE',manifest['algorithm'],index+1,flush=True)
    d=np.concatenate(errors)
    summary={'algorithm':manifest['algorithm'],'observations':len(rows),'motion_mae':float(d.mean()),
        'motion_p99_abs':float(np.quantile(d,.99)),'motion_max_abs':float(d.max()),
        'gripper_disagreements':disagreements,'confident_gripper_count':confident,
        'gripper_disagreement_fraction':disagreements/max(1,confident),
        'clipping_rate_increase':(clips_got-clips_ref)/(len(rows)*10*7),
        'comparison':'same source PyTorch float versus offline canonical dequantized weights; same observations/noise/steps',
        'native_INT4_engine_built':False,'4070_8GB_accepted':False,'Panda_closed_loop_accepted':False}
    summary['offline_accuracy_passed']=summary['motion_mae']<=.01 and summary['motion_p99_abs']<=.05 and \
        summary['gripper_disagreement_fraction']<=.005 and summary['clipping_rate_increase']<=.01
    write_json(out/'cases.json',records);write_json(out/'summary.json',summary)
    print('OFFLINE COMPARISON COMPLETE',summary,flush=True)
    # A failed candidate is retained as a measured comparison; the pipeline must
    # inspect offline_accuracy_passed before any later deployment decision.


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--pilot',type=pathlib.Path,required=True);p.add_argument('--candidate',type=pathlib.Path,required=True);a=p.parse_args()
    try:main(a.out,a.pilot,a.candidate)
    except Exception:
        a.out.mkdir(exist_ok=True,parents=True);write_json(a.out/'failure.json',{'traceback':traceback.format_exc()});raise
