"""Identify the actual BF16 SDPA probability rounding, without editing repository code."""
import pathlib
import torch
import torch.nn.functional as F
from common import ROOT,load_model,write_json
from subgraphs import Vision
from runtime import metrics

torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
net,_=load_model();vision=Vision(net).eval()
run=ROOT/'runs/trt_t0_t2_20261007_r1'
data=torch.load(run/'t0_attempt_r4/reference_inputs.pt',weights_only=True)
original=F.scaled_dot_product_attention
results={}
try:
    with torch.inference_mode():
        ref=[vision(x.cuda()) for x in data['all_images']]
        for rounding in [False,True]:
            def custom(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,scale=None,enable_gqa=False):
                assert attn_mask is None and dropout_p==0 and not is_causal and not enable_gqa
                s=q.shape[-1]**-.5 if scale is None else scale
                p=torch.softmax((q.float()@k.float().transpose(-1,-2))*s,-1)
                if rounding:p=p.to(q.dtype).float()
                return (p@v.float()).to(q.dtype)
            F.scaled_dot_product_attention=custom
            results['BF16_probabilities' if rounding else 'FP32_probabilities']=[metrics(y,vision(x.cuda())) for x,y in zip(data['all_images'],ref)]
finally:
    F.scaled_dot_product_attention=original
write_json(run/'sdpa_probe_r1.json',results)
print('SDPA PROBE',results,flush=True)
