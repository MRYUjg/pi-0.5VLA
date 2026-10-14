"""Engineering properties only; synthetic probes are not VLA calibration data."""
import argparse
import pathlib
import torch
from codec import pack,unpack,scales,quantize
from gptq import quantize_gptq
from common import write_json

p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
q=torch.arange(-8,8,dtype=torch.int8).repeat(16).reshape(16,16)
assert torch.equal(q,unpack(pack(q.numpy()),q.shape))
assert pack(torch.tensor([-8,-1,0,7],dtype=torch.int8).numpy()).tolist()==[248,112]
torch.manual_seed(912345)
w=torch.randn(64,256,device='cuda').bfloat16()
x=torch.eye(256,dtype=torch.bfloat16)
rtn=quantize(w,scales(w)).cpu()
gptq,s,m=quantize_gptq(w,x)
assert torch.equal(rtn,gptq),int((rtn!=gptq).sum())
write_json(a.out,{'passed':True,'signed_nibble_roundtrip':True,'byte_order':True,
    'gptq_identity_hessian_matches_RTN':True,'synthetic_engineering_probe':True,
    'used_for_task_calibration':False,'gptq_metadata':m})
print('CODEC PROBE PASS')
