"""One canonical signed INT4 codec for every algorithm; no downstream requantization."""
import numpy as np
import torch
from onnx import TensorProto


def pack(q):
    values=np.asarray(q,dtype=np.int8).reshape(-1)
    assert values.size%2==0 and values.min()>=-8 and values.max()<=7
    u=values.astype(np.uint8)&15
    return torch.from_numpy(u[::2] | (u[1::2]<<4))


def unpack(packed,shape):
    p=packed.cpu().numpy().astype(np.uint8)
    q=np.empty(p.size*2,dtype=np.int8)
    q[::2]=(p&15).astype(np.int8);q[1::2]=(p>>4).astype(np.int8)
    q[q>=8]-=16
    return torch.from_numpy(q.reshape(shape))


def scales(weight,group=128):
    n,k=weight.shape
    assert k%group==0
    return (weight.float().reshape(n,k//group,group).abs().amax(-1)/7).clamp_min(1e-8).bfloat16()


def quantize(weight,scale,group=128):
    n,k=weight.shape
    s=scale.float().repeat_interleave(group,-1)
    q=torch.round(weight.float()/s).clamp(-8,7).to(torch.int8)
    return q


def dequantize(bundle):
    q=unpack(bundle['packed'],bundle['shape']).float()
    return (q*bundle['scale'].float().repeat_interleave(bundle['group_size'],-1)).bfloat16()


def tensor(name,q):
    return TensorProto(name=name,data_type=TensorProto.INT4,dims=list(q.shape),raw_data=pack(q.numpy()).numpy().tobytes())


def bf16_tensor(name,value):
    value=value.contiguous().cpu().bfloat16()
    return TensorProto(name=name,data_type=TensorProto.BFLOAT16,dims=list(value.shape),
        raw_data=value.view(torch.int16).numpy().tobytes())
