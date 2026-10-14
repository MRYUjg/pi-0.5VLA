"""Controlled Linear GPTQ adapter: second-order sequential error compensation.

Implements the published GPTQ update with fixed groups, no act-order permutation,
1% diagonal damping and FP32 Hessian. Does not pretend pi05 is a supported text model.
"""
import torch
from codec import scales,quantize


def quantize_gptq(weight,x,group=128,block=128,damp_fraction=.01):
    w=weight.float().clone()
    n,k=w.shape
    h=torch.zeros(k,k,device=w.device,dtype=torch.float32)
    for part in x.split(512):
        z=part.to(w.device).float()
        h.addmm_(z.T,z,alpha=2/x.shape[0])
    dead=h.diag()==0
    h[dead,dead]=1;w[:,dead]=0
    damping=damp_fraction*torch.mean(torch.diag(h))
    h.diagonal().add_(damping)
    chol=torch.linalg.cholesky(h)
    hinv=torch.cholesky_inverse(chol)
    upper=torch.linalg.cholesky(hinv,upper=True)
    del h,chol,hinv
    codes=torch.empty_like(w,dtype=torch.int8)
    scale=torch.empty((n,k//group),device=w.device,dtype=torch.bfloat16)
    for start in range(0,k,block):
        end=min(start+block,k)
        local=w[:,start:end].clone()
        errors=torch.zeros_like(local)
        for j in range(end-start):
            col=start+j
            if col%group==0:
                # Static contiguous group; BF16 scale is chosen before codes.
                source=local[:,j:j+group] if j+group<=local.shape[1] else w[:,col:col+group]
                s=scales(source,group)[:,0]
                scale[:,col//group]=s
            s=scale[:,col//group].float()
            q=torch.round(local[:,j]/s).clamp(-8,7).to(torch.int8)
            codes[:,col]=q
            err=(local[:,j]-q.float()*s)/upper[col,col]
            errors[:,j]=err
            local[:,j:]-=err[:,None]*upper[col,col:end][None,:]
        if end<k:
            w[:,end:]-=errors@upper[start:end,end:]
        if start%2048==0:print('GPTQ COLUMNS',start,k,flush=True)
    assert torch.isfinite(scale).all() and (scale>0).all()
    return codes.cpu(),scale.cpu(),{'damp_fraction':damp_fraction,'damping':float(damping),
        'hessian_dtype':'float32','act_order':False,'group_size':group,'block_size':block,'calibration_tokens':x.shape[0]}
