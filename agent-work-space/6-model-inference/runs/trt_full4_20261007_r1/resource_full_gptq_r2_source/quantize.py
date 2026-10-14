"""AWQ/GPTQ all reachable Linear weights; exact zero padding for INT4 groups."""
import argparse
import copy
import gc
import json
import math
import pathlib
import time
import traceback

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import torch
from torch.nn import functional as F

from common import ROOT, RUN, OLD, sha, verified, write_json, float_helpers, wrappers, frozen_module


def scale_for(w, dtype):
    assert w.shape[1]%128==0
    return (w.float().reshape(w.shape[0],-1,128).abs().amax(-1)/7).clamp_min(1e-8).to(dtype)


def codes_for(w,s):
    return torch.round(w.float()/s.float().repeat_interleave(128,-1)).clamp(-8,7).to(torch.int8)


@torch.no_grad()
def gptq(weight, samples):
    w=weight.float().clone();n,k=w.shape;dtype=weight.dtype
    h=torch.zeros(k,k,device=w.device,dtype=torch.float32)
    for part in samples.split(512):
        z=part.to(w.device).float();h.addmm_(z.T,z,alpha=2/samples.shape[0])
    dead=h.diag()==0;h[dead,dead]=1;w[:,dead]=0
    damping=.01*torch.mean(torch.diag(h));h.diagonal().add_(damping)
    chol=torch.linalg.cholesky(h);hinv=torch.cholesky_inverse(chol)
    upper=torch.linalg.cholesky(hinv,upper=True);del h,chol,hinv
    q=torch.empty_like(w,dtype=torch.int8);scales=torch.empty((n,k//128),device=w.device,dtype=dtype)
    for start in range(0,k,128):
        end=start+128;local=w[:,start:end].clone();errors=torch.zeros_like(local)
        s=scale_for(local,dtype)[:,0];scales[:,start//128]=s
        for j in range(128):
            col=start+j;code=torch.round(local[:,j]/s.float()).clamp(-8,7).to(torch.int8);q[:,col]=code
            err=(local[:,j]-code.float()*s.float())/upper[col,col];errors[:,j]=err
            local[:,j:]-=err[:,None]*upper[col,col:end][None,:]
        if end<k:w[:,end:]-=errors@upper[start:end,end:]
    assert torch.isfinite(scales).all() and (scales>0).all()
    return q.cpu(),scales.cpu(),torch.ones(k,dtype=dtype),{
        'method':'controlled GPTQ second-order sequential compensation',
        'damp_fraction':.01,'damping':float(damping),'hessian_dtype':'FP32','act_order':False,
        'group':128,'block':128,'calibration_rows':len(samples)}


@torch.no_grad()
def awq(mod, samples):
    import modelopt.torch.quantization as mtq
    layer=copy.deepcopy(mod);wrapper=torch.nn.Sequential(layer)
    config={'quant_cfg':{'default':{'enable':False},
        '*weight_quantizer':{'num_bits':4,'block_sizes':{-1:128,'type':'static'},'enable':True},
        '*input_quantizer':{'enable':False}},'algorithm':{'method':'awq_lite','alpha_step':.1}}
    def forward(m):
        with torch.inference_mode():
            for x in samples.split(256):m(x.to('cuda'))
    mtq.quantize(wrapper,config,forward);layer=wrapper[0]
    pre=getattr(layer.input_quantizer,'pre_quant_scale',None)
    if pre is None:pre=getattr(layer.input_quantizer,'_pre_quant_scale',None)
    pre=torch.ones(mod.in_features,device='cuda',dtype=mod.weight.dtype) if pre is None else pre.detach().reshape(-1)
    s=scale_for(layer.weight,mod.weight.dtype);q=codes_for(layer.weight,s)
    return q.cpu(),s.cpu(),pre.to(mod.weight.dtype).cpu(),{
        'method':'NVIDIA ModelOpt 0.33.1 awq_lite on controlled Linear',
        'alpha_step':.1,'calibration_rows':len(samples),
        'pre_scale_min':float(pre.min()),'pre_scale_max':float(pre.max())}


def float_tensor(name, value):
    value=value.contiguous().cpu()
    if value.dtype==torch.bfloat16:
        return TensorProto(name=name,data_type=TensorProto.BFLOAT16,dims=list(value.shape),
                           raw_data=value.view(torch.int16).numpy().tobytes())
    assert value.dtype==torch.float32,value.dtype
    return numpy_helper.from_array(value.numpy(),name)


def export(stage, packed, out):
    source=OLD/'t1_attempt_r6'/f'{stage}.onnx';verified(source)
    model=onnx.load(source);init={t.name:t for t in model.graph.initializer}
    producer={o:n for n in model.graph.node for o in n.output};nodes=[];scope=[]
    codec=frozen_module('full4_codec_export',ROOT/'versions/trt_t0_t2_20261007_r1/codec.py')
    for node in model.graph.node:
        param=None;transposed=False
        if node.op_type=='MatMul' and node.input[1] in producer:
            t=producer[node.input[1]]
            if t.op_type=='Transpose' and t.input[0] in init:
                param=t.input[0];transposed=True
        elif node.op_type=='Gemm' and node.input[1] in init:
            attrs={a.name:helper.get_attribute_value(a) for a in node.attribute}
            assert attrs.get('transB',0)==1,(node.name,attrs)
            param=node.input[1]
        key=param.removesuffix('.weight') if param else None
        if key not in packed:nodes.append(node);continue
        b=torch.load(packed[key],weights_only=True)
        q=codec.unpack(b['packed'],b['shape']);s=b['scale'];pre=b['input_multiplier']
        if transposed:q=q.T.contiguous();s=s.T.contiguous()
        qname=param+'.int4';sname=param+'.scale';dqname=param+'.dequantized'
        init[qname]=codec.tensor(qname,q);init[sname]=float_tensor(sname,s)
        nodes.append(helper.make_node('DequantizeLinear',[qname,sname],[dqname],
                     name=param+'.DQ',axis=0 if transposed else 1,block_size=128))
        data=node.input[0]
        if b['pad_K']:
            rank=b['input_rank'];pads=[0]*(2*rank);pads[-1]=b['pad_K']
            pname=param+'.zero_pads';zname=param+'.zero';padded=data+'.'+param+'.padded'
            init[pname]=numpy_helper.from_array(np.array(pads,np.int64),pname)
            init[zname]=float_tensor(zname,torch.tensor(0.,dtype=b['scale'].dtype))
            nodes.append(helper.make_node('Pad',[data,pname,zname],[padded],name=param+'.group_padding',mode='constant'))
            data=padded
        if not torch.equal(pre,torch.ones_like(pre)):
            pname=param+'.input_multiplier';scaled=data+'.'+param+'.scaled'
            init[pname]=float_tensor(pname,pre)
            nodes.append(helper.make_node('Mul',[data,pname],[scaled],name=param+'.input_scaling'))
            data=scaled
        node.input[0]=data;node.input[1]=dqname;nodes.append(node)
        scope.append({'module':key,'weight':param,'matmul':node.name,'op':node.op_type,
            'original_shape':b['original_shape'],'INT4_shape':list(q.shape),
            'scale_shape':list(s.shape),'axis':0 if transposed else 1,'block_size':128,
            'padded_input_channels':b['pad_K'],'input_rank':b['input_rank'],
            'scale_dtype':str(b['scale'].dtype),'algorithm':b['algorithm']})
    assert set(packed)=={s['module'] for s in scope},(stage,'missing',set(packed)-{s['module'] for s in scope})
    # Retain only nodes reaching outputs; the old floating Transpose weights are
    # otherwise dead after replacement, and must not persist beside INT4 copies.
    producer={o:n for n in nodes for o in n.output};needed=set();seen=set()
    def visit(value):
        if value in seen:return
        seen.add(value)
        if value in producer:
            n=producer[value];needed.add(id(n))
            for x in n.input:visit(x)
    for value in model.graph.output:visit(value.name)
    kept=[n for n in nodes if id(n) in needed];used={x for n in kept for x in n.input}
    del model.graph.node[:];model.graph.node.extend(kept)
    del model.graph.initializer[:];model.graph.initializer.extend(t for name,t in init.items() if name in used)
    model.ir_version=10
    path=out/f'{stage}_int4.onnx'
    onnx.save_model(model,str(path),save_as_external_data=True,all_tensors_to_one_file=True,
        location=f'{stage}_int4.data',size_threshold=1024,convert_attribute=False)
    del model,init,nodes,producer;gc.collect();onnx.checker.check_model(str(path))
    graph=onnx.load(path,load_external_data=False)
    actual=[t for t in graph.graph.initializer if t.data_type==TensorProto.INT4]
    assert len(actual)==len(scope)
    write_json(out/'onnx_scope.json',{'stage':stage,'nodes':scope,'target_reachable_Linear':len(packed),
        'INT4_initializers':len(actual),'all_target_Linear_weights_quantized':True,
        'zero_padding_preserves_original_external_interface_and_float_Linear_algebra':True,
        'float_source_Linear_weights_removed':True,'source_ONNX_sha256':sha(source),
        'exported_ONNX_sha256':sha(path)})
    print('EXPORT COMPLETE',stage,len(scope),flush=True)


@torch.no_grad()
def main(out,stage,algorithm,calibration):
    out.mkdir(exist_ok=False);folder=out/'packed_layers';folder.mkdir()
    manifest=json.loads((calibration/'manifest.json').read_text())
    entries={r['module']:r for r in manifest['layers'] if r['stage']==stage}
    h=float_helpers();net,_=h.load_model(device='cpu');module=wrappers(net)[stage]
    codec=frozen_module('full4_codec',ROOT/'versions/trt_t0_t2_20261007_r1/codec.py')
    packed={};records=[]
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for i,(name,mod) in enumerate((n,m) for n,m in module.named_modules() if isinstance(m,torch.nn.Linear)):
        begin=time.time();entry=entries[name];assert sha(entry['path'])==entry['sha256']
        sample=torch.load(entry['path'],weights_only=True)
        k=mod.in_features;pad=math.ceil(k/128)*128-k
        original=mod.weight.detach().cuda();padded=F.pad(original,(0,pad))
        layer=torch.nn.Linear(k+pad,mod.out_features,bias=mod.bias is not None,device='cuda',dtype=mod.weight.dtype)
        with torch.no_grad():
            layer.weight.copy_(padded)
            if mod.bias is not None:layer.bias.copy_(mod.bias.cuda())
        samples=F.pad(sample,(0,pad))
        if algorithm=='AWQ':q,s,pre,meta=awq(layer,samples)
        else:q,s,pre,meta=gptq(layer.weight,samples)
        assert torch.isfinite(s).all() and (s>0).all() and torch.isfinite(pre).all(),name
        b={'packed':codec.pack(q.numpy()),'shape':list(q.shape),'scale':s,
            'input_multiplier':pre,'original_shape':list(mod.weight.shape),'pad_K':pad,
            'input_rank':entry['input_rank'],'group_size':128,'algorithm':algorithm,'metadata':meta}
        assert torch.equal(q,codec.unpack(b['packed'],b['shape']))
        recovered=(q.float()*s.float().repeat_interleave(128,-1)).to(mod.weight.dtype).cuda()
        ix=torch.linspace(0,len(samples)-1,min(128,len(samples))).long()
        x=samples[ix].cuda();original_result=sample[ix].cuda()@original.T
        approximate=(x*pre.cuda())@recovered.T
        mse=float((original_result.float()-approximate.float()).square().mean())
        assert math.isfinite(mse),name
        path=folder/f'layer_{i:03d}.pt';torch.save(b,path);packed[name]=path
        records.append({'module':name,'file':str(path),'sha256':sha(path),
            'original_weight_parameters':mod.weight.numel(),'packed_parameters_including_zeros':q.numel(),
            'pad_K':pad,'scale_dtype':str(s.dtype),'calibration_rows':len(samples),
            'layer_output_mse':mse,'seconds':time.time()-begin,'metadata':meta})
        del sample,samples,original,padded,layer,q,s,pre,recovered,x,original_result,approximate,b
        gc.collect();torch.cuda.empty_cache()
        print('QUANTIZED',algorithm,stage,i+1,'/',len(entries),name,'MSE',mse,flush=True)
    del module,net;gc.collect();torch.cuda.empty_cache()
    write_json(out/'packed_manifest.json',{'stage':stage,'algorithm':algorithm,'layers':records,
        'calibration_manifest_sha256':sha(calibration/'manifest.json'),'all_Linear_quantized':True,
        'original_Linear_parameters':sum(r['original_weight_parameters'] for r in records),
        'padded_parameters':sum(r['packed_parameters_including_zeros'] for r in records)})
    export(stage,packed,out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True)
    p.add_argument('--stage',choices=['vision','action_step'],required=True)
    p.add_argument('--algorithm',choices=['AWQ','GPTQ'],required=True)
    p.add_argument('--calibration',type=pathlib.Path,required=True);a=p.parse_args()
    try:main(a.out,a.stage,a.algorithm,a.calibration)
    except Exception:
        if a.out.exists():write_json(a.out/'failure.json',{'traceback':traceback.format_exc()})
        raise
