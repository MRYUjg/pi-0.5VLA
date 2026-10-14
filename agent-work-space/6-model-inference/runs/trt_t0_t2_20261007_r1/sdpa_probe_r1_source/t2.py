"""Matched RTN / NVIDIA ModelOpt AWQ / controlled GPTQ, canonical INT4 exports."""
import argparse
import copy
import gc
import json
import pathlib
import time
import traceback

import numpy as np
import onnx
from onnx import helper,TensorProto
import torch

from common import load_model, selected_language, write_json, sha, SOURCE_SHA
from codec import pack,unpack,scales,quantize,dequantize,tensor,bf16_tensor
from gptq import quantize_gptq


def awq(mod,x):
    import modelopt.torch.quantization as mtq
    layer=copy.deepcopy(mod)
    wrapper=torch.nn.Sequential(layer)
    config={'quant_cfg':{'default':{'enable':False},
        '*weight_quantizer':{'num_bits':4,'block_sizes':{-1:128,'type':'static'},'enable':True},
        '*input_quantizer':{'enable':False}},'algorithm':{'method':'awq_lite','alpha_step':.1}}
    def forward(m):
        with torch.inference_mode():
            for chunk in x.split(256):m(chunk.to('cuda'))
    mtq.quantize(wrapper,config,forward)
    layer=wrapper[0]
    pre=getattr(layer.input_quantizer,'pre_quant_scale',None)
    if pre is None:pre=getattr(layer.input_quantizer,'_pre_quant_scale',None)
    pre=torch.ones(mod.in_features,device='cuda') if pre is None else pre.detach().reshape(-1)
    # The final codec is registered up-front: BF16 scales/input multipliers and
    # round-to-even signed INT4. These are chosen before any artifact is exported.
    w=layer.weight.detach().float()
    scale=scales(w)
    q=quantize(w,scale)
    metadata={'implementation':'NVIDIA ModelOpt 0.33.1 awq_lite on controlled Linear',
        'alpha_step':.1,'calibration_tokens':x.shape[0],
        'pre_scale_min':float(pre.min()),'pre_scale_max':float(pre.max()),
        'final_scale_precision':'bfloat16','final_input_multiplier_precision':'bfloat16',
        'selected_by_real_complete_forward_inputs':True}
    return q.cpu(),scale.cpu(),pre.bfloat16().cpu(),metadata


def export(prefix,bundles,out):
    m=onnx.load(prefix)
    producer={v:n for n in m.graph.node for v in n.output}
    nodes=[];initializers={t.name:t for t in m.graph.initializer};remove_nodes=set();remove_weights=set();scope=[]
    for node in m.graph.node:
        if node.op_type!='MatMul' or node.input[1] not in producer:
            nodes.append(node);continue
        trans=producer[node.input[1]]
        if trans.op_type!='Transpose' or trans.input[0] not in initializers:
            nodes.append(node);continue
        param=trans.input[0]
        name='paligemma_with_expert.paligemma.model.'+param.removesuffix('.weight')
        if name not in bundles:
            nodes.append(node);continue
        b=torch.load(bundles[name],weights_only=True)
        q=unpack(b['packed'],b['shape']).T.contiguous()
        qname=param+'.int4';sname=param+'.scale';dqname=param+'.dequantized'
        initializers[qname]=tensor(qname,q)
        initializers[sname]=bf16_tensor(sname,b['scale'].T)
        nodes.append(helper.make_node('DequantizeLinear',[qname,sname],[dqname],name=param+'.DQ',axis=0,block_size=128))
        if not torch.equal(b['input_multiplier'],torch.ones_like(b['input_multiplier'])):
            pname=param+'.input_multiplier';scaled=node.input[0]+'.'+param+'.scaled'
            initializers[pname]=bf16_tensor(pname,b['input_multiplier'])
            nodes.append(helper.make_node('Mul',[node.input[0],pname],[scaled],name=param+'.input_scaling'))
            node.input[0]=scaled
        node.input[1]=dqname;nodes.append(node)
        remove_nodes.add(trans.name);remove_weights.add(param)
        scope.append({'module':name,'matmul':node.name,'shape':list(q.shape),'axis':0,'block_size':128,
                      'scale_shape':list(b['scale'].T.shape),'dtype':'INT4','zero_point':0})
    expected_unused={f'paligemma_with_expert.paligemma.model.language_model.layers.17.{s}'
        for s in ['self_attn.q_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']}
    assert set(bundles)-{n['module'] for n in scope}==expected_unused,(len(scope),set(bundles)-{n['module'] for n in scope})
    del m.graph.node[:];m.graph.node.extend(n for n in nodes if n.name not in remove_nodes)
    used={v for n in m.graph.node for v in n.input}
    del m.graph.initializer[:]
    m.graph.initializer.extend(t for name,t in initializers.items() if name in used)
    m.ir_version=10
    path=out/'prefix_int4.onnx'
    onnx.save_model(m,str(path),save_as_external_data=True,all_tensors_to_one_file=True,
        location='prefix_int4.data',size_threshold=1024,convert_attribute=False)
    del m;gc.collect()
    onnx.checker.check_model(str(path))
    graph=onnx.load(path,load_external_data=False)
    stored={t.name:t for t in graph.graph.initializer if t.data_type==TensorProto.INT4}
    assert len(stored)==121
    # Parser-only compatibility check belongs to export validation; full INT4
    # engine construction remains the separately authorized T3 stage.
    import tensorrt as trt
    logger=trt.Logger(trt.Logger.WARNING)
    builder=trt.Builder(logger)
    network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser=trt.OnnxParser(network,logger)
    parser_passed=parser.parse_from_file(str(path))
    errors=[str(parser.get_error(i)) for i in range(parser.num_errors)]
    write_json(out/'parser_validation.json',{'passed':parser_passed,'errors':errors,'engine_built':False,
        'inputs':[network.get_input(i).name for i in range(network.num_inputs)],
        'outputs':[network.get_output(i).name for i in range(network.num_outputs)]})
    assert parser_passed,errors
    write_json(out/'onnx_scope.json',{'nodes':scope,'INT4_initializers':len(stored),
        'source_eligible_layers':126,'reachable_layers':121,'unused_layers':sorted(expected_unused),
        'external_data':True,'opset':21,'contains_floating_source_quantized_weights':False,
        'source_prefix_sha256':sha(prefix)})
    return path


def main(out,calibration,prefix,algorithm):
    out.mkdir(parents=True,exist_ok=False)
    registry={'algorithm':algorithm,'source_sha256':SOURCE_SHA,'group_size':128,'signed_range':[-8,7],
        'symmetric':True,'zero_point':0,'scale_dtype':'bfloat16','input_multiplier_dtype':'bfloat16',
        'rounding':'nearest_even','packing':'even index low nibble, odd index high nibble',
        'scope':'126 language attention/MLP Linear','activation_source':'full sample_actions with all ten steps',
        'calibration_manifest_sha256':sha(calibration/'activation_manifest.json'),
        'quantized_engine_build_stage':'T3, not executed here'}
    write_json(out/'registration.json',registry)
    torch.backends.cuda.matmul.allow_tf32=False
    net,_=load_model(device='cpu')
    manifest=json.loads((calibration/'activation_manifest.json').read_text())
    folder=out/'packed_layers';folder.mkdir()
    bundles={};results=[]
    for index,(name,mod) in enumerate(selected_language(net).items()):
        start=time.time()
        info=manifest['activation_groups'][manifest['modules'][name]]
        assert sha(info['path'])==info['sha256']
        x=torch.load(info['path'],weights_only=True)
        w=mod.weight.cuda()
        pre=torch.ones(w.shape[1],dtype=torch.bfloat16)
        if algorithm=='RTN':
            s=scales(w);q=quantize(w,s);q=q.cpu();s=s.cpu();metadata={'method':'RTN engineering baseline'}
        elif algorithm=='AWQ':
            mod=mod.to('cuda');q,s,pre,metadata=awq(mod,x);mod=mod.cpu()
        else:q,s,metadata=quantize_gptq(w,x)
        b={'packed':pack(q.numpy()),'shape':list(q.shape),'scale':s,'input_multiplier':pre,
           'group_size':128,'algorithm':algorithm,'metadata':metadata,'source_sha256':SOURCE_SHA}
        recovered=unpack(b['packed'],b['shape'])
        assert torch.equal(q,recovered)
        path=folder/f'layer_{index:03d}.pt';torch.save(b,path)
        reread=torch.load(path,weights_only=True)
        assert torch.equal(dequantize(b),dequantize(reread))
        sample=x[::max(1,len(x)//128)][:128].cuda()
        actual=sample@w.T
        approximated=(sample*pre.cuda())@dequantize(b).cuda().T
        loss=float((actual.float()-approximated.float()).square().mean())
        results.append({'module':name,'file':str(path),'sha256':sha(path),'seconds':time.time()-start,
            'packed_weight_bytes':b['packed'].numel(),'scale_bytes':s.numel()*2,'input_multiplier_bytes':pre.numel()*2,
            'input_tokens':x.shape[0],'output_mse':loss,'metadata':metadata,'reload_exact':True})
        bundles[name]=path
        del x,w,s,q,b,sample,actual,approximated;gc.collect();torch.cuda.empty_cache()
        print('QUANTIZED',algorithm,index+1,126,name,flush=True)
    del net;gc.collect();torch.cuda.empty_cache()
    write_json(out/'packed_manifest.json',{'algorithm':algorithm,'layers':results,'layer_count':len(results),
        'packed_weight_bytes':sum(r['packed_weight_bytes'] for r in results),
        'scale_bytes':sum(r['scale_bytes'] for r in results),'real_packing':True,'reload_exact':True})
    export(prefix,bundles,out)
    write_json(out/'status.json',{'status':'exported','algorithm':algorithm,'quantized_engine':'not_executed',
        'Panda_success_rate':'not_accepted','4070_8GB':'not_executed'})
    print('T2 EXPORT PASS',algorithm,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--calibration',type=pathlib.Path,required=True)
    p.add_argument('--prefix',type=pathlib.Path,required=True);p.add_argument('--algorithm',choices=['RTN','AWQ','GPTQ'],required=True);a=p.parse_args()
    try:main(a.out,a.calibration,a.prefix,a.algorithm)
    except Exception:
        a.out.mkdir(parents=True,exist_ok=True);write_json(a.out/'failure.json',{'traceback':traceback.format_exc()});raise
