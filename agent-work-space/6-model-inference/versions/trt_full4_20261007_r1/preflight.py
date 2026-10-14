"""Inspect native INT4 partial groups before committing the full calibration."""
import json
import math
import pathlib

import numpy as np
import onnx
from onnx import helper, TensorProto
import tensorrt as trt
import torch

from common import ROOT, RUN, OLD, verified, write_json, float_helpers, wrappers


def parser_probe(k, n):
    shape = (k, n)
    codes = TensorProto(name='w', data_type=TensorProto.INT4, dims=shape,
                        raw_data=np.full(k*n//2, 0x11, np.uint8).tobytes())
    scale = onnx.numpy_helper.from_array(np.ones((math.ceil(k/128), n), np.float32), 'scale')
    graph = helper.make_graph([
        helper.make_node('DequantizeLinear', ['w','scale'], ['wd'], axis=0, block_size=128),
        helper.make_node('MatMul', ['x','wd'], ['y'])], 'partial_int4',
        [helper.make_tensor_value_info('x',TensorProto.FLOAT,[1,10,k])],
        [helper.make_tensor_value_info('y',TensorProto.FLOAT,[1,10,n])], [codes,scale])
    model = helper.make_model(graph,opset_imports=[helper.make_opsetid('',21)],ir_version=10)
    onnx.checker.check_model(model)
    logger=trt.Logger(trt.Logger.WARNING);builder=trt.Builder(logger)
    network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser=trt.OnnxParser(network,logger)
    ok=parser.parse(model.SerializeToString())
    return {'K':k,'N':n,'passed':ok,'errors':[str(parser.get_error(i)) for i in range(parser.num_errors)]}


def main():
    folder=RUN/'preflight_r1';folder.mkdir(exist_ok=False)
    checks=[parser_probe(32,1024),parser_probe(4304,1152)]
    write_json(folder/'partial_group_parser.json',checks)
    h=float_helpers();net,_=h.load_model(device='cpu')
    inventory={}
    for stage,module in wrappers(net).items():
        path=OLD/'t1_attempt_r6'/f'{stage}.onnx';verified(path)
        graph=onnx.load(path,load_external_data=False)
        initial={t.name:t for t in graph.graph.initializer}
        records=[]
        for name,m in module.named_modules():
            if not isinstance(m,torch.nn.Linear):continue
            key=name+'.weight'
            records.append({'module':name,'weight':key,'K':m.in_features,'N':m.out_features,
                'dtype':str(m.weight.dtype),'weight_initializer_present':key in initial,
                'parameters':m.weight.numel(),'K_multiple_128':m.in_features%128==0})
        inventory[stage]=records
    write_json(folder/'linear_inventory.json',inventory)
    print(json.dumps({'parser_probes':checks,'Linear_counts':{s:len(r) for s,r in inventory.items()}},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
