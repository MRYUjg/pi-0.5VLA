"""Unchanged TensorRT build settings, native INT4 graphs and detailed inspector."""
import argparse
import gc
import json
import pathlib
import time
import traceback

import onnx
from onnx import TensorProto
import tensorrt as trt

from common import sha, write_json, verified


def main(source,out,stage,algorithm):
    out.mkdir(exist_ok=False)
    verified(source)
    model=onnx.load(source,load_external_data=False)
    external={v.value for t in model.graph.initializer for v in t.external_data if v.key=="location"}
    externals={n:verified(source.parent/n) for n in external}
    qweights=[t for t in model.graph.initializer if t.data_type==TensorProto.INT4]
    assert len(qweights)==(0 if algorithm=="float" else (163 if stage=="vision" else 167 if stage=="action_step" else 121))
    write_json(out/'registration.json',{'algorithm':algorithm,'stage':stage,
        'source_ONNX_sha256':sha(source),'external_data_sha256':externals,'source_INT4_initializers':len(qweights),
        'TensorRT':trt.__version__,'build_GPU':__import__('torch').cuda.get_device_name(0),'strongly_typed':True,'TF32':False,
        'workspace_bytes':2*1024**3,'builder_optimization_level':3,'hardware_compatibility':'NONE'})
    logger=trt.Logger(trt.Logger.INFO);builder=trt.Builder(logger)
    network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser=trt.OnnxParser(network,logger)
    ok=parser.parse_from_file(str(source));errors=[str(parser.get_error(i)) for i in range(parser.num_errors)]
    write_json(out/'parser.json',{'passed':ok,'errors':errors})
    assert ok,errors
    io={'inputs':[{'name':network.get_input(i).name,'shape':list(network.get_input(i).shape),'dtype':str(network.get_input(i).dtype)} for i in range(network.num_inputs)],
        'outputs':[{'name':network.get_output(i).name,'shape':list(network.get_output(i).shape),'dtype':str(network.get_output(i).dtype)} for i in range(network.num_outputs)]}
    write_json(out/'io.json',io)
    config=builder.create_builder_config();config.clear_flag(trt.BuilderFlag.TF32)
    config.set_flag(trt.BuilderFlag.EDITABLE_TIMING_CACHE)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,2*1024**3)
    config.builder_optimization_level=3;config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
    config.hardware_compatibility_level=trt.HardwareCompatibilityLevel.NONE
    cache=config.create_timing_cache(b'');assert config.set_timing_cache(cache,False)
    begin=time.time();print('BUILD START',algorithm,stage,flush=True)
    plan=builder.build_serialized_network(network,config);assert plan is not None
    path=out/f'{stage}.engine'
    with path.open('xb') as f:f.write(bytes(plan))
    with (out/'timing.cache').open('xb') as f:f.write(bytes(config.get_timing_cache().serialize()))
    del plan,model,network,parser,builder,config,cache;gc.collect()
    runtime=trt.Runtime(logger);engine=runtime.deserialize_cuda_engine(path.read_bytes());assert engine is not None
    inspector=engine.create_engine_inspector()
    layers=json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    write_json(out/'engine_layers.json',layers)
    write_json(out/'build.json',{'passed':True,'algorithm':algorithm,'stage':stage,
        'seconds':time.time()-begin,'engine_sha256':sha(path),'engine_bytes':path.stat().st_size,
        'device_memory_size_v2':engine.device_memory_size_v2,'engine_layers':engine.num_layers,
        'source_INT4_weights':len(qweights),'native_inference_validation':'pending'})
    path.chmod(0o444);print('BUILD COMPLETE',algorithm,stage,path.stat().st_size,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True)
    p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--stage',required=True)
    p.add_argument('--algorithm',required=True);a=p.parse_args()
    try:main(a.source,a.out,a.stage,a.algorithm)
    except Exception:
        if a.out.exists():write_json(a.out/'failure.json',{'traceback':traceback.format_exc()})
        raise
