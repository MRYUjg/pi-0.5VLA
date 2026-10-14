"""Native strongly typed INT4 engine build with detailed inspector and cache."""
import argparse
import collections
import gc
import json
import pathlib
import subprocess
import time
import traceback

import onnx
from onnx import TensorProto
import tensorrt as trt

from common import verified, sha, write_json, SOURCE_SHA


def main(out, source, algorithm, timing_cache):
    out.mkdir(parents=True, exist_ok=False)
    onnx_sha = verified(source)
    graph = onnx.load(source, load_external_data=False)
    external = {v.value for t in graph.graph.initializer for v in t.external_data if v.key == 'location'}
    fingerprints = {name: verified(source.parent / name) for name in external}
    qweights = [t for t in graph.graph.initializer if t.data_type == TensorProto.INT4]
    assert len(qweights) == 121
    registration = {'algorithm': algorithm, 'source_model_sha256': SOURCE_SHA, 'source_ONNX_sha256': onnx_sha,
        'external_data_sha256': fingerprints, 'TensorRT': trt.__version__, 'strongly_typed': True,
        'TF32': False, 'workspace_bytes': 2 * 1024**3, 'builder_optimization_level': 3,
        'profiling_verbosity': 'DETAILED', 'editable_timing_cache': True, 'INT4_initializer_count': 121,
        'activation_precision': 'original BF16/FP32', 'hardware_compatibility': 'NONE',
        'portability': 'Development RTX4090 Linux x86_64, exact TensorRT; target4070 must rebuild/test',
        'target_4070_8GB_accepted': False, 'timing_cache_input': str(timing_cache) if timing_cache else None,
        'device': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                         '--format=csv,noheader'], text=True).strip()}
    write_json(out / 'registration.json', registration)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    ok = parser.parse_from_file(str(source))
    errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    assert ok, errors
    dtypes = collections.Counter(str(network.get_layer(i).get_output(j).dtype)
                                for i in range(network.num_layers)
                                for j in range(network.get_layer(i).num_outputs)
                                if network.get_layer(i).get_output(j) is not None)
    shapes = {'inputs': [{'name': network.get_input(i).name, 'shape': list(network.get_input(i).shape),
                          'dtype': str(network.get_input(i).dtype)} for i in range(network.num_inputs)],
              'outputs': [{'name': network.get_output(i).name, 'shape': list(network.get_output(i).shape),
                           'dtype': str(network.get_output(i).dtype)} for i in range(network.num_outputs)],
              'static_shapes': True, 'network_output_types': dict(dtypes), 'network_layers': network.num_layers}
    assert all(all(d > 0 for d in x['shape']) for x in shapes['inputs'] + shapes['outputs'])
    write_json(out / 'shape_profile.json', shapes)
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_flag(trt.BuilderFlag.EDITABLE_TIMING_CACHE)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024**3)
    config.builder_optimization_level = 3
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.hardware_compatibility_level = trt.HardwareCompatibilityLevel.NONE
    cache = config.create_timing_cache(timing_cache.read_bytes() if timing_cache else b'')
    assert config.set_timing_cache(cache, ignore_mismatch=False)
    start = time.time()
    print('BUILD START', algorithm, flush=True)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None, 'TensorRT build failed'
    with (out / 'prefix.engine').open('xb') as f:
        f.write(bytes(plan))
    with (out / 'timing.cache').open('xb') as f:
        f.write(bytes(config.get_timing_cache().serialize()))
    del plan, graph, parser, network, builder, config, cache
    gc.collect()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine((out / 'prefix.engine').read_bytes())
    assert engine is not None
    inspector = engine.create_engine_inspector()
    layers = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    write_json(out / 'engine_layers.json', layers)
    summary = {'build_passed': True, 'algorithm': algorithm, 'seconds': time.time() - start,
        'engine_sha256': sha(out / 'prefix.engine'), 'engine_bytes': (out / 'prefix.engine').stat().st_size,
        'timing_cache_sha256': sha(out / 'timing.cache'), 'timing_cache_bytes': (out / 'timing.cache').stat().st_size,
        'engine_layers': engine.num_layers, 'device_memory_size_v2': engine.device_memory_size_v2,
        'optimization_profiles': engine.num_optimization_profiles,
        'profiling_verbosity': str(engine.profiling_verbosity), 'INT4_initializers_in_source': 121,
        'reload_accuracy': 'pending', 'native_INT4_execution_coverage': 'inspect separately',
        'actual_4070_accepted': False}
    write_json(out / 'build.json', summary)
    for p in [out / 'prefix.engine', out / 'timing.cache']:
        p.chmod(0o444)
    print('BUILD COMPLETE', algorithm, summary['engine_bytes'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=pathlib.Path, required=True)
    parser.add_argument('--source', type=pathlib.Path, required=True)
    parser.add_argument('--algorithm', choices=['AWQ', 'GPTQ'], required=True)
    parser.add_argument('--timing-cache', type=pathlib.Path)
    a = parser.parse_args()
    try:
        main(a.out, a.source, a.algorithm, a.timing_cache)
    except Exception:
        if a.out.exists() and not (a.out / 'failure.json').exists():
            write_json(a.out / 'failure.json', {'traceback': traceback.format_exc()})
        raise
