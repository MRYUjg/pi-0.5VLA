"""1,000 real-observation serial requests and replay of frozen T3 outputs."""
import argparse
import base64
import hashlib
import json
import pathlib
import time
import urllib.request

import numpy as np


def quantiles(values):
    x=np.asarray(values,float)
    return {'count':len(x),'mean':float(x.mean()),'p50':float(np.quantile(x,.5)),
        'p95':float(np.quantile(x,.95)),'p99':float(np.quantile(x,.99)),'max':float(x.max())}


def main(out, variant, port):
    out.mkdir(parents=True,exist_ok=False)
    root=pathlib.Path('/work')
    manifest=root/'releases/trt_t3_20261007_r1/manifest.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='016814fb9a9eaba7d398bb9fa7e2b3952de2995d8767b215c76ee7ed10bcbe17'
    doc=json.loads(manifest.read_text())
    rows=json.loads((root/'runs/trt_t0_t2_20261007_r1/calibration_r1/pilot.json').read_text())
    payloads=[]
    for row in rows:
        path=pathlib.Path(row['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['file_sha256']
        raw=dict(np.load(path))
        obj={k:base64.b64encode(raw[k].tobytes()).decode() for k in ['image','wrist_image']}
        obj.update(state=raw['state'].tolist(),prompt=str(raw['prompt']),noise_seed=row['noise_seed'],client_id='resource')
        payloads.append(json.dumps(obj).encode())
    if variant in ['original_torch','trt_float']:
        path=root/'runs/trt_t0_t2_20261007_r1/float_validation_r3/cases.json'
        expected_key='reference_env_actions' if variant=='original_torch' else 'tensorrt_env_actions'
    else:
        attempt='awq_infer_r2' if variant=='awq' else 'gptq_infer_r2'
        path=root/f'runs/trt_t3_20261007_r1/{attempt}/cases.json';expected_key='env_actions'
    name=str(path.relative_to(root))
    entry=doc['files_relative_to_6_model_inference'].get(name)
    expected_sha=entry['sha256'] if entry else doc['unchanged_frozen_dependencies'][name]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected_sha
    old=json.loads(path.read_text())
    def infer(index):
        start=time.perf_counter()
        req=urllib.request.Request(f'http://127.0.0.1:{port}/infer',data=payloads[index],headers={'Content-Type':'application/json'})
        result=json.loads(urllib.request.urlopen(req,timeout=300).read())
        result['client_round_trip_ms']=(time.perf_counter()-start)*1000
        assert result['variant']==variant
        actions=np.asarray(result['actions'])
        assert actions.shape==(10,7) and np.isfinite(actions).all()
        return result
    for i in range(20):infer(i)
    results=[];replay_max=0.;start=time.perf_counter()
    with (out/'requests.jsonl').open('x',buffering=1) as log:
        for i in range(1000):
            result=infer(i%128)
            results.append(result)
            log.write(json.dumps({'index':i,**result})+'\n')
            if i<128:
                diff=float(np.abs(np.asarray(result['actions'])-np.asarray(old[i][expected_key])).max())
                replay_max=max(replay_max,diff)
            if (i+1)%100==0:print('RESOURCE REQUESTS',variant,i+1,'/1000',flush=True)
    repeat=infer(0)
    repeat_max=float(np.abs(np.asarray(repeat['actions'])-np.asarray(results[0]['actions'])).max())
    server_metrics=['preprocessing_and_noise_ms','sample_wall_ms','sample_stream_elapsed_ms',
        'postprocessing_and_timing_collection_ms','model_path_ms','decode_ms','server_processing_ms']
    latency={k:quantiles([r['timing'][k] for r in results]) for k in server_metrics}
    latency['client_round_trip_ms']=quantiles([r['client_round_trip_ms'] for r in results])
    modules={k:quantiles([r['timing']['module_gpu_ms'][k] for r in results]) for k in ['vision','prefix','action_step']}
    with (out/'summary.json').open('x') as f:
        json.dump({'completed':True,'variant':variant,'requests':1000,'warmup_requests':20,
            'seconds':time.perf_counter()-start,'all_finite':True,'frozen_128_observation_replay_max_abs':replay_max,
            'replay_passed':replay_max<=1e-6,'repeat_after_1000_max_abs':repeat_max,'repeat_passed':repeat_max<=1e-6,
            'latency_ms':latency,'module_gpu_ms':modules,
            'latency_includes_timing_instrumentation':True,'includes_real_camera_capture':False,
            'source_JAX_not_benchmarked':True,'two_hour_continuous_validation':False,
            'target_4070_accepted':False},f,indent=2)
    assert replay_max<=1e-6 and repeat_max<=1e-6,(replay_max,repeat_max)
    print('RESOURCE BENCHMARK COMPLETE',variant,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True)
    p.add_argument('--variant',required=True);p.add_argument('--port',type=int,default=18879)
    a=p.parse_args();main(a.out,a.variant,a.port)
