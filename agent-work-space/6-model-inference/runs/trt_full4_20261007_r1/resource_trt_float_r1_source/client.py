"""Matched 128-action regression and serial resource test with fresh encoding."""
import argparse
import base64
import json
import pathlib
import time
import urllib.request

import numpy as np

from common import ROOT, OLD, verified, write_json, frozen_module


def stats(values):
    a=np.asarray(values,float)
    return {'count':len(a),'mean':float(a.mean()),'p50':float(np.quantile(a,.5)),
        'p95':float(np.quantile(a,.95)),'p99':float(np.quantile(a,.99)),'max':float(a.max())}


def main(out,variant,port,count):
    out.mkdir(exist_ok=False);pilot=OLD/'calibration_r1/pilot.json';verified(pilot)
    rows=json.loads(pilot.read_text());observations=[]
    for row in rows:
        verified(pathlib.Path(row['path']));observations.append(dict(np.load(row['path'])))
    oldpath=OLD/'float_validation_r3/cases.json';verified(oldpath);old=json.loads(oldpath.read_text())
    assert len(rows)==len(old)==128
    def infer(i):
        raw=observations[i];row=rows[i]
        start=time.perf_counter()
        obj={k:base64.b64encode(np.ascontiguousarray(raw[k]).tobytes()).decode() for k in ('image','wrist_image')}
        obj.update(state=raw['state'].tolist(),prompt=str(raw['prompt']),noise_seed=row['noise_seed'],client_id='full4')
        payload=json.dumps(obj).encode();encoded=time.perf_counter()
        req=urllib.request.Request(f'http://127.0.0.1:{port}/infer',data=payload,headers={'Content-Type':'application/json'})
        result=json.loads(urllib.request.urlopen(req,timeout=300).read())
        actions=np.asarray(result['actions'],np.float32);internal=np.asarray(result['internal_actions'],np.float32)
        assert actions.shape==(10,7) and internal.shape==(1,10,32)
        assert np.isfinite(actions).all() and np.isfinite(internal).all()
        end=time.perf_counter();result['client_full_ms']=(end-start)*1000
        result['client_HTTP_ms']=(end-encoded)*1000
        assert result['variant']==variant
        return result
    for i in range(20):infer(i)
    results=[];start=time.perf_counter()
    with (out/'requests.jsonl').open('x',buffering=1) as log:
        for i in range(count):
            r=infer(i%128);results.append(r);log.write(json.dumps({'index':i,**r})+'\n')
            if (i+1)%100==0:print('REQUESTS',variant,i+1,'/',count,flush=True)
    serial_seconds=time.perf_counter()-start
    repeat=infer(0)
    cases=[]
    for row,r,prior in zip(rows,results[:128],old):
        assert tuple(row[k] for k in ('task','seed','step','file_sha256'))==tuple(prior[k] for k in ('task','seed','step','file_sha256'))
        cases.append({**{k:row[k] for k in ('task','seed','step','file_sha256','noise_seed')},
            'env_actions':r['actions'],'internal_actions':r['internal_actions']})
    write_json(out/'cases.json',cases)
    candidate=np.asarray([r['env_actions'] for r in cases]);oldfloat=np.asarray([r['tensorrt_env_actions'] for r in old])
    original=np.asarray([r['reference_env_actions'] for r in old])
    metrics=frozen_module('full4_original_quality_gates',ROOT/'versions/trt_t3_20261007_r1/compare.py')
    repeat_env=float(np.abs(np.asarray(repeat['actions'])-np.asarray(results[0]['actions'])).max())
    repeat_internal=float(np.abs(np.asarray(repeat['internal_actions'])-np.asarray(results[0]['internal_actions'])).max())
    latency={'client_full_ms':stats([r['client_full_ms'] for r in results]),
        'client_HTTP_ms':stats([r['client_HTTP_ms'] for r in results]),
        'model_path_ms':stats([r['timing']['model_path_ms'] for r in results])}
    modules={s:stats([r['timing']['module_gpu_ms'][s] for r in results]) for s in ('vision','prefix','action_step')}
    summary={'completed':True,'variant':variant,'requests':count,'warmup':20,'observations':128,
        'all_finite':True,'serial_seconds':serial_seconds,'completed_requests_per_second_including_logging':count/serial_seconds,
        'latency_ms':latency,'module_timeline_ms':modules,'events_may_include_host_enqueue_gaps':True,
        'fresh_client_payload_encoding_included':True,'captured_input_already_224':True,
        'raw_internal_action_in_response_audit_overhead_included_for_all_variants':True,
        'quality_vs_frozen_TRT_float':metrics.quality(oldfloat,candidate),
        'quality_cumulative_vs_original_Torch':metrics.quality(original,candidate),
        'repeat_env_max_abs':repeat_env,'repeat_internal_max_abs':repeat_internal,
        'repeat_passed':repeat_internal<=1e-6,'source_JAX_not_tested':True,
        'closedloop_success_rate_tested':False,'real_camera_or_rendering_included':False,
        'two_hour_tested':False,'formal_200_tested':False,
        '20Hz_latency_screen_passed':latency['client_full_ms']['p99']<=50,
        '128_pilot_is_calibration_related_regression_not_generalization':True}
    if variant=='trt_float':
        summary['frozen_float_env_replay_max_abs']=float(np.abs(candidate-oldfloat).max())
        assert summary['frozen_float_env_replay_max_abs']<=1e-6,summary
    write_json(out/'summary.json',summary)
    assert summary['repeat_passed'],summary
    print('BENCHMARK COMPLETE',variant,'P99',latency['client_full_ms']['p99'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True)
    p.add_argument('--variant',required=True);p.add_argument('--port',type=int,default=18883)
    p.add_argument('--count',type=int,default=1000);a=p.parse_args();main(a.out,a.variant,a.port,a.count)
