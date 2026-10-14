"""128 fixed observations on A100; no replay assumption across GPU models."""
import argparse,base64,json,pathlib,time,urllib.request
import numpy as np
from common import ROOT,verified,write_json


def main(variant,out,port):
    out.mkdir(exist_ok=False);p=ROOT/'runs/trt_t0_t2_20261007_r1/calibration_r1/pilot.json';verified(p)
    rows=json.loads(p.read_text());assert len(rows)==128
    records=[]
    for row in rows:
        p=pathlib.Path(row['path']);verified(p);records.append(dict(np.load(p)))
    def infer(i):
        raw=records[i];row=rows[i]
        obj={k:base64.b64encode(np.ascontiguousarray(raw[k]).tobytes()).decode() for k in ['image','wrist_image']}
        obj.update(state=raw['state'].tolist(),prompt=str(raw['prompt']),noise_seed=row['noise_seed'],client_id='a100_pilot')
        start=time.perf_counter()
        req=urllib.request.Request(f'http://127.0.0.1:{port}/infer',data=json.dumps(obj).encode(),headers={'Content-Type':'application/json'})
        d=json.loads(urllib.request.urlopen(req,timeout=300).read());assert d['variant']==variant
        a=np.asarray(d['actions']);b=np.asarray(d['internal_actions'])
        assert a.shape==(10,7) and b.shape==(1,10,32) and np.isfinite(a).all() and np.isfinite(b).all()
        return d,(time.perf_counter()-start)*1000
    for i in range(20):infer(i)
    cases=[];times=[]
    for i,row in enumerate(rows):
        d,ms=infer(i);times.append(ms)
        cases.append({**{k:row[k] for k in ['task','seed','step','file_sha256','noise_seed']},
            'env_actions':d['actions'],'internal_actions':d['internal_actions'],'timing':d['timing']})
    repeated,_=infer(0)
    diff=float(np.abs(np.asarray(repeated['internal_actions'])-np.asarray(cases[0]['internal_actions'])).max())
    write_json(out/'cases.json',cases)
    write_json(out/'summary.json',{'variant':variant,'device':'A100','observations':128,'warmup':20,
        'all_finite':True,'repeat_internal_max_abs':diff,'repeat_passed':diff<=1e-6,
        'round_trip_ms':{'p50':float(np.quantile(times,.5)),'p99':float(np.quantile(times,.99))},
        'calibration_related_not_generalization_test':True,'1000_request_or_two_hour_claim':False})
    assert diff<=1e-6,diff
    print('A100 PILOT PASS',variant,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--port',type=int,required=True)
    a=p.parse_args();main(a.variant,a.out,a.port)
