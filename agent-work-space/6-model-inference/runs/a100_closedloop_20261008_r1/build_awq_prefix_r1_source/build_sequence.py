"""Rebuild the same nine source graphs on A100; no requantization."""
import json,pathlib,subprocess,time
ROOT=pathlib.Path('/mnt/data/vla-policy/openpi/agent-work-space/6-model-inference')
VERSION='a100_closedloop_20261008_r1';SRC=ROOT/'versions'/VERSION;RUN=ROOT/'runs'/VERSION
sources=[]
for stage in ['vision','prefix','action_step']:
    sources.append(('float_'+stage,stage,'float','trt_t0_t2_20261007_r1/t1_attempt_r6/'+stage+'.onnx'))
for alg in ['awq','gptq']:
    sources.append((alg+'_prefix','prefix',alg.upper(),'trt_t0_t2_20261007_r1/'+alg+'_r1/prefix_int4.onnx'))
    for stage in ['vision','action_step']:
        sources.append((alg+'_'+stage,stage,alg.upper(),'trt_full4_20261007_r1/'+alg+'_'+stage+'_r1/'+stage+'_int4.onnx'))
results=[]
for label,stage,alg,source in sources:
    name='build_'+label+'_r1';begin=time.time()
    command=['python3',str(SRC/'execute.py'),'--name',name,'build.py',
        '--source','/work/runs/'+source,'--out','/work/runs/'+VERSION+'/'+name,
        '--stage',stage,'--algorithm',alg]
    print('START',name,flush=True);code=subprocess.run(command).returncode
    results.append({'name':name,'command':command,'exit_code':code,'seconds':time.time()-begin})
    with (RUN/(name+'_host_exit.json')).open('x') as f:json.dump(results[-1],f,indent=2)
    if code:break
with (RUN/'build_sequence_summary.json').open('x') as f:json.dump({'results':results,'all_nine_completed':len(results)==9 and all(r['exit_code']==0 for r in results)},f,indent=2)
raise SystemExit(0 if len(results)==9 and all(r['exit_code']==0 for r in results) else 1)
