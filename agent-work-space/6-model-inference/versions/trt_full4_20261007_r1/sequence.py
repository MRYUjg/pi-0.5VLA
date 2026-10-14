"""Run new named attempts serially; retain each actual exit code."""
import argparse
import json
import pathlib
import subprocess
import time

ROOT=pathlib.Path('/mnt/data/vla-policy/openpi/agent-work-space/6-model-inference')
VERSION='trt_full4_20261007_r1';RUN=ROOT/'runs'/VERSION
SRC=ROOT/'versions'/VERSION;CRUN='/work/runs/'+VERSION


def main(phase):
    results=[]
    if phase=='gptq':
        commands=[]
        for stage,label in [('action_step','action'),('vision','vision')]:
            name='gptq_'+stage+'_r1'
            commands.append(['python3',str(SRC/'execute.py'),'--name',name,'quantize.py',
                '--out',CRUN+'/'+name,'--stage',stage,'--algorithm','GPTQ',
                '--calibration',CRUN+'/capture_r1'])
            name='gptq_'+label+'_build_r1'
            commands.append(['python3',str(SRC/'execute.py'),'--name',name,'build.py',
                '--source',CRUN+'/gptq_'+stage+'_r1/'+stage+'_int4.onnx',
                '--out',CRUN+'/'+name,'--stage',stage,'--algorithm','GPTQ'])
        commands.append(['python3',str(SRC/'execute.py'),'--name','gptq_execution_r1',
                         'bundle_coverage.py','--algorithm','GPTQ'])
    else:
        if phase=='benchmark':variants=['trt_float','awq','gptq','full_awq','full_gptq']
        elif phase=='recheck':
            variants=[v for v in ['trt_float','awq','gptq','full_awq','full_gptq']
                if not json.loads((RUN/('resource_'+v+'_r1')/'memory_summary.json').read_text())['isolated_compute']]
        else:variants=['full_awq','full_gptq']
        attempt={'benchmark':'r1','reload':'r2','recheck':'r3'}[phase]
        count=128 if phase=='reload' else 1000
        commands=[['python3',str(SRC/'resources.py'),'--variant',v,
                   '--attempt',attempt,'--count',str(count)] for v in variants]
    for command in commands:
        begin=time.time();print('START',command,flush=True)
        code=subprocess.run(command).returncode
        results.append({'command':command,'exit_code':code,'seconds':time.time()-begin})
        with (RUN/(phase+'_sequence_'+str(len(results))+'_exit.json')).open('x') as f:
            json.dump(results[-1],f,indent=2);f.write('\n')
        if code and phase=='gptq':break
    with (RUN/(phase+'_sequence_summary.json')).open('x') as f:
        json.dump({'phase':phase,'results':results,'all_exit_zero':all(r['exit_code']==0 for r in results)},f,indent=2);f.write('\n')
    raise SystemExit(0 if all(r['exit_code']==0 for r in results) else 1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['gptq','benchmark','reload','recheck'],required=True)
    main(p.parse_args().phase)
