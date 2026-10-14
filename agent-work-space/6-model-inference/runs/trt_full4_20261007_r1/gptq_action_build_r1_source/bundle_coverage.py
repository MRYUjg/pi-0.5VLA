"""Bind full policy and execute all three engines with source/inspector audit."""
import argparse
import json
import pathlib

import numpy as np
import torch

from common import ROOT, RUN, OLD, verified, sha, write_json, runtime, config_and_transforms, prepare


def bind(algorithm):
    short=algorithm.lower();path=RUN/'bundles'/f'full_{short}.json'
    old=ROOT/f'runs/trt_t3_20261007_r1/bundles/{short}.json';verified(old)
    engines={'prefix':json.loads(old.read_text())['engines']['prefix']}
    for stage,label in [('vision',f'{short}_vision_build_r1'),('action_step',f'{short}_action_build_r1')]:
        folder=RUN/label;build=json.loads((folder/'build.json').read_text());file=folder/f'{stage}.engine'
        assert sha(file)==build['engine_sha256']
        engines[stage]={'path':str(file),'sha256':build['engine_sha256'],'bytes':file.stat().st_size}
    spec={'version':'trt_full4_20261007_r1','algorithm':algorithm,'engines':engines,
        'total_engine_file_bytes':sum(e['bytes'] for e in engines.values()),
        'all_reachable_Linear_count':451,'vision_Linear_count':163,'expert_Linear_count':167,
        'prefix_reachable_Linear_count':121,'original_prefix_bundle_sha256':sha(old),
        'activation_precision':'original BF16/FP32','denoising_steps':10,
        'source_model_instantiated_at_inference':False,'source_JAX_tested':False,
        'non_Linear_and_bias_float_remainder':True,'20Hz_accepted':False,'closedloop_accepted':False}
    write_json(path,spec);path.chmod(0o444)
    return spec


def coverage(algorithm,spec):
    out=RUN/(algorithm.lower()+'_execution_r1');out.mkdir(exist_ok=False)
    policy=runtime().Policy(spec);policy.set_profiling(True)
    cfg,inputs,outputs=config_and_transforms();pilot=OLD/'calibration_r1/pilot.json';verified(pilot)
    row=json.loads(pilot.read_text())[0];verified(pathlib.Path(row['path']))
    ob=prepare(dict(np.load(row['path'])),inputs);torch.manual_seed(row['noise_seed'])
    result=policy.infer(ob,torch.randn(1,10,32,device='cuda'),audit=True,trace=True)
    assert result['cache_immutable'] and result['finite']
    profiles=policy.profiles();write_json(out/'executed_layers.json',profiles)
    report={}
    for stage,label in [('vision','vision'),('action_step','action')]:
        prefix=algorithm.lower()
        scope=json.loads((RUN/f'{prefix}_{stage}_r1/onnx_scope.json').read_text())['nodes']
        layers=json.loads((RUN/f'{prefix}_{label}_build_r1/engine_layers.json').read_text())['Layers']
        rows=[]
        for source in scope:
            tag='[ONNX Layer: '+source['matmul']+']'
            mats=[l for l in layers if tag in str(l.get('Metadata','')) and l.get('LayerType') in ('gemm','fusion')]
            dqtag='[ONNX Layer: '+source['weight']+'.DQ]'
            dqs=[l for l in layers if dqtag in str(l.get('Metadata',''))]
            names={l['Name'] for l in mats}
            observed=[n for n in names if n in profiles[stage] and profiles[stage][n]['calls']>0]
            rows.append({'module':source['module'],'source_INT4_weight':source['weight']+'.int4',
                'GEMM_engine_layers':sorted(names),'executed_GEMM_layers':observed,
                'direct_DQ_tags':[l['Name'] for l in dqs],
                'mapping_proved':len(mats)==1 and len(observed)==1,
                'same_layer_DQ_GEMM_fusion':len(mats)==1 and any(l['Name']==mats[0]['Name'] for l in dqs),
                'tactics':[l.get('TacticName') for l in mats]})
        report[stage]={'source_Linear_count':len(scope),'directly_mapped_and_executed':sum(r['mapping_proved'] for r in rows),
            'all_directly_mapped_and_executed':all(r['mapping_proved'] for r in rows),
            'fusion_directly_visible':sum(r['same_layer_DQ_GEMM_fusion'] for r in rows),
            'executed_engine_layers':len(profiles[stage]),'rows':rows}
    write_json(out/'summary.json',{'algorithm':algorithm,'three_engines_complete_ten_steps_executed':True,
        'all_finite':True,'KV_immutable':True,'module_counts':{'vision':3,'prefix':1,'action_step':10},
        'source_Linear_INT4_count':451,'new_stage_coverage':report,
        'prefix_source_mapping_reused_frozen_T3':True,'pure_INT4_arithmetic_claimed':False,
        'unresolved_source_mappings_require_explicit_review':True})
    torch.save({'internal_actions':result['internal_actions'],'velocity_fields':result['velocity_fields']},out/'output.pt')
    print('FULL EXECUTION',algorithm,{s:r['directly_mapped_and_executed'] for s,r in report.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--algorithm',choices=['AWQ','GPTQ'],required=True)
    a=p.parse_args();coverage(a.algorithm,bind(a.algorithm))
