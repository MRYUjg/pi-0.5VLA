"""Bind only SHA-checked freshly rebuilt A100 engines to the five variants."""
import json,pathlib
from common import RUN,ROOT,sha,write_json,verified


def main():
    sequence=json.loads((RUN/'build_sequence_summary.json').read_text());assert sequence['all_nine_completed']
    bundles=RUN/'bundles';bundles.mkdir(exist_ok=False)
    for variant in ['trt_float','awq','gptq','full_awq','full_gptq']:
        alg=variant.removeprefix('full_');engines={};sources={}
        for stage in ['vision','prefix','action_step']:
            label=('float' if variant=='trt_float' or (not variant.startswith('full_') and stage!='prefix') else alg)+'_'+stage
            folder=RUN/('build_'+label+'_r1');build=json.loads((folder/'build.json').read_text())
            registration=json.loads((folder/'registration.json').read_text());p=folder/(stage+'.engine')
            assert build['passed'] and sha(p)==build['engine_sha256'] and 'A100' in registration['build_GPU']
            engines[stage]={'path':str(p),'sha256':build['engine_sha256'],'bytes':p.stat().st_size}
            sources[stage]=registration
        obj={'version':'a100_closedloop_20261008_r1','variant':variant,'device':'A100-SXM4-40GB','engines':engines,
            'source_graph_registrations':sources,'same_frozen_quantized_weights':True,
            'total_engine_file_bytes':sum(e['bytes'] for e in engines.values()),'denoising_steps':10,
            'new_closedloop_acceptance':'pending','4090_speed_comparison_claimed':False}
        p=bundles/(variant+'.json');write_json(p,obj);p.chmod(0o444)
    print('FIVE A100 BUNDLES BOUND',flush=True)


if __name__=='__main__':main()
