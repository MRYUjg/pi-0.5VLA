import argparse
import collections
import gc
import pathlib
import traceback
import os
import json

import onnx
import torch

from common import load_model, ROOT, sha, write_json, selected_language
from subgraphs import Vision, Prefix, ActionStep
from runtime import build, Engine, metrics
from onnx_export import register,register_attention_rounding


def main(out, reference, reuse=None, attention_rounding=False):
    out.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    register_attention_rounding() if attention_rounding else register()
    net, _ = load_model()
    data = torch.load(reference, weights_only=True)
    specs = {
        'vision': (Vision(net).eval(), (data['vision'].cuda(),), ['image'], ['image_embedding']),
        'prefix': (Prefix(net).eval(), tuple(x.cuda() for x in data['prefix']),
                   ['image_embeddings', 'tokens', 'pad_mask', 'position_ids', 'attention_mask'],
                   ['prefix_pad_mask'] + [f'{kind}_{i}' for i in range(18) for kind in ['key', 'value']]),
        'action_step': (ActionStep(net).eval(), tuple(x.cuda() for x in data['action']),
                        ['noisy_actions', 'time_embedding', 'position_ids', 'attention_mask'] +
                        [f'{kind}_{i}' for i in range(18) for kind in ['key', 'value']], ['v_t'])}
    # Gates are registered before results, inherited from the r4 motion budget.
    write_json(out / 'gates.json', {'internal_final_action_mae_max': .01, 'internal_final_action_p99_max': .05,
               'requires_finite_all_outputs': True, 'prefix_cache_read_only': True,
               'target_device_performance_acceptance': False})
    exports = {}
    with torch.inference_mode():
        for name, (module, args, names, outputs) in specs.items():
            path = out / (name + '.onnx')
            if reuse and (not attention_rounding or name=='vision') and all((reuse/(name+s)).exists() for s in ['.onnx','.engine','_export.json']):
                old_graph=onnx.load(reuse/(name+'.onnx'),load_external_data=False)
                locations={v.value for t in old_graph.graph.initializer for v in t.external_data if v.key=='location'}
                for filename in [name+'.onnx',name+'.engine',*locations]:
                    src=reuse/filename;dst=out/filename
                    assert not dst.exists()
                    os.link(src,dst)
                exports[name]=json.loads((reuse/(name+'_export.json')).read_text())
                exports[name]['reused_from']=str(reuse)
                write_json(out/(name+'_export.json'),exports[name])
                if name=='prefix':write_json(out/'onnx_layer_map.json',json.loads((reuse/'onnx_layer_map.json').read_text()))
                print('REUSE VERIFIED ARTIFACT',name,sha(path),flush=True)
                del old_graph
                continue
            print('EXPORT', name, flush=True)
            torch.onnx.export(module, args, str(path), opset_version=20, dynamo=False,
                input_names=names, output_names=outputs, do_constant_folding=False,
                training=torch.onnx.TrainingMode.EVAL)
            # The pinned Torch legacy exporter stops at opset 20. The official
            # ONNX converter upgrades the floating graph to 21 before INT4 is added.
            old=onnx.load(path,load_external_data=False)
            upgraded=onnx.version_converter.convert_version(old,21)
            onnx.save(upgraded,path)
            del old,upgraded
            gc.collect()
            onnx.checker.check_model(str(path))
            m = onnx.load(path, load_external_data=False)
            exports[name] = {'opset': m.opset_import[0].version,'torch_export_opset':20,'conversion':'ONNX official 20-to-21',
                'vision_sdpa_export':'FP32 scores/probabilities/value accumulation, original BF16 interfaces; separately validated',
                'attention_matmul_explicit_fp32_accumulation_bf16_output':attention_rounding,
                'operators': dict(collections.Counter(n.op_type for n in m.graph.node)),
                'inputs': [v.name for v in m.graph.input], 'outputs': [v.name for v in m.graph.output],
                'initializers': len(m.graph.initializer)}
            if name == 'prefix':
                nodes = []
                for n in m.graph.node:
                    if n.op_type == 'MatMul' and any(t in n.name for t in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']):
                        nodes.append({'node': n.name, 'input': n.input[0], 'weight': n.input[1], 'output': n.output[0]})
                # Prefill returns only K/V. Last-layer q/o/MLP cannot affect any
                # cache tensor and are legitimately removed by graph reachability.
                expected_unused={f'paligemma_with_expert.paligemma.model.language_model.layers.17.{s}'
                    for s in ['self_attn.q_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']}
                producer={o:n for n in m.graph.node for o in n.output}
                exported=set()
                for n in nodes:
                    t=producer[n['weight']]
                    assert t.op_type=='Transpose'
                    module_name='paligemma_with_expert.paligemma.model.'+t.input[0].removesuffix('.weight')
                    n['module']=module_name;exported.add(module_name)
                assert set(selected_language(net))-exported==expected_unused,(len(nodes),set(selected_language(net))-exported)
                write_json(out / 'onnx_layer_map.json', {'source_eligible_layers':126,'reachable_layers':len(nodes),
                    'unused_layers':sorted(expected_unused),'reason':'last layer outputs unused by prefix K/V', 'nodes':nodes})
            del m
            print('BUILD', name, flush=True)
            exports[name]['engine'] = build(path, out / (name + '.engine'))
            write_json(out / (name + '_export.json'), exports[name])
    write_json(out / 'exports.json', exports)
    # Drop source GPU weights before the three engines are loaded together.
    del specs, module, net
    gc.collect(); torch.cuda.empty_cache()
    vision = Engine(out / 'vision.engine'); prefix = Engine(out / 'prefix.engine'); step = Engine(out / 'action_step.engine')
    with torch.inference_mode():
        image_embeddings = [vision(im.cuda())[0] for im in data['all_images']]
        image_ref = data['prefix'][0].cuda()
        image_check = metrics(image_ref, torch.cat(image_embeddings, 1))
        pref_inputs = (torch.cat(image_embeddings, 1), *(x.cuda() for x in data['prefix'][1:]))
        pref = prefix(*pref_inputs)
        cache_checks = [metrics(a.cuda(), b) for a,b in zip(data['prefix_outputs'][1:], pref[1:])]
        frozen = [v.clone() for v in pref[1:]]
        x = data['noise'].cuda()
        steps = []
        times = data['time_embeddings'].cuda()
        for i in range(10):
            v, = step(x, times[i:i+1], *(a.cuda() for a in data['action'][2:4]), *pref[1:])
            teacher_forced, = step(data['step_inputs'][i].cuda(), times[i:i+1],
                *(a.cuda() for a in data['action'][2:4]), *pref[1:])
            steps.append({'step': i, 'finite': bool(torch.isfinite(v).all()), 'max_abs_v_t': float(v.abs().max()),
                'v_t_teacher_forced': metrics(data['step_outputs'][i].cuda(), teacher_forced),
                'v_t_accumulated_drift': metrics(data['step_outputs'][i].cuda(), v)})
            x = x + torch.tensor(-.1, device='cuda', dtype=torch.float32) * v
        final = metrics(data['float_actions'].cuda(), x)
        immutable = all(torch.equal(a,b) for a,b in zip(frozen,pref[1:]))
        passed = final['mae'] <= .01 and final['p99_abs'] <= .05 and final['finite'] and immutable
        torch.save({'actions': x.cpu()}, out / 'trt_float_actions.pt')
    write_json(out / 'validation.json', {'passed': passed, 'image_embedding': image_check,
        'prefix_kv': cache_checks, 'steps': steps, 'final_actions': final, 'prefix_cache_immutable': immutable,
        '4070_validated': False, 'quantized_engine_built': False})
    assert passed, 'Floating TensorRT reference failed accuracy gates; see validation.json'
    print('T1 PASS', flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--reference',type=pathlib.Path,required=True);p.add_argument('--reuse',type=pathlib.Path);p.add_argument('--attention-rounding',action='store_true');a=p.parse_args()
    try:
        main(a.out,a.reference,a.reuse,a.attention_rounding)
    except Exception:
        a.out.mkdir(parents=True,exist_ok=True)
        write_json(a.out/'failure.json',{'status':'failed','traceback':traceback.format_exc()});raise
