"""Trace every target INT4 DQ and MatMul through the optimized layer report."""
import argparse
import collections
import json
import pathlib
import re

from common import verified, write_json, sha


def main(out, build, source_scope, log):
    assert not out.exists()
    verified(source_scope)
    scope = json.loads(source_scope.read_text())['nodes']
    layers = json.loads((build / 'engine_layers.json').read_text())['Layers']
    registration = json.loads((build / 'registration.json').read_text())
    producer = {v['Name']: layer for layer in layers for v in layer.get('Outputs', [])}
    rows = []
    for source in scope:
        name = source['module'].removeprefix('paligemma_with_expert.paligemma.model.') + '.weight.DQ'
        dq = [l for l in layers if '[ONNX Layer: ' + name + ']' in l.get('Metadata', '')]
        mat = [l for l in layers if l['LayerType'] in ['gemm', 'fusion'] and
               '[ONNX Layer: ' + source['matmul'] + ']' in l.get('Metadata', '')]
        assert len(dq) <= 1 and len(mat) == 1, (name, len(dq), len(mat))
        members = [s for s in scope if '[ONNX Layer: ' + s['matmul'] + ']' in mat[0].get('Metadata', '')]
        assert members and len({s['shape'][0] for s in members}) == 1
        if not dq:
            # Q/K/V and gate/up may merge constant weights. The compiler keeps
            # only one original DQ tag. Require an explicit producer edge and
            # the exact combined matrix dimensions before mapping hidden tags.
            merged_shape = [1, source['shape'][0], sum(s['shape'][1] for s in members)]
            tags = ['[ONNX Layer: ' + s['module'].removeprefix('paligemma_with_expert.paligemma.model.')
                    + '.weight.DQ]' for s in members]
            if mat[0]['LayerType'] == 'fusion' and any(t in mat[0].get('Metadata', '') for t in tags):
                assert any(v['Dimensions'][-1] == merged_shape[1] for v in mat[0]['Inputs'])
                assert any(v['Dimensions'][-1] == merged_shape[2] for v in mat[0]['Outputs'])
                dq = [mat[0]]
            else:
                candidates = [producer[v['Name']] for v in mat[0].get('Inputs', [])
                              if v.get('Dimensions') == merged_shape and v['Name'] in producer]
                dq = [d for d in candidates if any(t in d.get('Metadata', '') for t in tags)]
            assert len(members) > 1 and len(dq) == 1, (name, merged_shape, len(dq))
        fused = dq[0]['Name'] == mat[0]['Name']
        rows.append({'module': source['module'], 'source_INT4_shape': source['shape'],
            'INT4_parameters': source['shape'][0] * source['shape'][1],
            'DQ_engine_layer': dq[0]['Name'], 'GEMM_engine_layer': mat[0]['Name'],
            'fused_DQ_GEMM': fused, 'GEMM_tactic': mat[0].get('TacticName'),
            'merged_source_modules': [s['module'] for s in members],
            'DQ_tag_directly_visible': '[ONNX Layer: ' + name + ']' in dq[0].get('Metadata', ''),
            'GEMM_inputs': mat[0].get('Inputs'), 'GEMM_outputs': mat[0].get('Outputs'),
            'classification': 'fused_INT4_dequantization_and_BF16_GEMM' if fused
                              else 'INT4_dequantization_then_separate_BF16_GEMM'})
    assert len(rows) == 121
    weights = re.findall(r'Total Weights Memory: (\d+) bytes', log.read_text())
    assert len(weights) == 1
    report = {'coverage_passed': True, 'algorithm': registration['algorithm'],
        'reachable_INT4_layers': len(rows), 'INT4_parameters': sum(r['INT4_parameters'] for r in rows),
        'fused_DQ_GEMM_layers': sum(r['fused_DQ_GEMM'] for r in rows),
        'separate_dequantization_GEMM_layers': sum(not r['fused_DQ_GEMM'] for r in rows),
        'builder_total_weights_memory_bytes': int(weights[0]),
        'persistent_weights_compressed': True, 'all_arithmetic_pure_integer_INT4': False,
        'precision_note': 'TensorRT INT4 weight-only: packed weights, original floating activations/GEMM; separate DQ fallback explicitly counted',
        'runtime_kernel_execution_confirmation': 'native inference plus per-layer profiler required',
        'engine_layers_sha256': sha(build / 'engine_layers.json'), 'layers': rows}
    write_json(out, report)
    print('COVERAGE', report['algorithm'], len(rows), 'fused', report['fused_DQ_GEMM_layers'],
          'separate_DQ', report['separate_dequantization_GEMM_layers'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=pathlib.Path, required=True)
    p.add_argument('--build', type=pathlib.Path, required=True)
    p.add_argument('--source-scope', type=pathlib.Path, required=True)
    p.add_argument('--log', type=pathlib.Path, required=True)
    a = p.parse_args()
    main(a.out, a.build, a.source_scope, a.log)
