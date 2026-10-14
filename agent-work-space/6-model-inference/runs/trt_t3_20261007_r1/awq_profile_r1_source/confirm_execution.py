"""Join optimized source mappings to TensorRT's actual per-layer callbacks."""
import argparse
import json
import pathlib

from common import sha, write_json


def main(out, coverage, profile):
    mapping = json.loads(coverage.read_text())
    trace = json.loads(profile.read_text())
    timings = trace['profiles']['prefix']
    rows = []
    for row in mapping['layers']:
        names = set([row['DQ_engine_layer'], row['GEMM_engine_layer']])
        witnesses = {name: timings[name] for name in names}
        assert all(v['calls'] > 0 and v['total_ms'] >= 0 for v in witnesses.values())
        rows.append({'module': row['module'], 'fused_DQ_GEMM': row['fused_DQ_GEMM'],
                     'executed_layers': witnesses})
    assert len(rows) == 121 and mapping['coverage_passed']
    gems = set(row['GEMM_engine_layer'] for row in mapping['layers'])
    dqs = set(row['DQ_engine_layer'] for row in mapping['layers'])
    write_json(out, {'passed': True, 'algorithm': mapping['algorithm'],
        'source_INT4_layers_with_execution_witness': len(rows),
        'unique_executed_GEMM_layers': len(gems), 'unique_executed_DQ_layers': len(dqs),
        'source_mapping_sha256': sha(coverage), 'profile_sha256': sha(profile),
        'method': 'DETAILED engine inspector source mapping joined to IProfiler execution callbacks',
        'all_arithmetic_pure_integer_INT4': False,
        'latency_or_closed_loop_accepted': False, 'layers': rows})
    print('EXECUTION WITNESS PASS', mapping['algorithm'], len(rows), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=pathlib.Path, required=True)
    p.add_argument('--coverage', type=pathlib.Path, required=True)
    p.add_argument('--profile', type=pathlib.Path, required=True)
    a = p.parse_args()
    main(a.out, a.coverage, a.profile)
