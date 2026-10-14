"""Read existing artifacts and write a new evidence summary; never run inference."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
VERSION = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = VERSION / 'summary_evidence.json'
    if output.exists():
        raise FileExistsError('Frozen or existing evidence must not be overwritten')
    comparison_path = ROOT / 'versions/a100_comparison_20261008_r1/comparison.json'
    comparison = json.loads(comparison_path.read_text())
    source_manifest = ROOT / 'releases/a100_closedloop_20261008_r1/manifest.json'
    assert digest(source_manifest) == comparison['source_manifest_sha256']
    supplied = {
        'trt_float': [25, 8.12, 93.23],
        'awq': [25, 5.49, 92.93],
        'gptq': [27, 5.44, 93.73],
        'full_awq': [24, 4.47, 106.76],
        'full_gptq': [29, 4.47, 101.78],
    }
    for row in comparison['overview']:
        assert [row['successes'], round(row['device_peak_GB'], 2),
                round(row['request_p99_ms'], 2)] == supplied[row['variant']]
    supplied_modules = {
        'trt_float': [13.34, 21.86, 42.42],
        'awq': [13.17, 26.58, 42.01],
        'gptq': [13.24, 25.44, 42.18],
        'full_awq': [17.57, 26.59, 49.31],
        'full_gptq': [16.03, 25.36, 48.36],
    }
    for row in comparison['module_mean_ms_same_128_observations']:
        assert [round(row[key], 2) for key in ['vision', 'prefix', 'action_step']] == supplied_modules[row['variant']]
    inherited = {}
    for rel, entry in comparison['source_files'].items():
        path = ROOT / rel
        assert digest(path) == entry['sha256'], rel
        inherited[rel] = {'sha256': entry['sha256'], 'bytes': path.stat().st_size}
    documents = [
        'versions/trt_t0_t2_20261007_r1/T0-T2实施记录.md',
        'versions/trt_t3_20261007_r1/T3实施记录.md',
        'versions/trt_t4_20261007_r1/T4闭环与资源对比记录.md',
        'versions/trt_full4_20261007_r1/全Linear量化快速验证结果.md',
        'versions/a100_closedloop_20261008_r1/A100统一闭环测试记录.md',
        'versions/a100_comparison_20261008_r1/A100模型综合对比.md',
        'versions/a100_comparison_20261008_r1/comparison.json',
        'versions/trt_full4_20261007_r1/quantize.py',
        'versions/trt_t3_20261007_r1/runtime.py',
        'releases/a100_closedloop_20261008_r1/manifest.json',
        'releases/a100_comparison_20261008_r1/manifest.json',
        '当前实施索引_20261008_r2.md',
    ]
    for rel in documents:
        path = ROOT / rel
        inherited[rel] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    engine_stats = []
    for module in ['vision', 'action_step']:
        for algorithm in ['float', 'awq', 'gptq']:
            rel = f'runs/a100_closedloop_20261008_r1/build_{algorithm}_{module}_r1/engine_layers.json'
            path = ROOT / rel
            inherited[rel] = {'sha256': digest(path), 'bytes': path.stat().st_size}
            layers = json.loads(path.read_text())['Layers']
            consumers = {}
            for layer in layers:
                for tensor in layer.get('Inputs', []):
                    consumers.setdefault(tensor['Name'], []).append(layer)
            separate = [layer for layer in layers if layer['LayerType'] == 'kgen'
                        and '.DQ]' in layer.get('Metadata', '')]
            direct = []
            for layer in separate:
                successors = [s for tensor in layer.get('Outputs', [])
                              for s in consumers.get(tensor['Name'], [])]
                if any(s['LayerType'] in ['gemm', 'fusion'] for s in successors):
                    direct.append(layer)
            qkv_merged = [layer['Name'] for layer in layers
                          if all(tag in layer.get('Metadata', '')
                                 for tag in ['v_proj/MatMul', 'k_proj/MatMul', 'q_proj/MatMul'])]
            if algorithm != 'float':
                assert len(separate) == (55 if module == 'vision' else 77)
                assert len(direct) == len(separate)
            engine_stats.append({
                'module': module, 'algorithm': algorithm,
                'inspector_layers_including_sync_pseudolayers': len(layers),
                'visible_separate_DQ_layers': len(separate),
                'DQ_layers_with_direct_gemm_or_fusion_consumers': len(direct),
                'combined_QKV_execution_layers': qkv_merged,
                'note': 'Metadata and producer edges establish graph structure; not a kernel timing breakdown.',
            })
    deltas = {}
    for algorithm in ['awq', 'gptq']:
        old, full = supplied[algorithm], supplied['full_' + algorithm]
        modules_old, modules_full = supplied_modules[algorithm], supplied_modules['full_' + algorithm]
        deltas[algorithm] = {
            'vision_prefix_expert_mean_delta_ms': [round(b-a, 2) for a, b in zip(modules_old, modules_full)],
            'request_p99_delta_ms': round(full[2]-old[2], 2),
            'request_p99_percent_increase': (full[2]/old[2]-1)*100,
            'device_peak_percent_change': (full[1]/old[1]-1)*100,
            'basis': 'Rounded values supplied by the user; module mean deltas do not equal request P99 deltas.',
        }
    catalog = json.loads((VERSION / 'source_catalog.json').read_text())
    for source in catalog['sources']:
        snapshot = VERSION / 'source_snapshots' / source['relative_path']
        assert digest(snapshot) == source['sha256'], source['relative_path']
    evidence = {
        'version': VERSION.name,
        'created_at_beijing': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'work_type': 'documentation_and_read_only_existing_evidence_analysis',
        'new_inference_or_training_performed': False,
        'upstream_documents_verified_as_byte_identical_snapshots': len(catalog['sources']),
        'upstream_original_data_audited_in_this_run': False,
        'supplied_A100_success_memory_p99': supplied,
        'supplied_module_means_ms_vision_prefix_expert': supplied_modules,
        'historical_PyTorch_4090_supplied_row': {'successes': 24, 'trials': 30, 'device_peak_GB': 8.26, 'request_p99_ms': 241.86},
        'PyTorch_A100_measured': False,
        'full_vs_old_deltas': deltas,
        'A100_engine_structure': engine_stats,
        'unchanged_quantization_dependencies_relative_to_6_model_inference': inherited,
        'production_release_accepted': False,
    }
    with output.open('x') as handle:
        json.dump(evidence, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({'output': str(output), 'source_snapshots': len(catalog['sources']),
                      'frozen_quantization_sources_checked': len(inherited),
                      'A100_engines_inspected': len(engine_stats)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
