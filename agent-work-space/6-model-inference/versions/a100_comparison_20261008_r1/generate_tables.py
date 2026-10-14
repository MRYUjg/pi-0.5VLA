"""Derive comparison tables from frozen A100 records; no inference rerun."""
import argparse
import hashlib
import json
import math
import pathlib

from PIL import Image, ImageDraw, ImageFont

SOURCE = 'a100_closedloop_20261008_r1'
VERSION = 'a100_comparison_20261008_r1'
SOURCE_MANIFEST_SHA = '68e55030753fd346559c69351f9d820fbee5133e25d350490ab256cc2a3a3cd3'
VARIANTS = ['trt_float', 'awq', 'gptq', 'full_awq', 'full_gptq']
LABELS = ['TensorRT float', '旧 AWQ INT4（prefix）', '旧 GPTQ INT4（prefix）',
          '全 Linear AWQ INT4', '全 Linear GPTQ INT4']
SHORT = ['TensorRT float', '旧 AWQ', '旧 GPTQ', '全 Linear AWQ', '全 Linear GPTQ']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as f:
        if isinstance(value, str):
            f.write(value)
        else:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write('\n')


def render(out, rows, modules, font_path):
    canvas = Image.new('RGB', (1800, 1260), 'white')
    draw = ImageDraw.Draw(canvas)
    fonts = {size: ImageFont.truetype(str(font_path), size) for size in [20, 24, 26, 28, 30, 32, 40]}
    dark, gray, line = '#24272B', '#60666F', '#E2E5E9'

    def text(x, y, value, size=30, color=dark):
        draw.text((x, y), value, font=fonts[size], fill=color)

    def table(y, widths, headers, body, row_height, header_size=30, body_size=32):
        xs = [40]
        for width in widths:
            xs.append(xs[-1] + width)
        for i, cell in enumerate(headers):
            font = fonts[header_size]
            w = draw.textlength(cell, font=font)
            assert w < widths[i] - 32, (cell, w, widths[i])
            x = xs[i] + 12 if i == 0 else xs[i + 1] - 16 - w
            text(x, y + 15, cell, header_size)
        draw.line((40, y + row_height, 1760, y + row_height), fill=line, width=2)
        for number, record in enumerate(body):
            top = y + row_height * (number + 1)
            for i, cell in enumerate(record):
                size = 28 if i == 0 else body_size
                w = draw.textlength(cell, font=fonts[size])
                assert w < widths[i] - 32, (cell, w, widths[i])
                x = xs[i] + 12 if i == 0 else xs[i + 1] - 16 - w
                text(x, top + 17, cell, size)
            draw.line((40, top + row_height, 1760, top + row_height), fill=line, width=1)

    text(40, 30, 'A100 综合对比', 40)
    text(40, 95, 'Panda + PandaGripper · LIBERO Spatial · 固定初态与种子 · 10 个任务 × 3 次 / 模型', 24, gray)
    body = [['原始 PyTorch', '未测', '未测', '未测']]
    for r in rows:
        body.append([r['label'], f"{r['successes']}/30，{r['success_fraction']*100:.1f}%",
                     f"{r['device_peak_GB']:.2f} GB", f"{r['request_p99_ms']:.2f} ms"])
    table(150, [650, 400, 335, 335], ['模型', '闭环成功率', '设备显存峰值', '请求 P99'], body, 75)
    text(40, 708, '显存：单模型启动至闭环结束的设备总峰值；GB = 10^9 bytes。', 24, gray)
    text(40, 748, 'P99：20 次预热后，128 个固定观测串行请求；不含相机采集与图像 base64 编码。', 24, gray)
    text(40, 808, '模块平均耗时（相同 128 个观测，ms）', 32)
    stage_rows = [[label] + [f'{m[stage]:.2f}' for m in modules] for stage, label in [
        ('vision', '视觉编码（3 次调用合计）'),
        ('prefix', 'prefix 图像与语言处理'),
        ('action_step', '动作专家（10 轮去噪合计）')]]
    table(860, [420, 260, 260, 260, 260, 260], ['模块'] + SHORT, stage_rows, 65, 26, 30)
    text(40, 1148, '模块事件包含 CPU enqueue 间隙，不能视为纯 GPU kernel 时间。', 24, gray)
    text(40, 1186, '两种 GPTQ 的数值门槛仍未通过；闭环成功率高不等于部署验收通过。', 24, gray)
    text(40, 1228, '来源：a100_closedloop_20261008_r1  |  2026-10-08  |  原始 PyTorch 本轮未测', 20, gray)
    canvas.save(out / 'A100综合对比表.png')


def main(root, out, font):
    manifest_path = root / 'releases' / SOURCE / 'manifest.json'
    assert sha(manifest_path) == SOURCE_MANIFEST_SHA
    manifest = json.loads(manifest_path.read_text())
    inputs = {}

    def read(key):
        path = root / key
        expected = manifest['files_relative_to_6_model_inference'][key]
        assert sha(path) == expected['sha256'] and path.stat().st_size == expected['bytes'], key
        inputs[key] = expected
        return json.loads(path.read_text())

    report = read(f'runs/{SOURCE}/report_r1/summary.json')
    assert report['initial_condition_alignment_passed'] and report['all_150_episodes_complete']
    rows, modules, reference_ids = [], [], None
    for variant, label in zip(VARIANTS, LABELS):
        r = report['variants'][variant]
        prefix = f'runs/{SOURCE}/closedloop_{variant}_r1/pilot/'
        cases, pilot = read(prefix + 'cases.json'), read(prefix + 'summary.json')
        assert len(cases) == 128 and pilot['warmup'] == 20 and pilot['repeat_passed'] and pilot['all_finite']
        assert pilot == r['pilot_summary']
        identifiers = [tuple(c[k] for k in ['task', 'seed', 'step', 'file_sha256', 'noise_seed']) for c in cases]
        if reference_ids is None:
            reference_ids = identifiers
        assert reference_ids == identifiers
        assert r['closedloop_memory']['single_model_resident']
        for c in cases:
            assert c['timing']['module_calls'] == {'vision': 3, 'prefix': 1, 'action_step': 10}
            assert all(math.isfinite(c['timing']['module_gpu_ms'][stage]) for stage in ['vision', 'prefix', 'action_step'])
        rows.append({'variant': variant, 'label': label, 'successes': r['successes'],
                     'trials': 30, 'success_fraction': r['success_fraction'],
                     'device_peak_GB': r['closedloop_memory']['device_peak_bytes'] / 1e9,
                     'process_peak_GB': r['closedloop_memory']['process_peak_bytes'] / 1e9,
                     'request_p99_ms': pilot['round_trip_ms']['p99'],
                     'closedloop_gate_passed': r['closedloop_original_gate_passed'],
                     'numerical_128_gate_passed': r['pilot_128_vs_A100_TRT_float']['original_gates_passed'],
                     'numerical_30_initial_gate_passed': r['heldout_30_initial_observations_vs_A100_TRT_float']['original_gates_passed']})
        modules.append({'variant': variant, **{stage: math.fsum(c['timing']['module_gpu_ms'][stage] for c in cases) / 128
                                                for stage in ['vision', 'prefix', 'action_step']}})
    data = {'version': VERSION, 'source_version': SOURCE, 'source_manifest_sha256': SOURCE_MANIFEST_SHA,
            'source_files': inputs, 'device': report['device'], 'overview': rows,
            'module_mean_ms_same_128_observations': modules,
            'PyTorch_A100_measured': False, 'inference_rerun': False,
            'request_scope': '20 warmups, then 128 serial preencoded-observation HTTP requests; camera and image base64 encoding excluded.',
            'memory_scope': 'NVML whole-device lifetime peak, including loading, warmups, pilot and closed-loop.',
            'module_scope': 'CUDA event intervals, including CPU enqueue gaps; vision 3 calls, prefix 1 call, expert 10 calls summed per request, then arithmetic mean across 128 requests.',
            'production_release_accepted': False}
    write(out / 'comparison.json', data)
    lines = ['# A100 模型综合对比', '',
             '日期：2026-10-08。依据冻结的 A100 五组统一测试重新整理，只进行数据汇总和制表，没有新增推理或修改模型、控制器及判定规则。', '',
             '## 综合指标', '',
             '| 模型 | 闭环成功率 | 设备显存峰值 | 请求 P99 |', '|---|---:|---:|---:|',
             '| 原始 PyTorch | 未测 | 未测 | 未测 |']
    for r in rows:
        lines.append(f"| {r['label']} | {r['successes']}/30，{r['success_fraction']*100:.1f}% | {r['device_peak_GB']:.2f} GB | {r['request_p99_ms']:.2f} ms |")
    lines += ['', '设备为 NVIDIA A100-SXM4-40GB。闭环采用 Panda+PandaGripper、LIBERO Spatial，每组 10 个任务各 3 个相同初态和种子。原始 PyTorch 在本轮 A100 未测试；图示中的原始 PyTorch 80% 和 241.86 ms 来自旧 4090 实验，不填入 A100 表。', '',
              '设备显存是 NVML 记录的整卡总峰值，包含加载、预热、128 观测回归和闭环；不同于模型进程显存。GB 按 10^9 bytes 计。', '',
              '请求 P99 来自 20 次预热后的 128 个相同观测串行请求，包含 JSON 请求构造、HTTP、服务端解码、预处理、完整三模块与 10 轮去噪、反归一化及响应解码。不含图像 base64 编码、相机采集、OSMesa 渲染；不是旧图中的 1000 请求基准，也不是主闭环双客户端排队时间。', '',
              '## 模块平均耗时', '',
              '| 模块（ms） | TensorRT float | 旧 AWQ | 旧 GPTQ | 全 Linear AWQ | 全 Linear GPTQ |',
              '|---|---:|---:|---:|---:|---:|']
    for stage, label in [('vision', '视觉编码，3 次调用合计'), ('prefix', 'prefix 图像与语言处理'),
                         ('action_step', '动作专家，10 轮去噪合计')]:
        lines.append('| ' + label + ' | ' + ' | '.join(f'{m[stage]:.2f}' for m in modules) + ' |')
    lines += ['', '模块值从与请求 P99 相同的 128 个观测逐请求记录求算术平均，没有用闭环分化后的输入混算。CUDA 事件区间含 CPU enqueue 间隙，不是纯 GPU kernel 时间。分模块均值相加也不能等同于端到端 P99。', '',
              '## 全量化相对旧量化', '',
              '| 模型 | 成功数变化 | 设备峰值变化 | 请求 P99 变化 |', '|---|---:|---:|---:|']
    for full, old in [(3, 1), (4, 2)]:
        a, b = rows[full], rows[old]
        lines.append(f"| {a['label']} 相对 {b['label']} | {a['successes']-b['successes']:+d} 集 | {(a['device_peak_GB']/b['device_peak_GB']-1)*100:+.1f}% | {(a['request_p99_ms']/b['request_p99_ms']-1)*100:+.1f}% |")
    lines += ['', '本轮全 Linear 量化降低设备显存峰值，但固定观测请求 P99 增加。全 AWQ 少成功 1 集，全 GPTQ 多成功 2 集；每组仅 30 集，不据此认定算法普遍优劣。', '',
              '**两种 GPTQ 的数值门槛均未通过**。全 GPTQ 29/30 不能替代数值验收；AWQ 的闭环成功率也未达到 27/30。本轮没有模型同时通过闭环与两组数值门槛，20 Hz / 实机部署未验收。', '',
              '[表格图片](A100综合对比表.png)、[机器可读数据与来源 SHA256](comparison.json)、[原始测试记录](../a100_closedloop_20261008_r1/A100统一闭环测试记录.md)。', '',
              '![A100 综合对比表](A100综合对比表.png)', '']
    write(out / 'A100模型综合对比.md', '\n'.join(lines))
    render(out, rows, modules, font)
    write(out / 'render_environment.json', {'font': str(font), 'font_sha256': sha(font),
                                          'image_pixels': [1800, 1260], 'font_not_redistributed': True})
    print('A100 COMPARISON TABLES GENERATED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    parser.add_argument('--font', type=pathlib.Path, required=True)
    args = parser.parse_args()
    main(args.root, args.out, args.font)
