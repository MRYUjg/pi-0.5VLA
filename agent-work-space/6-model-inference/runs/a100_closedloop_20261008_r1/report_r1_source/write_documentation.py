"""Write a new report from actual audited results, leaving old releases untouched."""
import argparse
import json
import pathlib

VERSION = 'a100_closedloop_20261008_r1'
VARIANTS = ['trt_float', 'awq', 'gptq', 'full_awq', 'full_gptq']
LABELS = {'trt_float': 'TensorRT float', 'awq': '旧 AWQ（prefix）',
          'gptq': '旧 GPTQ（prefix）', 'full_awq': '全 Linear AWQ',
          'full_gptq': '全 Linear GPTQ'}


def state(passed):
    return '通过' if passed else '未通过'


def pairs(rows):
    return '、'.join(f"task{r['task']}/init{r['init_index']}" for r in rows) or '无'


def save(path, content):
    with path.open('x') as f:
        f.write(content)


def main(root):
    run = root / 'runs' / VERSION
    version = root / 'versions' / VERSION
    data = json.loads((run / 'report_r1/summary.json').read_text())
    audit = json.loads((run / 'artifact_audit_r1/verified.json').read_text())
    assert data['all_150_episodes_complete'] and data['initial_condition_alignment_passed'] and audit['passed']
    assert json.loads((run / 'closedloop_sequence_r1_exit.json').read_text())['exit_code'] == 0
    rows = data['variants']
    lines = [
        '# A100 五组统一闭环测试记录', '',
        f'日期：2026-10-08。新版本：`{VERSION}`。实际设备为 NVIDIA A100-SXM4-40GB（40 GB），TensorRT 10.13.3.9。', '',
        '已完成 Panda+PandaGripper、LIBERO Spatial 的五组闭环，共 150 集：每组 10 个任务，各 3 个固定初态。下表成功数来自环境 `_check_success()`，没有排除失败、增加超时时间或更改成功判定。', '',
        '## 实际成功率', '',
        '| 模型 | 成功数 | 成功率 | 相对本轮 float 丢失 / 恢复 / 净损失 | 闭环原门槛 |',
        '|---|---:|---:|---:|---|',
    ]
    for variant in VARIANTS:
        r = rows[variant]
        p = r['paired_vs_A100_TRT_float']
        lines.append(f"| {LABELS[variant]} | {r['successes']}/30 | {r['success_fraction']*100:.2f}% | {len(p['lost'])} / {len(p['recovered'])} / {p['net_loss']} | {state(r['closedloop_original_gate_passed'])} |")
    lines += ['', '闭环门槛沿用既定 ≥27/30，且相对同设备 float 净损失≤1 集、初态全部一致。该门槛是开发筛查；30 集不是正式 200 集验收。成功率的 Wilson 描述区间保存在 summary.json，不能由少量集数的差异断言算法普遍优劣。', '',
              '## 全量化相对旧量化', '']
    for full, old in [('full_awq', 'awq'), ('full_gptq', 'gptq')]:
        r = rows[full]
        p = r['paired_vs_same_algorithm_prefix_only']
        delta = r['successes'] - rows[old]['successes']
        lines += [f"- {LABELS[full]} {r['successes']}/30，对比 {LABELS[old]} {rows[old]['successes']}/30，成功数变化 {delta:+d} 集。对应场景新增失败 {len(p['lost'])} 集（{pairs(p['lost'])}），恢复 {len(p['recovered'])} 集（{pairs(p['recovered'])}）。"]
    lines += ['', '这次已补齐全量化模型闭环成功率。动作误差和任务成功率分别报告：相同初态也会因输出差异而产生不同后续观测，因此后续轨迹动作差不被当作同输入量化误差。', '',
              '## 分任务结果', '',
              '| task | 指令 | float | 旧 AWQ | 旧 GPTQ | 全 AWQ | 全 GPTQ |',
              '|---|---|---:|---:|---:|---:|---:|']
    scenes = json.loads((run / 'preflight_r1/scenes.json').read_text())
    for task in range(10):
        language = next(r['language'] for r in scenes if r['task'] == task)
        cells = [str(task), language] + [f"{rows[v]['per_task'][task]['successes']}/3" for v in VARIANTS]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '对应场景的视频示例（不增加或替换测试分母）：', '']
    for full, old in [('full_awq', 'awq'), ('full_gptq', 'gptq')]:
        lost = rows[full]['paired_vs_same_algorithm_prefix_only']['lost']
        if lost:
            ep = lost[0]
            task, index = ep['task'], ep['init_index']
            base = f'../../runs/{VERSION}/closedloop_'
            tail = f'_r1/task{task:02d}/init{index:02d}.mp4'
            lines.append(f"- task{task}/init{index}：[旧模型成功视频]({base}{old}{tail})、[{LABELS[full]}失败视频]({base}{full}{tail})。")
    lines += ['', '## 同输入动作回归', '',
              '参照本轮 A100 TensorRT float，复用原数值指标及门槛：六维运动差为 `abs(candidate-reference)/2`，MAE≤0.01、P99≤0.05；参考夹爪绝对值>0.2 时方向分歧≤0.5%；未裁剪七维动作越界率增量≤1 个百分点。', '',
              '| 模型 | 样本 | 运动 MAE | 运动 P99 | 夹爪方向分歧 | 越界增量 pp | 数值门槛 |',
              '|---|---|---:|---:|---:|---:|---|']
    for variant in VARIANTS[1:]:
        for key, label in [('pilot_128_vs_A100_TRT_float', '旧 128 观测'),
                           ('heldout_30_initial_observations_vs_A100_TRT_float', '30 初始观测')]:
            q = rows[variant][key]
            lines.append(f"| {LABELS[variant]} | {label} | {q['motion_mae']:.6f} | {q['motion_p99_abs']:.6f} | {q['gripper_disagreements']}/{q['confident_gripper_count']} ({q['gripper_disagreement_fraction']*100:.3f}%) | {q['clipping_rate_increase']*100:+.6f} | {state(q['original_gates_passed'])} |")
    lines += ['', '旧 128 观测与校准相关，仅用于回归。30 个首观测是五组指纹一致的开发初态，每个计入完整 10 步动作，共 300 个夹爪值。', '',
              '**历史未通过项仍保留**：4090 上旧 GPTQ 的夹爪分歧 7/1280（0.547%）未通过；全 GPTQ 的运动 P99 0.053015 和夹爪分歧 15/1280（1.172%）未通过。旧 T1 float 越界率漂移也没有因用户允许推进而改判通过。本轮 A100 结果不能撤销这些记录，也不能单凭闭环成功数批准部署。', '',
              '## 引擎、初态与运行协议', '',
              '使用同源原始 pi05_libero BF16 权重值，未加载 ECO65/ZhiXing 微调权重。旧 AWQ/GPTQ 只量化 prefix；全 Linear 量化包括 vision 163、prefix 121、action_step 167，共 451 个图可达 Linear。embedding、卷积、bias、norm 等非 Linear 参数仍为浮点，不是所有参数均为 4bit。', '',
              '九个 TensorRT 引擎均由冻结 ONNX 和外部权重在 A100 上重新构建：SM80、强类型、TF32 关闭、builder optimization 3、2 GiB workspace、hardware compatibility NONE、空 timing cache。源图及外部权重 SHA256 与旧冻结清单一致，没有重新校准、重新量化或改变数学计算链路。', '',
              '模型镜像固定为 `sha256:cf96556ab8b55eb3f80770d2a684d6961c0128c8100660a8cc4c7e61ecc60369`；仿真镜像固定为 `sha256:1ac69906b0b877c9fe1c2964794aa5b90112850da01967a445658e41a958f284`。模型包和源码与冻结 T3 一致；仿真包、源码、控制器和场景登记与冻结 T4 一致。', '',
              '固定官方初态索引 0、1、2，种子 `940000+100*task+init_index`，噪声 `episode_seed*1000+replan_index`。Panda/PandaGripper、原控制器、20 Hz 仿真控制、10 步等待、最多 220 策略步、10 次去噪、每次返回 10 步并执行前 5 步。执行前按既有规则裁剪至 [-1,1]；原始 chunk 和实际执行动作均保存。', '',
              '五组对应场景的 settled 状态、首帧图像/state、XML、控制器 SHA256 均相同。每次只驻留一个 GPU 模型，两个 CPU 仿真客户端；50 个任务容器实际退出 0，五个模型服务正常退出 0，没有 OOM。NVML 覆盖加载至关闭，未出现其他 GPU 计算 PID。每组 128 观测输出有限，同进程重复首输入内部动作最大差不超过 1e-6；实测值逐组保存在 pilot/summary.json。', '',
              '## 当前 A100 资源与延迟观察', '',
              '| 模型 | 引擎文件 GB | 模型进程峰值 GB | 设备峰值 GB | pilot HTTP P50 / P99 ms | 闭环 server model path P99 ms | 双客户端相机步骤到动作 P99 ms |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for variant in VARIANTS:
        r = rows[variant]
        m = r['closedloop_memory']
        pilot = r['pilot_summary']['round_trip_ms']
        bundle = json.loads((run / 'bundles' / (variant + '.json')).read_text())
        lines.append(f"| {LABELS[variant]} | {bundle['total_engine_file_bytes']/1e9:.3f} | {m['process_peak_bytes']/1e9:.3f} | {m['device_peak_bytes']/1e9:.3f} | {pilot['p50']:.2f} / {pilot['p99']:.2f} | {r['closedloop_server_model_path_ms']['p99']:.2f} | {r['closedloop_latency_ms']['camera_step_to_action_ms']['p99']:.2f} |")
    historical = data['A100_float_vs_frozen_4090_float_same_pilot_128']
    lines += ['', 'GB 为十进制 10^9 bytes。pilot 20 次预热后测 128 次预编码观测请求，不包含图像 base64 编码或相机采集。闭环 server model path 是已有 instrumentation 的服务内时间；主闭环 HTTP 和相机步骤到动作时间包含双客户端排队、OSMesa 软件渲染及视频记录，不能作为单客户端纯推理吞吐。模块事件也可能包含 CPU enqueue 间隙，详见 summary.json。', '',
              f"本轮 A100 float 与旧冻结 4090 float 的相同 128 观测动作最大绝对差为 {historical['max_abs_env_actions']:.8f}。该跨 GPU 数值差只作描述，不能当作量化误差；本轮量化质量始终以同设备 A100 float 为参考。", '',
              '本轮没有新增 1000 次串行资源基准、单客户端延迟探针、两小时稳定性测试、正式 200 集、真实 8 GB 设备或实机测试。20 Hz 是仿真时间控制频率，不表示完整推理已达到 50 ms。A100 数据与旧 4090 数据分别保留，不跨 GPU 宣称速度改善。', '',
              '## 证据与冻结', '',
              f"全部 {audit['full_video_decodes']} 个主闭环视频已从头到尾解码，共 {audit['decoded_frames']} 帧；初态数组与逐步动作关系已回读验证。失败集均完整执行到 220 步，没有重复测试替代失败分母。", '',
              '- [机器可读结果](../../runs/' + VERSION + '/report_r1/summary.json)、[150 集结果表](../../runs/' + VERSION + '/report_r1/episodes.csv)。',
              '- [视频与动作回读](../../runs/' + VERSION + '/artifact_audit_r1/verified.json)、[初态与环境对齐](../../runs/' + VERSION + '/simulation_alignment_r1.json)。',
              '- [实际总退出码](../../runs/' + VERSION + '/closedloop_sequence_r1_exit.json)、[九个引擎构建](../../runs/' + VERSION + '/build_sequence_summary.json)。',
              '- [协议](protocol.json)、[新冻结清单](../../releases/' + VERSION + '/manifest.json)、[服务器校验](../../releases/' + VERSION + '/server_verify_r1.json)。', '',
              '每集证据位于 `runs/' + VERSION + '/closedloop_<variant>_r1/taskXX/`：`initYY.json` 保存 chunk、逐步动作、成功标志和计时，`initYY.mp4` 保存双相机视频，另有首观测 npz、settled 状态 npy 和初始化指纹。服务器目录为 `/mnt/data/vla-policy/openpi/agent-work-space/6-model-inference`。', '',
              '文件冻结只表示版本和证据可追溯，不等于适配部署验收通过。旧冻结版本和实验均只读保留。本地为部分证据镜像，不含大体量引擎、权重或全部视频；完整证据以服务器新 run 目录为准。', '',
    ]
    save(version / 'A100统一闭环测试记录.md', '\n'.join(lines))
    save(version / 'README.md', f'# A100 统一闭环测试\n\n版本 `{VERSION}`，五组各 30 集，共 150 集。\n\n[测试记录](A100统一闭环测试记录.md)、[协议](protocol.json)、[机器可读结果](../../runs/{VERSION}/report_r1/summary.json)。\n\n复用冻结模型、控制协议和审计门槛，只在 A100 重建引擎并补齐统一闭环对照。实验完成、文件冻结与部署验收分别记录，未通过项没有豁免。\n')
    index = '\n'.join([
        '# 当前实施索引：A100 五组统一闭环', '',
        '日期：2026-10-08。当前新增执行版本为 `' + VERSION + '`，接续 4090 全 Linear 快速验证。', '',
        '[A100 统一闭环测试记录](versions/' + VERSION + '/A100统一闭环测试记录.md)、[机器可读结果](runs/' + VERSION + '/report_r1/summary.json)、[冻结清单](releases/' + VERSION + '/manifest.json)。', '',
        *[f"- {LABELS[v]}：{rows[v]['successes']}/30（{rows[v]['success_fraction']*100:.2f}%），闭环原门槛{state(rows[v]['closedloop_original_gate_passed'])}。" for v in VARIANTS], '',
        '全部 150 集完成，五组初态指纹一致；动作和视频均已回读。原始权重、控制器、采样和成功判定没有改变。', '',
        '实际设备 A100-SXM4-40GB。旧 GPTQ 数值失败仍保留；本轮不表示 20Hz、正式 200 集、两小时、真实 8 GB 或实机验收。冻结文件不等于部署通过。', '',
        '[旧 4090 全 Linear 结果](versions/trt_full4_20261007_r1/全Linear量化快速验证结果.md)、[旧 T4 闭环结果](versions/trt_t4_20261007_r1/T4闭环与资源对比记录.md)、[旧 r6 索引](当前实施索引_20261007_r6.md)。', '',
    ])
    save(root / '当前实施索引_20261008_r1.md', index)
    print('NEW A100 DOCUMENTATION WRITTEN', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    main(parser.parse_args().root)
