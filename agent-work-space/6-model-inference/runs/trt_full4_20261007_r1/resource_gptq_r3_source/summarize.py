"""Report measured results without promoting failed gates to acceptance."""
import json
import pathlib

ROOT=pathlib.Path('/mnt/data/vla-policy/openpi/agent-work-space/6-model-inference')
VERSION='trt_full4_20261007_r1';RUN=ROOT/'runs'/VERSION


def read(p):return json.loads(p.read_text())


def maxabs(a,b):
    if isinstance(a,list):
        assert len(a)==len(b)
        return max(maxabs(x,y) for x,y in zip(a,b))
    return abs(a-b)


def write(p,d):
    with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    results={};reloads={}
    for v in ['trt_float','awq','gptq','full_awq','full_gptq']:
        folder=RUN/('resource_'+v+'_r1')
        original_memory=read(folder/'memory_summary.json')
        attempt='r1'
        if not original_memory['isolated_compute'] and (RUN/('resource_'+v+'_r3')/'benchmark/summary.json').exists():
            attempt='r3';folder=RUN/('resource_'+v+'_r3')
        summary=read(folder/'benchmark/summary.json');memory=read(folder/'memory_summary.json')
        assert summary['requests']==1000 and summary['all_finite']
        p=RUN/'bundles'/f'{v}.json' if v.startswith('full_') else ROOT/'runs/trt_t3_20261007_r1/bundles'/('float.json' if v=='trt_float' else v+'.json')
        bundle=read(p);engine_bytes=sum(e['bytes'] for e in bundle['engines'].values())
        if attempt=='r1':code=read(RUN/('benchmark_sequence_'+str(len(results)+1)+'_exit.json'))['exit_code']
        else:
            matches=[r for r in read(RUN/'recheck_sequence_summary.json')['results'] if v in r['command']]
            assert len(matches)==1;code=matches[0]['exit_code']
        results[v]={'benchmark':summary,'memory':memory,'engine_bytes':engine_bytes,
                    'selected_attempt':attempt,'original_r1_memory':original_memory,
                    'resource_process_actual_exit_code':code,
                    'performance_measurement_isolated':memory['isolated_compute']}
    for v in ['full_awq','full_gptq']:
        a=read(RUN/('resource_'+v+'_r1')/'benchmark/cases.json')
        b=read(RUN/('resource_'+v+'_r2')/'benchmark/cases.json')
        assert len(a)==len(b)==128
        for x,y in zip(a,b):
            assert tuple(x[k] for k in ['task','seed','step','file_sha256','noise_seed'])==tuple(y[k] for k in ['task','seed','step','file_sha256','noise_seed'])
        env=max(maxabs(x['env_actions'],y['env_actions']) for x,y in zip(a,b))
        internal=max(maxabs(x['internal_actions'],y['internal_actions']) for x,y in zip(a,b))
        pids=[read(RUN/('resource_'+v+'_'+r)/'launch.json')['host_pid'] for r in ['r1','r2']]
        metas=[read(RUN/('resource_'+v+'_'+r)/'server/ready.json')['metadata'] for r in ['r1','r2']]
        reloads[v]={'observations':128,'env_max_abs':env,'internal_max_abs':internal,
                    'distinct_host_pids':pids[0]!=pids[1],'same_engine_metadata':metas[0]==metas[1],
                    'passed':env<=1e-6 and internal<=1e-6 and pids[0]!=pids[1] and metas[0]==metas[1]}
    data={'version':VERSION,'variants':results,'independent_process_reload':reloads,
          'source_Linear_INT4_count':451,'source_reachable_Linear_parameters':2716168192,
          'all_parameters_4bit':False,'closedloop_success_rate_tested':False,
          'formal_200_and_two_hour_tested':False,'actual_8GB_hardware_tested':False,
          '20Hz_accepted':False,'new_quality_failures_waived':False}
    write(RUN/'comparison.json',data)
    names={'trt_float':'TensorRT float','awq':'旧 AWQ（prefix）','gptq':'旧 GPTQ（prefix）',
           'full_awq':'全 Linear AWQ','full_gptq':'全 Linear GPTQ'}
    lines=['# 全 Linear INT4 快速验证结果','',
           '日期：2026-10-07。版本：`trt_full4_20261007_r1`。只验证一个 group128 配置，原始 pi0.5 LIBERO 权重、Panda 观测、固定噪声与 10 轮去噪。没有更改控制器、采集行为、原数值审计或运行时计算逻辑。','',
           '## 实际量化范围','',
           'vision 163、prefix 121、action_step 167，共 451 个图可达 Linear，原始 Linear 权重共 2,716,168,192 个参数。prefix 复用冻结的 AWQ/GPTQ；新 vision/expert 使用 200 个已有 Panda 校准观测，expert 覆盖全部 10 个时刻。embedding、卷积、bias、norm 非 Linear 参数等仍是浮点，不能称所有参数均为 4bit。','',
           'TensorRT 对 vision 的 K=4304 分组输入解析失败，因此将其补零至 4352；action_in 的 K=32 补至 128。输入和权重同时补零，外部维度保持不变。量化包含小投影、时间投影和条件 Linear，没有选择性精度回退。','',
           '## 同路径实测','',
           '每个模型单独启动，20 次预热后串行 1000 次请求。客户端每次重新编码观测，包含 HTTP、完整模型、解码和有限值检查；请求使用已采集的 224 图像，不包含相机或仿真渲染。所有变体均返回原始内部动作用于审计；日志开销计入串行吞吐。','',
           '| 模型 | 引擎文件 GB | 进程峰值 GB | 设备峰值 GB | 完整请求 P50 / P99 ms | 串行请求/s | 独占 GPU |',
           '|---|---:|---:|---:|---:|---:|---|']
    for v,r in results.items():
        s=r['benchmark'];m=r['memory'];t=s['latency_ms']['client_full_ms']
        lines.append(f"| {names[v]} | {r['engine_bytes']/1e9:.3f} | {m['process_peak_bytes']/1e9:.3f} | {m['device_peak_bytes']/1e9:.3f} | {t['p50']:.2f} / {t['p99']:.2f} | {s['completed_requests_per_second_including_logging']:.2f} | {'是' if m['isolated_compute'] else '否'} |")
    contaminated=[v for v,r in results.items() if not r['performance_measurement_isolated']]
    rechecked=[v for v,r in results.items() if r['selected_attempt']=='r3']
    if rechecked:
        lines+=['', '初次 r1 的 '+ '、'.join(names[v] for v in rechecked)+' 受其他 GPU 任务干扰，资源进程实际退出 1。已保留原始记录，在其他任务结束后使用相同配置补测 r3；上表选用 r3，其他模型选用无干扰的 r1。']
    if contaminated:
        lines+=['', '性能测量期间存在其他 GPU 计算进程，NVML 原始记录及 foreign_GPU_compute_pids 已保留。上表为共享 GPU 条件下观察值，不能作为独占 GPU 的速度或设备显存验收；资源进程实际非零退出不改成通过。进程自身显存单独统计。']
    lines+=['','各模块事件区间平均耗时如下，vision 含 3 次调用、expert 含 10 次调用。事件可能包含 CPU enqueue 间隙，不能当作纯 GPU kernel 时间。','',
            '| 模型 | vision ms | prefix ms | expert 10 轮 ms | 完整模型路径 P99 ms |',
            '|---|---:|---:|---:|---:|']
    for v,r in results.items():
        s=r['benchmark'];m=s['module_timeline_ms']
        lines.append(f"| {names[v]} | {m['vision']['mean']:.2f} | {m['prefix']['mean']:.2f} | {m['action_step']['mean']:.2f} | {s['latency_ms']['model_path_ms']['p99']:.2f} |")
    lines+=['','## 动作数值回归','',
            '128 个相同观测与种子的动作，参照冻结 TensorRT float，直接复用旧数值门槛。该样本与校准相关，只用于数值回归；通过也不能代表任务成功率。运动偏差为六维动作差绝对值除以 2。','',
            '| 模型 | 运动 MAE | 运动 P99 | 夹爪方向分歧 | clipping 增量百分点 | 原门槛 |',
            '|---|---:|---:|---:|---:|---|']
    for v,r in results.items():
        q=r['benchmark']['quality_vs_frozen_TRT_float']
        lines.append(f"| {names[v]} | {q['motion_mae']:.6f} | {q['motion_p99_abs']:.6f} | {q['gripper_disagreements']}/{q['confident_gripper_count']} ({q['gripper_disagreement_fraction']*100:.3f}%) | {q['clipping_rate_increase']*100:.3f} | {'通过' if q['original_gates_passed'] else '未通过'} |")
    lines+=['','门槛：运动 MAE≤0.01、P99≤0.05；参考夹爪绝对值>0.2 时方向分歧≤0.5%；clipping 增量≤1 个百分点。原始 PyTorch 累计偏差另存 comparison.json，旧 T1 边界漂移不自动豁免新量化错误。','',
            '## 运行与验收边界','']
    for v,r in reloads.items():
        lines.append(f"- {names[v]} 另起进程复测 128 个观测：内部动作最大差 {r['internal_max_abs']:.9g}、环境动作最大差 {r['env_max_abs']:.9g}；独立加载复现{'通过' if r['passed'] else '未通过'}。")
    lines+=['- 每个完整模型 1000 次推理均返回有限动作，重复同一输入检查和容器退出记录保留。没有完成两小时稳定性测试。',
            '- 三个 TensorRT 引擎实际执行，prefix 量化审计继承冻结 T3；新模块的 source/engine/profiler 对照另存 execution_r1/summary.json。INT4 权重并不表示全部计算为 INT4。',
            '- 20Hz、LIBERO 闭环成功率、正式 200 集及真实 8GB 设备部署均未验收。本次没有生成新的成功率。',
            '- 旧版本、旧模型和结果只读保留；新校准、模型、日志和结果均位于新版本目录。','',
            '结果入口：[comparison.json](../../runs/trt_full4_20261007_r1/comparison.json)。原始请求、逐层审计、内存采样和实际退出码见同级 run 目录。','',
            '## 本次结论','']
    baseline=results['trt_float']
    for v in ['full_awq','full_gptq']:
        r=results[v];q=r['benchmark']['quality_vs_frozen_TRT_float'];t=r['benchmark']['latency_ms']['client_full_ms']['p99']
        saving=1-r['memory']['process_peak_bytes']/baseline['memory']['process_peak_bytes']
        lines.append(f"- {names[v]}：进程峰值显存比 float 减少 {saving*100:.1f}%；完整请求 P99 {t:.2f}ms；数值回归{'通过' if q['original_gates_passed'] else '未通过'}。")
    lines+=['- 本次全 Linear INT4 验证证明了量化压缩与完整推理路径；没有达到 50ms / 20Hz。权重压缩不会自动带来相同比例的速度提升，当前完整请求和模块时间已经实测。',
            '- AWQ 可作为后续闭环验证候选；GPTQ 的失败项需保留，本次没有做精度恢复、参数搜索或新闭环验收。结论限于本次配置，不代表两种算法的普遍优劣。']
    report=ROOT/'versions'/VERSION/'全Linear量化快速验证结果.md'
    with report.open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'variants':{v:{'p99_ms':r['benchmark']['latency_ms']['client_full_ms']['p99'],
        'quality_passed':r['benchmark']['quality_vs_frozen_TRT_float']['original_gates_passed'],
        'isolated':r['performance_measurement_isolated']} for v,r in results.items()},
        'reloads':reloads},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
