"""Publish unchanged MP4 copies and summarize a completed registered campaign."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path('/mnt/data/vla-policy/openpi/gpt-rpent-control')
RUN_ID = 'eco65_zhixing_spatial10_20261007_r1'
REPORT = ROOT / 'reports/eco65_zhixing_20261007_r1'
RUN = ROOT / 'runs' / RUN_ID


def save(path, data):
    with path.open('x') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


audit = json.loads((REPORT / 'trajectory_audit.json').read_text())
media = json.loads((REPORT / 'video_check/video_manifest.json').read_text())
workflow = json.loads((RUN / 'workflow_summary.json').read_text())
assert audit['trials'] == media['tasks'] == 10
assert media['all_videos_decoded'] and media['MP4_count'] == 20
assert workflow['all_task_processes_exit_zero'] and workflow['policy_exited_zero']
assert not workflow['policy_running_after_cleanup']
assert audit['all_audits_passed'], 'retain failures and investigate; do not publish an acceptance claim'

videos = ROOT / 'videos' / RUN_ID
videos.mkdir(parents=True, exist_ok=False)
published, diagnostics = [], []
for row in media['episodes']:
    task = row['task']
    episode = RUN / f'task{task:02d}_seed0'
    for kind, original in [('dual_camera', episode / 'env/dual_camera.mp4'),
                           ('rpent_episode', episode / 'artifacts/episode.mp4')]:
        target = videos / f'task{task:02d}_{kind}.mp4'
        shutil.copy2(original, target)
        sha = hashlib.sha256(target.read_bytes()).hexdigest()
        assert sha == row[kind]['sha256']
        published.append(dict(task=task, kind=kind, relative_path=str(target.relative_to(ROOT)),
                              source=str(original.relative_to(ROOT)), sha256=sha,
                              bytes=target.stat().st_size, frames=row[kind]['frames'],
                              fps=row[kind]['fps'], seconds=row[kind]['seconds']))
    transitions = [json.loads(line) for line in (episode / 'env/transitions.jsonl').read_text().splitlines()]
    registration = json.loads((episode / 'env/registration.json').read_text())
    keys = [key for key in transitions[0]['diagnostics']['objects'] if 'bowl' in key.lower()]
    values = {}
    for key in keys:
        initial = registration['initialization_transfer']['diagnostic_before_settle']['objects'][key]
        poses = [t['diagnostics']['objects'][key] for t in transitions]
        if isinstance(initial, list) and len(initial) == 7:
            values[key] = dict(before_settle_xyz=initial[:3], first_action_xyz=poses[0][:3],
                               final_xyz=poses[-1][:3], max_center_z=max(p[2] for p in poses),
                               min_center_z=min(p[2] for p in poses))
    diagnostics.append(dict(task=task, native_success=row['native_success'], steps=row['steps'],
                            bowl_qpos_diagnostics=values,
                            final_gripper_qpos=transitions[-1]['diagnostics']['gripper_qpos'],
                            final_gripper_target=transitions[-1]['diagnostics']['gripper_target'],
                            close_action_steps=sum(t['executed_action'][6] > 0 for t in transitions),
                            open_action_steps=sum(t['executed_action'][6] < 0 for t in transitions)))
save(REPORT / 'published_videos.json', published)
save(REPORT / 'physical_diagnostics.json', diagnostics)

before = json.loads((ROOT / 'reports/eco65_zhixing_dependencies_before_20261007_r1.json').read_text())
after = json.loads((ROOT / 'reports/eco65_zhixing_dependencies_after_20261007_r1.json').read_text())
assert before == after, 'reused source changed'
candidate = json.loads((REPORT / 'candidate_lock_r4.json').read_text())
assert all(hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == entry['sha256']
           for rel, entry in candidate.items())
save(REPORT / 'final_source_check.json', dict(reused_dependencies_equal=True,
       reused_files_checked=len(before['reused_files_sha256']), candidate_files_checked=len(candidate),
       candidate_unchanged=True))

summary = dict(run_id=RUN_ID, code_version='eco65_zhixing_20261007_r4',
               native_successes=audit['successes'], trials=10, MP4_count=20,
               all_trajectory_audits_passed=audit['all_audits_passed'],
               all_videos_decoded=media['all_videos_decoded'],
               control_steps=sum(r['steps'] for r in audit['episodes']),
               VLA_requests=sum(r['VLA_requests'] for r in audit['episodes']),
               actual_planner_models=sorted({m for r in audit['episodes'] for m in r['actual_models']}),
               workflow_seconds=workflow['seconds'], policy_variant='original_torch',
               physical_hardware_tested=False, formal_multi_initial_benchmark=False,
               control_profile='legacy', gripper_speed=0.01,
               observation_post_action_force_refresh=True)
save(REPORT / 'summary.json', summary)

lines = ['# ECO65＋ZhiXing 闭环验证与视频 20261007 r1', '',
         f"正式批次 `{RUN_ID}`：Spatial 10×1，seed=0、官方初态index0，**原生成功 {audit['successes']}/10**。轨迹审计10/10通过，20个MP4全部逐帧解码通过。失败不剔除。", '',
         '真实调用链：DeepSeek文本规划 → RPent ApiAgentLoop/LiberoToolkit/LiberoPrimitives → 原精度冻结pi0.5 → 每5步重新观测 → ECO65＋ZhixingGripper原OSC/夹爪 → LIBERO原生成功标志。未加载LoRA、未使用脚本教师，没有微调或修改模型输出。上层实际模型来自API返回，见审计。', '',
         '双视角视频左侧主相机、右侧腕部相机，512×256、20FPS，含初始帧及每个动作后的帧（T+1）。RPent视频为256×256、20FPS、T帧。视频按仿真时间播放；等待模型时仿真暂停，不代表墙钟实时20Hz。播放副本与原始MP4的SHA256完全一致。', '',
         '| task | 原生任务 | 成功 | 控制步 | VLA请求 | 双视角MP4 | RPent MP4 |',
         '|---:|---|---|---:|---:|---|---|']
for row in audit['episodes']:
    task = row['task']
    lines.append(f"| {task} | {row['language']} | {'成功' if row['native_success'] else '失败'} | {row['steps']} | {row['VLA_requests']} | [播放](videos/{RUN_ID}/task{task:02d}_dual_camera.mp4) | [播放](videos/{RUN_ID}/task{task:02d}_rpent_episode.mp4) |")
lines += ['', f"共 {summary['control_steps']} 个控制步、{summary['VLA_requests']} 次VLA预测，工作流墙钟 {workflow['seconds']:.1f} 秒；任务进程及本次原模型服务正常退出，服务已关闭。", '',
          '## 与Panda基线的比较范围', '',
          '上一批Panda/PandaGripper为10/10。两批使用相同Spatial任务、seed0、官方物体初态、预算、原精度模型、去噪轮数、前5步重观测及文本规划工具范围。ECO65保留既有按ZhiXing分派的初始关节角、安装和相机，官方非机器人关节qpos/qvel按名字迁移，已从两份状态归档重算核对。', '',
          '本批控制profile固定legacy（与Panda相同），夹爪默认速度0.01；没有切换脚本教师采集使用的eco65_no_extra_pd。state末两维为ZhiXing左右主动关节角，沿用采集编码，不伪造Panda指间距。每步强制刷新观测以解决实际关节状态审计发现的缓存延迟；这是相对旧Panda适配器的观测时序差异，因此两批结果不构成严格单因素因果实验。', '',
          '结论范围为当前配置下的原模型跨机械臂/夹爪仿真闭环。一次10×1结果不代表多初态成功率、整体机械臂适配验收或实机表现。本入口仍只接view_env_state、pi0_pick、pi0_doubled、finish；规划按完整原任务调用VLA，未证明LLM自主分解/恢复的额外增益，SAM3、几何工具、完整CLI/Dashboard未测试。', '',
          '## 可复核证据', '',
          '[轨迹审计](reports/eco65_zhixing_20261007_r1/trajectory_audit.json)、[视频逐帧检查](reports/eco65_zhixing_20261007_r1/video_check/video_manifest.json)、[汇总](reports/eco65_zhixing_20261007_r1/summary.json)、[物理诊断](reports/eco65_zhixing_20261007_r1/physical_diagnostics.json)、[工作流](runs/'+RUN_ID+'/workflow_summary.json)。', '',
          '[行为定位](reports/eco65_zhixing_20261007_r1/行为定位_20261007_r1.md)、[观测编码说明](reports/eco65_zhixing_20261007_r1/观测编码说明_20261007_r1.md)。', '',
          '[实现与固定方案](versions/eco65_zhixing_20261007_r4/README.md)、[执行前源码锁](reports/eco65_zhixing_20261007_r1/candidate_lock_r4.json)、[源码不变检查](reports/eco65_zhixing_20261007_r1/final_source_check.json)。', '',
          '开发过程保留：r1环境预检因NumPy整数JSON序列化失败；r2环境预检通过，但策略链在0动作时因参考Panda渲染上下文释放失败；r3策略链执行10步、2预测、正常退出但实际关节状态审计未通过；r4刷新观测后的10步链路及独立审计通过。开发批次均不计入正式成功率，源文件和失败证据保留。', '',
          '[开发检查记录](reports/eco65_zhixing_20261007_r1/development_status.json)、[最终链路审计](reports/eco65_zhixing_20261007_r1/chain_audit_r4.json)。旧冻结版未覆盖。']
with (ROOT / 'ECO65_ZhiXing闭环验证与视频_20261007_r1.md').open('x') as file:
    file.write('\n'.join(lines) + '\n')
with (ROOT / '当前版本与文件索引_20261007_r3.md').open('x') as file:
    file.write('# 当前版本与文件索引 20261007 r3\n\n'
               '[ECO65＋ZhiXing原生模型结果与20个视频](ECO65_ZhiXing闭环验证与视频_20261007_r1.md)。\n\n'
               '实现：versions/eco65_zhixing_20261007_r4；正式运行：runs/'+RUN_ID+'；审计：reports/eco65_zhixing_20261007_r1。\n\n'
               '[Panda对照与视频](视频索引_20261007_r2.md)、[旧版本索引](当前版本与文件索引_20261007_r2.md)保留。\n\n'
               '新发布清单：releases/'+RUN_ID+'/manifest.json。文件冻结不等同整体适配验收；详见本批结论范围。\n')
print(json.dumps(summary, ensure_ascii=False))
