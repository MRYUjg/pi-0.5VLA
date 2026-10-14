"""Publish the registered existing-profile run and its paired legacy comparison."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path('/mnt/data/vla-policy/openpi/gpt-rpent-control')
RUN_ID = 'eco65_zhixing_deploy_spatial10_20261007_r1'
LEGACY_ID = 'eco65_zhixing_spatial10_20261007_r1'
REPORT = ROOT / 'reports/eco65_zhixing_deploy_20261007_r1'
RUN = ROOT / 'runs' / RUN_ID


def load(path):
    return json.loads(path.read_text())


def save(path, data):
    with path.open('x') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


audit = load(REPORT / 'trajectory_audit.json')
legacy = load(ROOT / 'reports/eco65_zhixing_20261007_r1/trajectory_audit.json')
media = load(REPORT / 'video_check/video_manifest.json')
workflow = load(RUN / 'workflow_summary.json')
assert audit['trials'] == media['tasks'] == 10
assert audit['all_audits_passed'] and media['all_videos_decoded']
assert media['MP4_count'] == 20
assert workflow['all_task_processes_exit_zero'] and workflow['policy_exited_zero']
assert not workflow['policy_running_after_cleanup']
videos = ROOT / 'videos' / RUN_ID
videos.mkdir(parents=True, exist_ok=False)
published, paired = [], []
for row in media['episodes']:
    task = row['task']
    episode = RUN / f'task{task:02d}_seed0'
    old = ROOT / 'runs' / LEGACY_ID / f'task{task:02d}_seed0'
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
    current = load(episode / 'env/registration.json')
    reference = load(old / 'env/registration.json')
    first = json.loads((episode / 'vla/predictions.jsonl').read_text().splitlines()[0])
    first_old = json.loads((old / 'vla/predictions.jsonl').read_text().splitlines()[0])
    fields = ['initial_state_sha256', 'bddl_sha256', 'XML_sha256', 'initial_main_sha256',
              'initial_wrist_sha256', 'settled_state_sha256', 'arm_init_qpos',
              'gripper_state_joint_names', 'gripper_state_indices', 'gripper_speed']
    paired.append(dict(task=task, registration_checks={f:current[f] == reference[f] for f in fields},
                       first_input_checks={f:first[f] == first_old[f] for f in
                           ['image_sha256', 'wrist_sha256', 'state8', 'prompt', 'noise_seed']},
                       first_prediction_exact=first['raw_actions'] == first_old['raw_actions'],
                       deployed_native_success=row['native_success'],
                       legacy_native_success=legacy['episodes'][task]['native_success']))
save(REPORT / 'published_videos.json', published)
save(REPORT / 'paired_initial_and_first_prediction.json', paired)
before = load(ROOT / 'reports/eco65_zhixing_deploy_dependencies_before_20261007_r1.json')
after = load(ROOT / 'reports/eco65_zhixing_deploy_dependencies_after_20261007_r1.json')
assert before == after
candidate = load(REPORT / 'candidate_lock_r5.json')
assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == entry['sha256']
           for name, entry in candidate.items())
save(REPORT / 'final_source_check.json', dict(reused_dependencies_equal=True,
       reused_files_checked=len(before['reused_files_sha256']), candidate_files_checked=len(candidate),
       candidate_unchanged=True))
equal_init = sum(all(row['registration_checks'].values()) and all(row['first_input_checks'].values())
                 for row in paired)
equal_pred = sum(row['first_prediction_exact'] for row in paired)
summary = dict(run_id=RUN_ID, code_version='eco65_zhixing_20261007_r5',
               native_successes=audit['successes'], trials=10, MP4_count=20, total_two_profiles_MP4=40,
               legacy_successes=legacy['successes'], legacy_trials=10,
               all_trajectory_audits_passed=True, all_videos_decoded=True,
               control_steps=sum(row['steps'] for row in audit['episodes']),
               VLA_requests=sum(row['VLA_requests'] for row in audit['episodes']),
               actual_planner_models=sorted({m for row in audit['episodes'] for m in row['actual_models']}),
               workflow_seconds=workflow['seconds'], policy_variant='original_torch',
               control_profile='eco65_no_extra_pd', initialization_profile='legacy',
               gripper_speed=0.01, paired_initial_inputs_equal=equal_init,
               paired_first_predictions_exact=equal_pred,
               physical_hardware_tested=False, formal_multi_initial_benchmark=False)
save(REPORT / 'summary.json', summary)

lines = ['# ECO65＋ZhiXing 既有部署配置闭环复验与视频', '',
         f"**既有部署配置 eco65_no_extra_pd：{audit['successes']}/10；legacy：{legacy['successes']}/10。** 两套配置各Spatial 10×1、seed0、官方初态index0，分别统计，不合并成同一成功率。原生DeepSeek＋RPent＋原精度pi0.5，未微调，未使用脚本教师。", '',
         'Panda/PandaGripper上一批为10/10。ECO65两批均保留已有ZhiXing初态、安装/相机、默认速度0.01、OSC增益/尺度、原生物体与成功谓词。第二批复用1–2已有eco65_no_extra_pd，初始化及10步静置仍为legacy，首个模型动作前切换已有profile；不调整其他参数或生产默认值。', '',
         f"两组初始物理状态、双RGB/8维state、首个模型输入逐项一致：{equal_init}/10；相同噪声种子下首个10×7预测完全一致：{equal_pred}/10。详见配对检查。两组后续动作随各自实际观测闭环变化。", '',
         '两批共40个MP4全部逐帧解码通过，均保存失败全过程。双视角512×256、20FPS、T+1帧，左主相机/右腕部相机；RPent主视角256×256、20FPS、T帧。副本SHA256与原始文件一致。按仿真时间播放，等待模型时暂停仿真，不证明墙钟实时20Hz。', '',
         '| task | 原生任务 | legacy结果/步数 | 既有部署结果/步数 | 部署双视角 | 部署RPent | legacy双视角 |',
         '|---:|---|---|---|---|---|---|']
for row in audit['episodes']:
    task = row['task']
    old = legacy['episodes'][task]
    describe = lambda x:f"{'成功' if x['native_success'] else '失败'}/{x['steps']}"
    lines.append(f"| {task} | {row['language']} | {describe(old)} | {describe(row)} | [播放](videos/{RUN_ID}/task{task:02d}_dual_camera.mp4) | [播放](videos/{RUN_ID}/task{task:02d}_rpent_episode.mp4) | [播放](videos/{LEGACY_ID}/task{task:02d}_dual_camera.mp4) |")
lines += ['', f"部署批次共{summary['control_steps']}控制步、{summary['VLA_requests']}次VLA请求、墙钟{workflow['seconds']:.1f}秒。10条轨迹审计通过，任务与本次模型服务正常退出，服务关闭。", '',
          '## 结论范围', '',
          (f"当前既有部署配置下，原模型在本批完成{audit['successes']}/10任务；不能据此宣称新机械臂＋夹爪已实现高成功率即插即用适配。" if audit['successes'] < 10 else
           '当前既有部署配置下，本批10个固定初态全部成功；多初态统计和整体适配仍未验收。'), '',
          '这是仿真原模型闭环，不是ECO65实机验证。上层为DeepSeek文本/工具反馈，pi0.5接双RGB；按完整原任务反复调用VLA，尚未证明LLM自主分解/恢复的额外增益。只接view_env_state、pi0_pick、pi0_doubled、finish；完整CLI/Dashboard、SAM3及几何工具未测试。', '',
          'state姿态保持原LIBEROrobot0_eef_quat，夹爪末两位沿用采集主动关节编码；没有验证完整8维姿态定义与采集逐项等价。相对旧Panda适配器每步强制刷新观测，解决实际关节状态审计发现的缓存延迟。因此Panda与ECO两批不构成严格只改变形态的单因素实验；ECO两个profile则按本批配对检查评估。', '',
          '## 证据', '',
          '[闭合协议核对](reports/eco65_zhixing_deploy_20261007_r1/闭合协议核对_20261007_r1.md)、[原生工具边界](reports/eco65_zhixing_deploy_20261007_r1/原生工具适配边界_20261007_r1.md)。', '',
          '[部署轨迹审计](reports/eco65_zhixing_deploy_20261007_r1/trajectory_audit.json)、[20个视频检查](reports/eco65_zhixing_deploy_20261007_r1/video_check/video_manifest.json)、[汇总](reports/eco65_zhixing_deploy_20261007_r1/summary.json)、[配对初态/预测](reports/eco65_zhixing_deploy_20261007_r1/paired_initial_and_first_prediction.json)、[工作流](runs/'+RUN_ID+'/workflow_summary.json)。', '',
          '[部署实现与固定方案](versions/eco65_zhixing_20261007_r5/README.md)、[真实10步链路审计](reports/eco65_zhixing_deploy_20261007_r1/chain_audit_r5.json)、[源码检查](reports/eco65_zhixing_deploy_20261007_r1/final_source_check.json)。', '',
          '[legacy完整结果与20个视频](ECO65_ZhiXing闭环验证与视频_20261007_r1.md)、[legacy行为定位](reports/eco65_zhixing_20261007_r1/行为定位_20261007_r1.md)、[观测编码说明](reports/eco65_zhixing_20261007_r1/观测编码说明_20261007_r1.md)、[Panda视频与基线](视频索引_20261007_r2.md)。', '',
          '旧冻结版本、开发失败和legacy正式批次全部保留。新发布清单releases/'+RUN_ID+'/manifest.json；文件冻结不扩大验收范围。']
with (ROOT / 'ECO65_ZhiXing部署配置复验与视频_20261007_r2.md').open('x') as file:
    file.write('\n'.join(lines) + '\n')
with (ROOT / '当前版本与文件索引_20261007_r4.md').open('x') as file:
    file.write('# 当前版本与文件索引 20261007 r4\n\n'
               '[ECO65＋ZhiXing两套配置结果与40个MP4](ECO65_ZhiXing部署配置复验与视频_20261007_r2.md)。\n\n'
               '最新实现versions/eco65_zhixing_20261007_r5；原精度模型与生产默认参数未修改。正式部署批次runs/'+RUN_ID+'。\n\n'
               '[旧ECO65 legacy索引](当前版本与文件索引_20261007_r3.md)、[Panda索引](当前版本与文件索引_20261007_r2.md)保留。\n\n'
               '仿真闭环结果与整体/实机适配验收区分，具体限制见本批记录。\n')
print(json.dumps(summary, ensure_ascii=False))
