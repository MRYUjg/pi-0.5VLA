import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/mnt/data/vla-policy/openpi/gpt-rpent-control')
RID = 'spatial10_video_20261007_r2'
video = json.loads((ROOT / 'reports' / RID / 'video_check_r2/video_manifest.json').read_text())
audit = json.loads((ROOT / 'reports' / (RID + '_audit.json')).read_text())
workflow = json.loads((ROOT / 'runs' / RID / 'workflow_summary.json').read_text())
before = json.loads((ROOT / 'reports' / (RID + '_dependencies_before.json')).read_text())
after = json.loads((ROOT / 'reports' / (RID + '_dependencies_after.json')).read_text())
assert audit['trials'] == 10 and audit['all_audits_passed']
assert video['tasks'] == 10 and video['MP4_count'] == 20 and video['all_videos_decoded']
assert workflow['all_task_processes_exit_zero'] and workflow['policy_exited_zero']
assert not workflow['policy_running_after_cleanup'] and before == after

target = ROOT / 'videos' / RID
target.mkdir(parents=True, exist_ok=False)
rows = ['| task | 原生任务 | 原生成功 | 控制步 | 双视角 MP4 | RPent MP4 |',
        '|---:|---|---|---:|---|---|']
manifest = []
for entry in video['episodes']:
    task = entry['task']
    episode = ROOT / 'runs' / RID / ('task%02d_seed0' % task)
    for key, source, suffix in [('dual_camera', episode / 'env/dual_camera.mp4', 'dual_camera'),
                                ('rpent_episode', episode / 'artifacts/episode.mp4', 'rpent_episode')]:
        destination = target / ('task%02d_%s.mp4' % (task, suffix))
        shutil.copy2(source, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        assert digest == entry[key]['sha256']
        manifest.append(dict(task=task, kind=key, file=str(destination.relative_to(ROOT)),
                             source=str(source.relative_to(ROOT)), sha256=digest,
                             bytes=destination.stat().st_size, frames=entry[key]['frames'],
                             fps=entry[key]['fps'], seconds=entry[key]['seconds']))
    links = ['[播放](videos/{}/task{:02d}_{}.mp4)'.format(RID, task, kind)
             for kind in ['dual_camera', 'rpent_episode']]
    rows.append('| {} | {} | {} | {} | {} | {} |'.format(task, entry['language'],
                 '成功' if entry['native_success'] else '失败', entry['steps'], *links))
with (ROOT / 'reports' / RID / 'published_videos.json').open('x') as file:
    json.dump(manifest, file, ensure_ascii=False, indent=2)

report = '''# 闭环 MP4 视频索引 20261007 r2

工作目录：`/mnt/data/vla-policy/openpi/gpt-rpent-control`。

本次重新执行真实 DeepSeek → RPent → 原精度 pi0.5 → MuJoCo Panda/PandaGripper 闭环，沿用冻结 `panda_libero_20261007_r1`，没有修改控制、模型、采集或成功判定。新运行目录为 `runs/spatial10_video_20261007_r2`，上一批次原样保留。

LIBERO Spatial 10任务各1个官方初态（seed=0）原生成功 {}/10，轨迹审计10/10通过。保留20个MP4，全部逐帧解码通过。

- 双视角：左侧主视角、右侧腕部视角，512×256，20 FPS，包含初态与每个实际控制步，帧数=控制步+1。
- RPent原生：主视角，256×256，20 FPS，帧数=控制步数。
- 视频按仿真动作时间播放。等待模型响应时仿真暂停，因此没有录制等待期间的重复帧。

## 可播放视频

'''.format(video['native_successes']) + '\n'.join(rows) + '''

## 实际运行与证据

完整工作流墙钟 {:.3f} 秒，{} 个控制步、{} 次VLA请求。两路CPU仿真共享一个串行原精度模型服务；所有任务进程和模型服务退出码0，模型服务已关闭。

- [轨迹审计](reports/{}_audit.json)：模型身份、每次最新状态反馈、预测/实际动作、时间和视频T+1关联全部通过。
- [MP4逐帧检查](reports/{}/video_check_r2/video_manifest.json)：实际帧数、FPS、尺寸、长度、SHA256与图像变化。
- [本页发布视频清单](reports/{}/published_videos.json)：每个播放文件与原始视频的SHA256相同，未剪辑或改动。
- [工作流与关闭证据](runs/{}/workflow_summary.json)。

复用框架/模型/控制源码前后一致，单初态工程验证不代替正式多初态benchmark或实机适配。运行代码版本与上轮完全相同，因此本次没有重复原有95项单元契约测试；实际闭环与视频检查已重新执行。
'''.format(workflow['seconds'], sum(e['steps'] for e in audit['episodes']),
           sum(e['VLA_requests'] for e in audit['episodes']), RID, RID, RID, RID)
with (ROOT / '视频索引_20261007_r2.md').open('x') as file:
    file.write(report)
index = '''# 当前版本与文件索引 20261007 r2

使用冻结代码 `versions/panda_libero_20261007_r1` 重新运行；旧代码和结果未覆盖。

| 内容 | 路径 | 状态 |
|---|---|---|
| 最新闭环视频与结果 | 视频索引_20261007_r2.md | 20 MP4逐帧解码通过 |
| 新任务轨迹 | runs/spatial10_video_20261007_r2/ | 10任务完成 |
| 视频播放文件 | videos/spatial10_video_20261007_r2/ | 与原始MP4哈希一致 |
| 新轨迹审计 | reports/spatial10_video_20261007_r2_audit.json | 10/10通过 |
| 新证据冻结 | releases/spatial10_video_20261007_r2/manifest.json | 不扩大验收范围 |
| 原实现与复现 | versions/panda_libero_20261007_r1/README.md | 原样复用 |
| 旧版本索引 | 当前版本与文件索引_20261007_r1.md | 保留 |

本轮固定Panda/PandaGripper，seed=0、Spatial 10×1，不代表多初态benchmark、完整RPent工具集或ECO65实机适配通过。
'''
with (ROOT / '当前版本与文件索引_20261007_r2.md').open('x') as file:
    file.write(index)
print('20 videos published with source-identical hashes; new index saved')
