"""Derive target motion and sampled contacts; never feed privileged data to a policy."""
from pathlib import Path
import hashlib
import json
import math
import re

ROOT = Path('/mnt/data/vla-policy/openpi/gpt-rpent-control')
RUN = ROOT / 'runs/eco65_zhixing_spatial10_20261007_r1'
REPORT = ROOT / 'reports/eco65_zhixing_20261007_r1'
BDDL = ROOT.parent / 'third_party/libero/libero/libero/bddl_files/libero_spatial'
by_sha = {hashlib.sha256(p.read_bytes()).hexdigest(): p for p in BDDL.glob('*.bddl')}
rows = []
for task in range(10):
    episode = RUN / f'task{task:02d}_seed0'
    reg = json.loads((episode / 'env/registration.json').read_text())
    result = json.loads((episode / 'result.json').read_text())
    records = [json.loads(line) for line in (episode / 'env/transitions.jsonl').read_text().splitlines()]
    spec = by_sha[reg['bddl_sha256']]
    goal = spec.read_text().split('(:goal', 1)[1]
    target = re.search(r'\(On\s+(akita_black_bowl_\d+)\s+plate_1\)', goal).group(1)
    key = target + '_joint0'
    heights = [r['diagnostics']['objects'][key][2] for r in records]
    peak = max(range(len(records)), key=lambda i: heights[i])
    record = records[peak]
    pos = record['diagnostics']['objects'][key][:3]
    tcp = record['diagnostics']['grip_site_pos']
    contacts = [c for c in record['diagnostics']['contacts']
                if any((c[g] or '').startswith(target + '_') for g in ['geom1', 'geom2'])]
    rows.append(dict(task=task, target=target, BDDL=str(spec), native_success=result['native_success'],
                     reference='first post-action sample, not a stable-grasp criterion',
                     first_target_z=heights[0], max_target_z=heights[peak],
                     rise_from_first_sample_mm=1000 * (heights[peak] - heights[0]),
                     peak_step=peak + 1, peak_target_xyz=pos, peak_tcp_xyz=tcp,
                     peak_target_TCP_distance_mm=1000 * math.dist(pos, tcp),
                     target_contact_records_at_peak=contacts,
                     final_target_xyz=records[-1]['diagnostics']['objects'][key][:3]))
with (REPORT / 'target_motion_and_contacts.json').open('x') as file:
    json.dump(rows, file, ensure_ascii=False, indent=2)
text = ['# ECO65＋ZhiXing 正式闭环行为定位 r1', '',
        '以下从已完成正式轨迹与实际BDDL目标提取，仅用于事后诊断，不提供给规划器或VLA。物体最高点、TCP距离和20Hz接触采样不等于逐物理步稳定夹持验收。', '',
        '| task | 原生成功 | 相对首个动作后采样的最大升高(mm) | 最高点步数 | 最高点目标到TCP距离(mm) | 最高点目标接触记录数 |',
        '|---:|---|---:|---:|---:|---:|']
for r in rows:
    text.append(f"| {r['task']} | {r['native_success']} | {r['rise_from_first_sample_mm']:.3f} | {r['peak_step']} | {r['peak_target_TCP_distance_mm']:.1f} | {len(r['target_contact_records_at_peak'])} |")
text += ['',
         '任务0目标几乎未升高，模型仍执行了向盘区域移动的后续轨迹；其原生任务失败。任务1目标曾升高约98mm，但最高点没有目标接触记录，且距TCP约0.30m；该升高不能证明稳定抓取，最终也没有通过盘上成功谓词。完整视频可用于核对其他阶段。', '',
         '已观察到新夹爪本体在腕相机中占据较大区域。新几何、腕部外观与遮挡、6自由度控制动力学、默认legacy额外PD都可能影响结果。当前不是逐项消融，不能把失败唯一归因于其中一项，也不能由一次失败直接推断模型架构不支持新机械臂。', '',
         '[原始数值](target_motion_and_contacts.json)、[完整任务与视频](../../ECO65_ZhiXing闭环验证与视频_20261007_r1.md)。']
with (REPORT / '行为定位_20261007_r1.md').open('x') as file:
    file.write('\n'.join(text) + '\n')
print(json.dumps([dict(task=r['task'], rise_mm=round(r['rise_from_first_sample_mm'], 3),
                       native_success=r['native_success']) for r in rows]))
