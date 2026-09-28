"""Produce traceable Markdown, table CSV and plots from completed measurements."""
import csv
import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / 'results'
data = json.loads((OUT / 'summary.json').read_text())
assert len(data) == 64, 'Full 2 robots x 2 PD modes x 16 cases must complete'
table = []
traces = {}
for r in data:
    stem = '{}_pd{}_{}'.format(r['robot'], 'on' if r['extra_pd'] else 'off', r['case'])
    with (OUT / (stem + '.csv')).open() as f:
        rows = [{k: float(v) for k,v in x.items()} for x in csv.DictReader(f)]
    traces[stem] = rows
    row = {k: r[k] for k in ('robot','extra_pd','case','tail_position_error_mm','tail_rotation_error_deg','screen_pass')}
    row.update({'error_at_50ms_mm': rows[0]['position_error_mm'],
                'error_at_250ms_mm': rows[4]['position_error_mm'],
                'peak_joint_saturation_fraction': max(r['torque_saturation_fraction_per_joint']),
                'robot_contact_pairs': len(r['robot_contact_pair_counts']),
                'warmup_drift_mm': r['warmup_drift_mm']})
    table.append(row)
with (OUT/'metrics.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)

lines=['# ECO65 + Panda 夹爪：控制器验证结果', '',
       '本轮为无模型、无渲染的 MuJoCo 仿真验证。固定 LIBERO spatial task0、seed=7、PandaGripper、20 Hz；'
       'Panda 为参考对照。未连接或控制实体机器人。', '',
       '## 实验与可信性', '',
       '- 四组：ECO65/Panda × 额外关节 PD 开/关，每组 16 个测试，共 64 个；另有 4 个冒烟测试。',
       '- 固定目标：静止保持，以及世界坐标 x/y/z 的 ±2 cm 平移和 ±0.1 rad 旋转；每项 3 秒。'
       '目标为等待 10 步后的控制器 grip_site 位姿加偏移，并持续保持。',
       '- 原生接口：x/y/z 各执行 +0.1 指令 5 步、-0.1 指令 5 步，再执行零动作至 3 秒；'
       '保留原生每步相对当前末端设置目标的语义。',
       '- PD 关闭通过测试进程内将原控制函数的 joint_kp=3.0 改为 0.0 实现；原 OSC 零空间控制不变。源码文件不变。',
       '- 每对 PD 开关实验的 reset 全部 qpos 哈希相同；每个控制子步均断言测量的限幅后力矩与执行器 ctrl 一致。',
       '- 位姿从控制器使用的 grip_site 读取；相机不参与本实验。每个固定目标在每个策略步均核查未漂移。',
       '- 筛查阈值：最后 0.5 秒平均位置误差 <2 mm 且姿态误差 <2°。这是本次工程筛查标准，非产品指标或任务成功率。', '',
       '## 固定目标结果（每组 13 项）', '',
       '| 机械臂 | 额外 PD | 通过数 | 最差尾段位置误差 mm | 最差尾段姿态误差 ° | 最大单关节饱和占比 |',
       '|---|---|---:|---:|---:|---:|']
for robot in ('ECO65','Panda'):
    for pd in (True,False):
        rr=[r for r in data if r['robot']==robot and r['extra_pd']==pd and not r['case'].startswith('native')]
        lines.append('| {} | {} | {}/13 | {:.3f} | {:.3f} | {:.2%} |'.format(robot,'开' if pd else '关',sum(r['screen_pass'] for r in rr),max(r['tail_position_error_mm'] for r in rr),max(r['tail_rotation_error_deg'] for r in rr),max(max(r['torque_saturation_fraction_per_joint']) for r in rr)))
lines += ['', '## 全部测试明细', '', '|机械臂|PD|测试|50ms位置误差 mm|250ms位置误差 mm|尾段位置误差 mm|尾段姿态误差 °|筛查|', '|---|---|---|---:|---:|---:|---:|---|']
for r in table:
    lines.append('|{}|{}|{}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{}|'.format(r['robot'],'开' if r['extra_pd'] else '关',r['case'],r['error_at_50ms_mm'],r['error_at_250ms_mm'],r['tail_position_error_mm'],r['tail_rotation_error_deg'],'不适用' if r['screen_pass'] is None else ('通过' if r['screen_pass'] else '未通过')))
lines += ['', '## 适用范围', '',
          '- 原生零动作会重新设置位置目标，尾段误差小不能证明之前的运动量全部执行到位；需结合轨迹图判断。',
          '- 测试仅覆盖单一场景、初始姿态附近的小范围运动，不能证明完整抓取路径可达，亦不能代替负载、接触或真实硬件验证。',
          '- 同一机器人 PD 开关的 reset 一致，但等待 10 步后的锚点允许受 PD 影响；warmup_drift_mm 与 start_eef_pos 已保存。',
          '- 雅可比最小奇异值混合米和弧度单位，仅作诊断线索，不能跨机器人直接作为统一奇异阈值。',
          '- 原始关节力矩饱和、关节限位利用率、接触对计数见逐项 JSON；每 50ms 位姿见逐项 CSV。', '',
          '![Fixed target tracking](results/fixed_tracking.png)', '', '![Native incremental response](results/native_response.png)']
(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')

fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
for j,ax in enumerate('xyz'):
    for robot in ('ECO65','Panda'):
        for pd in ('on','off'):
            label=robot+' PD '+pd
            rows=traces[robot+'_pd'+pd+'_pos_'+ax+'_plus']
            axes[0,j].plot([r['t_s'] for r in rows],[r['position_error_mm'] for r in rows],label=label)
            rows=traces[robot+'_pd'+pd+'_rot_'+ax+'_plus']
            axes[1,j].plot([r['t_s'] for r in rows],[r['rotation_error_deg'] for r in rows],label=label)
    axes[0,j].set_title('+20 mm world '+ax);axes[1,j].set_title('+0.1 rad world '+ax)
    axes[0,j].set_ylabel('Position error (mm)');axes[1,j].set_ylabel('Rotation error (deg)');axes[1,j].set_xlabel('Simulation time (s)')
    for i in (0,1): axes[i,j].grid(alpha=.25)
axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'fixed_tracking.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(13,4))
for j,ax in enumerate('xyz'):
    for robot in ('ECO65','Panda'):
        for pd in ('on','off'):
            stem=robot+'_pd'+pd+'_native_'+ax;rows=traces[stem]
            r=next(r for r in data if r['robot']==robot and r['extra_pd']==(pd=='on') and r['case']=='native_'+ax)
            p0=r['start_eef_pos'][j]
            axes[j].plot([x['t_s'] for x in rows],[(x['eef_'+ax]-p0)*1000 for x in rows],label=robot+' PD '+pd)
    axes[j].axvline(.25,color='gray',ls=':');axes[j].axvline(.5,color='gray',ls=':')
    axes[j].set_title('Native incremental '+ax);axes[j].set_xlabel('Simulation time (s)');axes[j].set_ylabel('Displacement from start (mm)');axes[j].grid(alpha=.25)
axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'native_response.png',dpi=160);plt.close(fig)
print('REPORT_COMPLETE', flush=True)

cp=ROOT/'contact_ablation'
if (cp/'summary.json').exists():
    ab=json.loads((cp/'summary.json').read_text())
    assert len(ab)==16
    ls=['# 基座—第一连杆碰撞对消融', '',
        '在 64 项基础验证发现持续自接触后，新增 16 项针对性验证。'
        '仅在测试环境 XML 中添加 robot0_base 与 robot0_link_1 的 contact/exclude；'
        '其他碰撞保留，仓库 robot.xml 未修改。所有 reset qpos 哈希一致。', '',
        '|PD|测试|原始位置误差 mm|排除碰撞后 mm|原始姿态误差 °|排除后 °|',
        '|---|---|---:|---:|---:|---:|']
    for pd in (True,False):
        for case in ('hold','pos_y_plus','pos_y_minus','rot_x_minus'):
            pair=[next(r for r in ab if r['extra_pd']==pd and r['case']==case and r['exclude_base_link1']==ex) for ex in (False,True)]
            a,b=pair
            ls.append('|{}|{}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|'.format('开' if pd else '关',case,a['tail_position_error_mm'],b['tail_position_error_mm'],a['tail_rotation_error_deg'],b['tail_rotation_error_deg']))
    ls += ['', '原始接触的最深穿透约 {:.3f} mm。接触力读取在首轮伴随原生内存错误，'
           '正式验证移除了该读取路径，不使用首轮接触力数值作结论。'.format(
               -min(v for r in ab for v in r['min_contact_distance_m'].values())*1000), '',
           '碰撞排除后上述 8 项全部通过尾段 2 mm / 2° 筛查。'
           '这支持该碰撞约束是本轮方向性跟踪异常的主要原因；尚未验证全部工作空间和策略成功率。', '',
           '![Contact ablation](contact_ablation/contact_ablation.png)']
    (ROOT/'CONTACT_ABLATION.md').write_text('\n'.join(ls)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for j,case in enumerate(('pos_y_plus','rot_x_minus')):
        for ex in (False,True):
            for pd in ('on','off'):
                p=cp/('exclude_pair' if ex else 'original_contact')/('ECO65_pd'+pd+'_'+case+'.csv')
                with p.open() as f: rr=list(csv.DictReader(f))
                axs[j].plot([float(r['t_s']) for r in rr],[float(r['position_error_mm']) for r in rr],label=('Excluded' if ex else 'Original')+' PD '+pd)
        axs[j].set_title(case);axs[j].set_ylabel('Position error (mm)');axs[j].set_xlabel('Simulation time (s)');axs[j].grid(alpha=.25)
    axs[0].legend(fontsize=8);fig.tight_layout();fig.savefig(cp/'contact_ablation.png',dpi=160);plt.close(fig)
    print('CONTACT_REPORT_COMPLETE',flush=True)
