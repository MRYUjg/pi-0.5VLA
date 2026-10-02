#!/usr/bin/env python3
"""Generate a scoped report from the immutable validation summary and audit."""
import json,pathlib,hashlib,datetime,collections
P=pathlib.Path('/mnt/data/vla-policy/openpi/agent-work-space/zhixing_gripper_solve')
s=json.loads((P/'evidence/real_objects_summary_20261002_r2.json').read_text());a=json.loads((P/'evidence/real_objects_validation_audit_20261002_r2/verification.json').read_text());batch=P/'evidence/real_objects_validation_20261002_r2'
rs=[json.loads(p.read_text()) for p in sorted(batch.glob('*/result.json'))]
def fmt(v):return f'{v:.3f}'
def span(case,key):
 v=s['cases'][case]['metrics_successes_only'][key];return '无成功样本' if v is None else f"{fmt(v['min'])}–{fmt(v['max'])}"
rows=[]
for c,name,task in [('can','罐体','LIBERO Object task0：alphabet_soup → basket'),('bowl','黑碗','LIBERO Spatial task0：指定 akita_black_bowl → plate')]:
 v=s['cases'][c];rows.append(f"| {name} | {task} | {v['success']}/{v['total']} | {'通过' if v['acceptance'] else '未通过'} |")
failure_lines=[]
for r in rs:
 if not r['success']:
  bad=[(ph,info) for ph,info in r['plan'].get('stage_arrivals',{}).items() if not info['ok']]
  phase,info=bad[-1] if bad else (r['failure']['stage'] if r['failure'] else 'gate',None)
  pos=f"{info['error'][0]*1000:.3f}" if info else '—';rot=f"{info['error'][1]:.5f}" if info else '—'
  failure_lines.append(f"| {r['case']} / {r['seed']} | {phase} | {r['failure']['reason'] if r['failure'] else '最终闸门未通过'} | {pos} | {rot} |")
failure_text=('本批无失败样本。' if not failure_lines else '本批失败保留，未作为成功或被其他种子替换：\n\n| 对象/seed | 实际失败阶段 | 原因 | 位置误差mm | 姿态误差rad |\n|---|---|---|---|---|\n'+'\n'.join(failure_lines))
metrics=[]
for label,key,unit in [('最大指物穿透','penetration_peak_mm','mm'),('最大实际TCP单步位移','max_step_disp_mm','mm'),('容器最大位移','recipient_shift_mm','mm'),('横移最低实际净空','min_horizontal_clearance_mm','mm'),('保持段最大指物相对滑移','hold_relative_slip_mm','mm'),('释放后窗口位移','post_drift_mm','mm'),('保持段双侧接触比例','bilateral_hold','')]:metrics.append(f'| {label} | {span("can",key)} | {span("bowl",key)} | {unit} |')
loaded={}
for c in ('can','bowl'):
 cr=[r for r in rs if r['case']==c and r['success']];loaded[c]={'minimum_carry_bilateral_fraction':min((r['phases'][ph]['bilateral_fraction_mean'] for r in cr for ph in ('clearance_lift','transport','place')),default=None),'selected_grasp_width_mm':[r['plan']['grasp_selected']['material_width_m']*1000 for r in cr],'carry_tilt_degrees':[r['plan']['tilt_deg'] for r in cr],'tcp_in_mount':[r['initial_geometry']['tcp_in_mount'] for r in cr]}
with (P/'evidence/real_objects_geometry_contact_details_20261002_r1.json').open('x') as f:json.dump(loaded,f,ensure_ascii=False,indent=2)
force=[]
for c,name in [('can','罐体'),('bowl','黑碗')]:
 v=s['cases'][c];f=v['force_mean_range_N'];force.append(f"- {name}：保持段左指平均法向力 {fmt(f[0][0])}–{fmt(f[1][0])} N，右指 {fmt(f[0][1])}–{fmt(f[1][1])} N；保持段外部支撑峰值 {v['max_unsupported_hold_contact_N']:.3g} N，平均接触合力与重力的不平衡范数最大 {v['max_hold_force_balance_norm_N']:.3g} N；搬运三段双侧接触比例最低 {loaded[c]['minimum_carry_bilateral_fraction']:.4f}。")
report='''# 真实碗罐几何接入：实现与验证记录 r1

日期：2026-10-02。专用目录：`zhixing_gripper_solve`。环境：纯 CPU、隔离 Python3.10 / MuJoCo3.2.6，ECO65 + ZhiXing，原生资产不变。

## 结论与范围

已把原生碗、罐的碰撞几何接入抓取规划，并重新验证完整抓放及接触行为。本次是两个指定原生任务上的脚本教师验证；不把它表述为 π0.5 模型成功率或 ECO65+ZhiXing 总体适配验收。

| 对象 | 任务 | 新种子成功数 | 本轮≥9/10目标 |
|---|---|---|---|
'''+ '\n'.join(rows)+f'''

{failure_text}

候选代码在复测前锁定，证据为 `maintenance/real_objects_20261002_r1/candidate_lock_r2.json`。两类均用 51100–51109，各10个独立进程；开发种子49100、首批验证50100–50109以及失败回归均不混入本表成功率，不补种子。两类初态哈希各有 {s['cases']['can']['unique_initial_states']} / {s['cases']['bowl']['unique_initial_states']} 个不同值，实际目标位置范围见汇总JSON。罐的目标初始位置实际上固定（第一批浮点差异仅舍入量级），主要变化来自其他场景状态；因此罐10/10不能称为10个不同抓取位置。按1微米量化的目标包围盒差异数量在汇总JSON中单列。碗初态位置/姿态有变化。原生reset范围有限，本结果不代表整个工作空间。

数值证据检查：**{'PASS' if a['pass'] else 'FAIL'}**，共 {s['total_physics_samples']:,} 个物理步样本；实际任务是否通过以表中结果为准。审计重算力、穿透、阶段到位和稳定性；原生谓词使用保存的引擎输出，本次没有独立重新执行全部轨迹。

## 后续开发与当前阻塞

r10的正式新种子为碗8/10，尚未达到9/10。后续开发没有把这个数字改写为通过：r11两个失败回归0/2；r12通过碗局部上沿取样和开口下降碰撞预检查，已在倾斜碗51106中形成持续双侧抓持、抬升并完成40步悬空保持，但仍未完成搬运到位，两个回归仍0/2。r13按加权雅可比最小奇异值挑选姿态，三个开发回归0/3，未作为当前推荐入口。

只读探针随后核实：当前有效profile是legacy，MountedECO65额外关节回拉PD的kp=3、kv≈3.464；控制器和教师TCP均为gripper0_grip_site。一个失败末态重建出的额外回拉力矩峰值约2.19 N·m。这一项在OSC后直接叠加，源码说明它将关节拉回init_qpos。它构成末端目标受到竞争的有依据假设，但未做A/B，不能认定为全部失败的唯一根因。

配置证据在 `evidence/real_objects_regression_20261002_r12/bowl_s51100_control_context.json`。完整[控制配置A/B方案](控制配置_AB验证方案_20261002_r1.md)和配对实验入口 `run_real_control_ab_r2.py` / `real_object_teacher_r15.py` 已准备，但尚未获授权执行。AGENTS.md第4条要求不得擅自改控制器参数，因此没有自行启用kp=0，也没有修改生产默认或上游冻结源。

当前推荐复现已完成重测的r10；r12用于复现倾斜碗的抓持改进，不能称为全链修复；r13不推荐；r14/r15只为未执行的A/B准备，不能混入结果。

## 实际几何如何进入规划

1. 从当前 MuJoCo 编译模型读取目标子树的实际碰撞几何、世界姿态和工具坐标，保留原模型。罐由21个、碗由40个碰撞盒构成；没有以诊断方块代换，也没有先把目标物旋转到夹爪闭合轴。
2. 对每个有姿态的碰撞盒做闭合射线求交，合并实体占据区间。罐采用整个截面的外夹；碗保留壳体间的空腔，选择一个碗沿实体区间。不能用碗的整体包围盒冒充实体可夹宽度。
3. 当前模型中罐外径约63 mm；碗外径约107 mm，大于当前夹爪约90 mm开口。碗改为单侧捏沿，实际候选材料宽约2.6 mm。工具TCP在安装体坐标系约 `[0, 0, 0.20254] m`，直接读取模型，不套用 Panda 的工具偏置。
4. 接近和下降候选先做独立 MjData 逆解；实际仿真仍只发送7维OSC动作，禁止把逆解写入真实episode关节位置。
5. 抓住后重新测量载荷、完整夹爪外形和目标容器包围盒；规划安全抬升、横移及下降。检查中间水平路径的逆解，抬升实际到位且实际净空达标后才横移，横移到位后才下降。姿态变化后重测载荷偏移，修正容器上方目标和释放高度。

![原生碰撞截面与抓取点](evidence/real_geometry_plots_20261002_r2/native_geometry.png)

图中的形状是实际碰撞盒在抓取高度的截面，虚线表示开口跨度；不把渲染外观当作碰撞实体。通用mesh碰撞体尚未接入，本规划器遇到非box目标碰撞体会拒绝，而不会猜测。

## 发现与定位的控制执行问题

- r1 的碗候选 IK 可行，但接近段实际位置误差约7.8 mm，未到达4 mm门限；r2的15°候选仍有约5.8 mm偏差。由此确认“有逆解”不足以说明原教师能实际到位。
- r4/r5增加教师位置残差补偿后，15°候选接近通过，但下降姿态仍未到位。r5受阻末态离线重建未发现机器人接触或关节限位，支持从教师闭环执行继续排查；这不是对全轨迹无碰撞的证明。
- r6近竖直抓法已实际捏住碗沿并提起，双侧接触持续；抬升段旋转指令持续限幅，位置/姿态门限未过。r7提高碗教师旋转指令范围后姿态收敛，但平移指令限幅仍使位置偏差约12.7 mm。
- r8提高碗教师平移范围后，抬升及40步悬空保持通过；切换25°搬运姿态时实际位置仍偏约33.8 mm，残差已离开旧补偿激活区间，流程停止在横移之前。
- r9扩大碗教师残差补偿的有界激活范围，并提高安全抬升/横移/下降的教师平移指令范围，开发样例完整通过；首批新种子50100–50109中罐10/10、碗5/10。5个碗失败仍在切换25°搬运姿态时，位置已收敛到0.02 mm内，但姿态误差0.047–0.085 rad未达标。
- r10仅提高碗搬运及释放阶段的教师旋转增量参数至0.12 rad，保留此前0.03 rad姿态到位门限；先回归上述5个失败，再锁定此版本，以51100–51109重新验证。首批失败始终保留，不用回归结果覆盖原来的5/10。

因此本轮暴露的是原生几何抓法与教师动作生成的适配问题，不能归为“ZhiXing无法产生夹持力”，也不能把所有误差唯一归因为机械碰撞或质量。全程未改底层控制器增益、动作尺度、关节限位、原生资产质量/摩擦/接触参数。

### 明确记录教师动作层变更

最终入口 `real_object_teacher_r10.py`。位置残差积分每步为0.05倍误差：罐每轴限12 mm、在位置误差<20 mm且姿态误差<0.03 rad时更新；碗每轴限25 mm、在位置误差<50 mm且姿态误差<0.1 rad时更新。碗教师平移增量参数：安全抬升/横移/下降30 mm，其余20 mm；旋转参数在搬运、释放及释放后阶段为0.12 rad，接近/抓取/抬升/保持为0.08 rad。罐保留前版限幅。

这些参数是教师动作生成限制，**不是放宽实际运动验收**。实际每控制步TCP位移仍要求<20 mm；碗下降抓取点的位置门限还收紧到1 mm。教师所有实际动作和补偿量均写入steps.jsonl。方案与门限见[协议r3](真实碗罐_实施与验收方案_20261002_r3.md)。

## 重新验证的接触行为

下表是成功样本中，各轨迹指标的最小–最大范围；接触力单位N，穿透按逐物理步峰值统计。

| 指标 | 罐体 | 黑碗 | 单位 |
|---|---|---|---|
'''+ '\n'.join(metrics)+'\n\n'+'\n'.join(force)+f'''

成功轨迹同时检查：初态任务未完成；物体抬升≥30 mm；40个控制步保持中高度≥25 mm且无外部支撑、双侧接触≥95%；穿透≤2 mm；各阶段实际到位；横移净空≥10 mm；容器未被推移超过5 mm；无超阈值（0.1 N）的机器人碰容器或载荷提前碰容器；释放后20步原生任务成功且接触目标支撑，漂移<5 mm；MuJoCo无数值警告、进程正常退出。附加审计也检查整条机械臂的非目标场景接触。

![罐体实际轨迹与接触](evidence/real_objects_plots_20261002_r2/can.png)

![黑碗实际轨迹与接触](evidence/real_objects_plots_20261002_r2/bowl.png)

## 证据、复现和限制

- 最终教师：`scripts/real_object_teacher_r10.py`；CPU入口：`scripts/run_real_object_r10.py`。
- 新种子全量证据：`evidence/real_objects_validation_20261002_r2/`，包括result、steps、physics.npz（含列名）、逐物理步recipient_contacts及environment.xml。
- 独立数值检查：`scripts/verify_real_object_r3.py`；结果：`evidence/real_objects_validation_audit_20261002_r2/verification.json`。
- [最终汇总](evidence/real_objects_summary_20261002_r2.json)、[几何与持续接触细节](evidence/real_objects_geometry_contact_details_20261002_r1.json)。
- 几何解析测试4/4：空腔保留、旋转盒截面、重叠合并、射线未命中。测试r8中的几何函数与最终r10相同；开发r1–r10的失败与日志全部保留。
- 上游 `n0_20260930_r1` 的2,113个受管文件及归档恢复仍为PASS；报告位于 `maintenance/real_objects_20261002_r1/upstream_verification.json`。

服务器仓库根目录执行，批次名必须为全新目录：

```bash
python3 agent-work-space/zhixing_gripper_solve/scripts/run_real_object_r10.py \\
  --batch 新的唯一批次名 --seeds 51110,51111 --workers 2
```

本轮保留原生仿真质量：罐约25.85 g、碗约5.59 g。这些质量、摩擦和碰撞表示不能直接等同于真实实物；整套仿真结果尚不代表真实硬件负载能力。物体仅覆盖当前两种资产和两个任务，尚未证明更多碗型、罐型、全部Spatial任务或实机。

没有切换正式N4生产采集器，没有生成RGB/LeRobot训练数据，也没有执行π0.5推理或微调。下一步先对实际控制冲突做经授权的隔离A/B，再重新完成碗任务≥9/10验证；当前不放行图像数据规模采集或训练。总体Spatial 10×20、≥90%且Panda对照的验收目标保持不变。
'''
with (P/'真实碗罐几何接入_实现与验证记录_20261002_r1.md').open('x') as f:f.write(report)
old=(P/'README.md').read_bytes();new='''# ZhiXing 夹爪专用优化目录

更新：2026-10-02。服务器纯CPU；上游冻结基线 `n0_20260930_r1` 未改变。

**当前状态：已实现真实碗/罐几何抓取规划；两个原生任务的新种子验证结果为罐 {can}/10、碗 {bowl}/10。此为脚本教师结果，π0.5与总体适配尚未验收。**

## 当前入口

1. [真实碗罐几何接入实现与验证记录](真实碗罐几何接入_实现与验证记录_20261002_r1.md)。
2. [任务协议r3](真实碗罐_实施与验收方案_20261002_r3.md)。
3. [最终汇总](evidence/real_objects_summary_20261002_r2.json)与[独立数值检查](evidence/real_objects_validation_audit_20261002_r2/verification.json)。
4. 教师 `scripts/real_object_teacher_r10.py`，CPU入口 `scripts/run_real_object_r10.py`，审计 `scripts/verify_real_object_r3.py`。
5. 候选锁定、源快照与完整性检查：`maintenance/real_objects_20261002_r1/`。

待解决：碗尚未达到≥9/10。r12已改善倾斜碗抓持，r13开发失败；已准备[隔离控制配置A/B方案](控制配置_AB验证方案_20261002_r1.md)，因AGENTS.md第4条尚待用户授权，未执行控制配置变更组。

## 前阶段与历史

- [安全教师试块验证](安全教师路径_实现与验证记录_20261002_r1.md)：试块5次回归+10个新增初态通过，教师safe_teacher_r8，入口run_safe_teacher_r9；这是试块历史，不与原生物体统计混算。
- [带载测量与分阶段定位](带载测量与分阶段定位_20261001_r1.md)。
- [测量口径与环境](测量口径与运行环境_20261001_r2.md)。
- r1–r9开发证据、首批r9验证（罐10/10、碗5/10）和r10失败回归全部保留，不与最终新种子统计混算。
- 清理台账：`maintenance/cleanup_20261001_r1/`。本轮未删除历史证据。

所有试验写入新目录，不覆盖已冻结版本。当前无RGB训练数据、无模型微调、无正式Spatial 200次验收或实机部署；生产采集器和上游闸门保持原状态。
'''.format(can=s['cases']['can']['success'],bowl=s['cases']['bowl']['success'])
(P/'README.md').write_text(new)
with (P/'maintenance/real_objects_20261002_r1/entry_update.json').open('x') as f:json.dump({'file':'README.md','previous_sha256':hashlib.sha256(old).hexdigest(),'current_sha256':hashlib.sha256(new.encode()).hexdigest(),'reason':'point to native-object geometry validation; preserve scoped baseline history'},f,indent=2)
print('REPORT WRITTEN')
