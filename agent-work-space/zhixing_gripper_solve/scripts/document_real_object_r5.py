"""Write the final native-geometry report from audited evidence, exclusively."""
import hashlib, json, pathlib, datetime
P = pathlib.Path('/mnt/data/vla-policy/openpi/agent-work-space/zhixing_gripper_solve')
E = P / 'evidence'
summary = json.loads((E / 'real_objects_summary_20261002_r3.json').read_text())
audit = json.loads((E / 'real_objects_validation_audit_20261002_r3/verification.json').read_text())
assert summary['evidence_pass'] and audit['pass']
can, bowl = (summary['cases'][x] for x in ('can', 'bowl'))
def ratio(x):
    return f"{x['success']}/{x['total']}"
def maxmetric(x, k):
    v = x['metrics_successes_only'][k]
    return '无成功样本' if v is None else f"{v['max']:.3f}"
def minmetric(x, k):
    v = x['metrics_successes_only'][k]
    return '无成功样本' if v is None else f"{v['min']:.3f}"
history = []
for name in ('real_objects_validation_20261002_r1', 'real_objects_validation_20261002_r2',
             'real_control_ab_bowl_20261002_r1', 'real_control_ab_can_20261002_r1',
             'real_objects_regression_20261002_r16', 'real_objects_regression_20261002_r17',
             'real_objects_regression_20261002_r18', 'real_objects_regression_20261002_r19',
             'real_objects_regression_20261002_r20', 'real_objects_can_regression_20261002_r20',
             'real_objects_validation_20261002_r3'):
    groups = {}
    for f in sorted((E / name).glob('*/result.json')):
        r = json.loads(f.read_text())
        key = (r['version'], r['case'], r.get('runtime_profile', {}).get('extra_joint_pd_profile', 'legacy'))
        groups.setdefault(key, []).append(r)
    for (teacher, case, profile), rs in groups.items():
        history.append(f"| {name} | {teacher} | {case} / {profile} | {sum(r['success'] for r in rs)}/{len(rs)} |")
failures = [f"- {r['case']} / seed {r['seed']}：{r['failure']['stage'] if r['failure'] else 'gate_only'}；{r['failure']['reason'] if r['failure'] else str(r['gates'])}" for r in audit['records'] if not r['success']]
verdict = '通过本阶段两任务门槛' if summary['task_scope_pass'] else '尚未通过本阶段两任务门槛'
report = f'''# 实际碗/罐几何接入：实现与验证记录 r1

日期：2026-10-02；工作目录：`zhixing_gripper_solve`；服务器纯CPU。本记录更新此前试块阶段和中间开发记录。

## 当前结论

统一候选 `real_object_teacher_r20` 在未使用种子52100–52109的独立测试中：**罐 {ratio(can)}，碗 {ratio(bowl)}；{verdict}**。本阶段门槛是每类至少9/10并且逐物理步证据审计通过。

这是ECO65+ZhiXing在两个原生LIBERO任务中的脚本教师验证。不是π0.5推理或微调结果，不是LIBERO Spatial十任务×20次最终验收，也不能直接代表实机成功率或额定负载能力。

## 接入了什么几何

- 罐使用LIBERO Object task0的原生alphabet_soup碰撞模型，21个碰撞盒，约62.9×62.5×76.2 mm，直接规划整体夹持。
- 碗使用LIBERO Spatial task0的原生akita_black_bowl，40个碰撞盒描述碗壁和底部。外径约107 mm，大于ZhiXing当前开口约89.6 mm，因此规划夹持局部碗沿，不能把整个包围盒当成实心抓取截面。
- 射线与各碰撞盒在其局部坐标内精确求交，再合并占据区间，保留碗腔空隙；倾斜碗增加基于物体坐标系上缘表面的候选和张开夹爪接近路径碰撞检查。
- TCP使用实际`gripper0_grip_site`，相对安装体平移约[0,0,0.20254] m。搬运安全高度同时考虑实际夹爪完整外形和带载物体包围盒，并在姿态变化后重新测量偏移。
- 执行顺序保持“垂直抬升到位→横向搬运到位→下降投放”。逆解在独立MjData中做预检，不向正在执行的仿真写入关节姿态，不替换物体、不强制摆正倾斜碗。

## 为什么旧教师仍然失败

第一轮真实物体新种子测试为罐10/10、碗5/10；改进搬运姿态跟踪后的第二轮为罐10/10、碗8/10。失败暴露了三个不同层面的问题：倾斜碗不能仅按世界水平截面找抓点；额外关节回拉PD与末端任务跟踪发生冲突；教师补偿和瞬时到位切换导致末段过冲或振荡。

用户授权后，r15配对A/B保持教师、初始qpos、资产、动作接口和门限相同，只把ECO65额外关节PD从kp=3改为已有kp=0配置。三个碗场景A组0/3、B组1/3，B组全部通过旧阻塞的抬升/搬运阶段，但其中两例在下降/松爪发生接触。罐A/B各1/1。它证实了该PD项对这些旧阻塞的影响，不能解释或解决全部失败。

额外PD源码将关节拉向init_qpos，力矩直接叠加在OSC后，没有任务零空间投影。关闭这项也会去掉其对应阻尼。生产默认配置与源码均未修改；本轮仅在授权的隔离进程中使用`eco65_no_extra_pd`。

逐步轨迹进一步显示：r15碗51100松爪后TCP从约0.972 m下降到0.909 m，目标约0.975 m，随后指垫碰桌。r17的另一失败则接近静止、无非法接触，肘部接近伸直，受静摩擦约束；r18倾斜碗有小幅姿态振荡。这些失败应分开定位，不能都归为夹持力不足。

## 最终教师修正

碗使用实际碗壁候选和倾斜碗回退规划，关闭kp0条件下的教师积分补偿，将旋转误差换算从除以0.05改为除以0.10，减小姿态增量放大；阶段切换要求连续5步满足原误差门限且每步实际TCP移动小于0.5 mm。罐保留r15已通过的教师行为，避免把碗的修改直接套用到罐。

OSC增益、动作接口尺度、力矩限值、夹爪速度、资产质量/摩擦、接触参数和原成功门限保持不变。指令幅度不是实际位移；实际TCP单步位移仍受20 mm门限审查。该最终组合同时包含PD配置和教师修改，其成功率不能全部归因于单独取消PD。

预注册方案为[实施与验收方案r7](真实碗罐_实施与验收方案_20261002_r7.md)，源码锁为`maintenance/real_objects_20261002_r1/candidate_lock_r3.json`。r20先完成3个碗和1个罐开发回归，再测试新种子。所有开发回归与正式测试分开，失败不替换，不补抽种子。

## 新种子结果与接触行为

| 指标 | 罐 | 碗 |
|---|---:|---:|
| 全流程通过 | {ratio(can)} | {ratio(bowl)} |
| 成功样本保持阶段最低双侧接触比例 | {minmetric(can,'bilateral_hold')} | {minmetric(bowl,'bilateral_hold')} |
| 成功样本最大穿透，mm | {maxmetric(can,'penetration_peak_mm')} | {maxmetric(bowl,'penetration_peak_mm')} |
| 成功样本最大实际TCP单步位移，mm | {maxmetric(can,'max_step_disp_mm')} | {maxmetric(bowl,'max_step_disp_mm')} |
| 成功样本横向搬运最低净空，mm | {minmetric(can,'min_horizontal_clearance_mm')} | {minmetric(bowl,'min_horizontal_clearance_mm')} |
| 成功样本最大接收物位移，mm | {maxmetric(can,'recipient_shift_mm')} | {maxmetric(bowl,'recipient_shift_mm')} |
| 成功样本最大保持相对滑移，mm | {maxmetric(can,'hold_relative_slip_mm')} | {maxmetric(bowl,'hold_relative_slip_mm')} |
| 教师执行监测区间非法接触采样数 | {can['unsafe_contact_samples']} | {bowl['unsafe_contact_samples']} |
| 教师执行监测区间机器人非目标接触数，>0.1 N | {can['robot_nontarget_contacts']} | {bowl['robot_nontarget_contacts']} |

共审计{summary['total_physics_samples']}个物理步采样。成功样本保持阶段必须悬空且无其他支撑；投放后的目标—盘/篮支撑接触是预期接触，不能误记为碰撞失败。失败样本不从已记录区间接触计数中排除；reset和120步静置原本不在该统计区间，不能将0次解读为整集无碰撞；表中注明“成功样本”的范围不代表失败样本也满足门限。

独立测试失败清单：

{chr(10).join(failures) if failures else '无。20次结果全部保留。'}

## 两个剩余失败已定位到不同阶段

52103在教师动作前的reset/静置阶段就发生碗—盘初始穿插，碗被弹出桌面后落地，导致规划不可达。只读逐物理步复现与原测试初始qpos哈希完全一致。52104仍在桌上，但接近抓点的位置误差10.80 mm超过4 mm门限。详见[独立验证失败定位](真实碗罐_独立验证失败定位_20261002_r1.md)。两例均保留为失败，阶段验收未通过，不放行规模采集。

## 统计与物理边界

罐原生任务的目标初始位置基本固定，本批以1 μm精度统计不同目标包围盒数为{can['unique_target_bboxes_at_1um']}，碗为{bowl['unique_target_bboxes_at_1um']}。罐的种子会改变场景，但不能声称验证了10个不同罐抓取位置。

原生仿真质量：罐{can['mass_kg']} kg，碗{bowl['mass_kg']} kg；这不是实测硬件物体质量。特别是碗约5.59 g，因此本轮不能证明真实重碗的负载能力。

审计从逐物理步数据重算力、双侧接触、穿透、支撑、位移、到位及稳定窗口，并核对源码和进程退出状态。原生任务谓词来自记录的环境输出，未声称进行全部物理引擎的独立重放。冻结检查与数值审计通过不等同总体机械臂适配验收。

## 批次索引

| 批次 | 教师 | 对象/配置 | 完整通过 |
|---|---|---|---:|
{chr(10).join(history)}

更早的失败开发版本同样保存在evidence目录，见冻结证据清单；未删除失败样本。历史文件中“尚未执行”反映其写作时状态，当前状态以本记录和原始结果为准。

## 可复核入口与下一步

- [本批汇总](evidence/real_objects_summary_20261002_r3.json)、[逐物理步审计](evidence/real_objects_validation_audit_20261002_r3/verification.json)、[原始批次](evidence/real_objects_validation_20261002_r3/)。
- [A/B执行与定位](控制配置_AB执行与定位记录_20261002_r1.md)、[教师开发定位补充](真实碗罐_开发定位补充_20261002_r1.md)。
- [实际轨迹/接触图：罐](evidence/real_objects_plots_20261002_r3/can.png)、[碗](evidence/real_objects_plots_20261002_r3/bowl.png)；[原生碰撞截面图](evidence/real_geometry_plots_20261002_r3/native_geometry.png)。
- 源码及证据冻结：`maintenance/real_objects_20261002_r1/`。上游n0_20260930_r1已进行2113个受管文件与归档核验，没有修改上游冻结文件。

后续先修复初始化有效性与接近位置残差，再扩展任务与初态覆盖，随后接入带成功筛选与动作/状态一致性检查的RGB任务数据采集。真实负载质量、更多物体和碰撞形态仍需单独验证。CPU服务器可继续仿真和数值检查；本轮没有启动GPU训练或部署π0.5。
'''
report_path = P / '真实碗罐几何接入_实现与验证记录_20261002_r1.md'
with report_path.open('x') as f:
    f.write(report)
readme = P / 'README.md'
old = readme.read_bytes()
ledger = P / 'maintenance/real_objects_20261002_r1'
with (ledger / 'README_before_native.md').open('xb') as f:
    f.write(old)
entry = f'''# ZhiXing夹爪专用优化目录

更新：2026-10-02；当前服务器纯CPU；上游冻结基线保持不变。

当前：**实际LIBERO罐{ratio(can)}、碗{ratio(bowl)}，{verdict}**。这是两个任务的脚本教师验证，整体机械臂适配、π0.5微调和Spatial十任务×20次正式验收仍未完成。

1. [当前实现与验证记录](真实碗罐几何接入_实现与验证记录_20261002_r1.md)。
2. [预注册方案r7](真实碗罐_实施与验收方案_20261002_r7.md)、[独立测试汇总](evidence/real_objects_summary_20261002_r3.json)。
3. [控制配置A/B执行记录](控制配置_AB执行与定位记录_20261002_r1.md)、[教师开发定位](真实碗罐_开发定位补充_20261002_r1.md)。
4. [此前诊断试块教师记录](安全教师路径_实现与验证记录_20261002_r1.md)。

当前候选：`scripts/real_object_teacher_r20.py`；启动器：`scripts/run_real_control_ab_r7.py`；检查：`scripts/verify_real_object_r6.py`。使用已有`eco65_no_extra_pd`隔离运行时配置；生产默认配置未切换。所有早期版本和失败批次仅作历史证据，不与当前独立测试合并统计。

服务器仓库根目录复现时须使用新批次名并显式指定profile：

```bash
python3 agent-work-space/zhixing_gripper_solve/scripts/run_real_control_ab_r7.py \\
  --batch 新的唯一批次名 --cases can,bowl --seeds 52100 \\
  --profiles eco65_no_extra_pd --workers 2
```

该入口保存数值证据，不生成训练数据。受管文件必须建立新版本；冻结完整性记录见`maintenance/real_objects_20261002_r1/`。原始证据已包括失败样本；不覆盖旧结果。
'''
readme.write_text(entry)
with (ledger / 'entrypoint_update.json').open('x') as f:
    json.dump({'file': 'README.md', 'previous_sha256': hashlib.sha256(old).hexdigest(),
               'new_sha256': hashlib.sha256(readme.read_bytes()).hexdigest(),
               'backup': 'README_before_native.md', 'report': report_path.name}, f, ensure_ascii=False, indent=2)
print(report_path)
