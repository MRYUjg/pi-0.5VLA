#!/usr/bin/env python3
"""Read-only numerical derivation from completed diagnostic artifacts."""
import argparse,json,math,pathlib
ap=argparse.ArgumentParser();ap.add_argument('batch');ap.add_argument('--out',required=True);a=ap.parse_args()
batch=pathlib.Path(a.batch);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=False)
status=json.loads((batch/'process_status.json').read_text());assert all(s['exit_code']==0 and s['result_exists'] for s in status)
chains=[];pulls=[];pairs={};checks=[]
for p in sorted(batch.glob('*/result.json')):
 d=json.loads(p.read_text());rs=[json.loads(x) for x in (p.parent/'steps.jsonl').read_text().splitlines()] if (p.parent/'steps.jsonl').exists() else []
 assert d['calibration']['pass'] and not any(d['physics_warning_counts'])
 base={'run':p.parent.name,'gripper':d['gripper'],'seed':d['seed'],'mujoco_version':d['mujoco_version']}
 if d['mode']=='pull':
  v=d['result'].get('pull',{});base.update({'force_N':d['force_N'],'valid_preload':v.get('valid_preload',False),'held':v.get('held_under_load'),
   'slip_mm':v.get('measurement',{}).get('object_in_grip_max_change_mm'),'bilateral_fraction':v.get('measurement',{}).get('bilateral_fraction_mean'),
   'force_world':v.get('force_world'),'reason':d['result'].get('reason')})
  if v.get('valid_preload'):
   assert v['force_world']==[d['force_N'],0.,0.] and v['measurement']['steps']==40
  pulls.append(base);continue
 result=d['result'];base['failed_gates']=result.get('failed_gates');base['planning_failure']=result.get('reason');base['legacy_success']=result.get('success',False);base['n4_style_success']=result.get('success_n4_style',False)
 base['phases']=d['phases'];base['phase_end']=result.get('phase_end');base['first_collision']=None
 if rs:
  j=d['joint_names'].index('basket_1_joint0');ad=d['joint_qposadr'][j];q0=rs[0]['before']['qpos'][ad:ad+7]
  base['basket_init_qpos']=q0;base['object_init']=rs[0]['before']['object_pos'];base['basket_shift_mm']={}
  for ph in d['phases']:
   pr=[r for r in rs if r['phase']==ph];q=pr[-1]['after']['qpos'][ad:ad+7]
   base['basket_shift_mm'][ph]=math.dist(q[:3],q0[:3])*1000
  hits=[r for r in rs if r['phase']=='transport' and r['other_contact_force_peak_N']>.1]
  if hits:
   r=hits[0];st=r['after'];rot=st['object_rotation'];bottom=st['object_pos'][2]-sum(abs(rot[2][i])*h for i,h in enumerate([.03,.015,.035]))
   base['first_collision']={'step':r['step'],'time':st['time'],'object_bottom_z':bottom,'object_pos':st['object_pos'],'grip_pos':st['grip_site'],
     'basket_shift_mm':math.dist(st['qpos'][ad:ad+3],q0[:3])*1000}
  base['hold_force_balance_norm_N']=math.sqrt(sum(v*v for v in d['phases']['hold']['object_force_balance_mean_N'])) if 'hold' in d['phases'] else None
  # No clipping or joint-limit claim inferred from task outcome: derive from recorded vectors.
  for ph in ['transport','place']:
   pr=[r for r in rs if r['phase']==ph]
   if not pr:continue
   aj=[j for j,idx in enumerate(d['joint_qposadr']) if idx in d['arm_qpos_indices'] and d['joint_limited'][j]]
   margins=[min(r['after']['qpos'][d['joint_qposadr'][j]]-d['joint_range'][j][0],d['joint_range'][j][1]-r['after']['qpos'][d['joint_qposadr'][j]]) for r in pr for j in aj]
   base.setdefault('arm_joint_margin_min_rad',{})[ph]=min(margins)
  pairs.setdefault(d['seed'],{})[d['gripper']]={'q0':q0,'object':base['object_init']}
 chains.append(base)
for seed,gs in pairs.items():
 if len(gs)==2:
  z,p=gs['ZhixingGripper'],gs['PandaGripper'];checks.append({'seed':seed,'basket_init_max_abs_diff':max(abs(x-y) for x,y in zip(z['q0'],p['q0'])),'object_init_center_diff_m':math.dist(z['object'],p['object'])})
summary={'processes':len(status),'normal_exit_count':sum(s['exit_code']==0 for s in status),'chain':chains,'pull':pulls,'paired_executed_initial_checks':checks,
         'scope':'Diagnostic only; no official P0/Spatial acceptance granted. Planner rejections retained in denominator.'}
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
lines=['# 带载诊断数值汇总','',f'独立进程正常退出 {len(status)}/{len(status)}。MuJoCo 3.2.6，诊断脚本 r4；本表不构成适配验收。','',
'| 夹爪 | seed | 规划 | 抓放旧判据 | N4含伺服判据 | 搬运末篮子位移mm | 释放末篮子位移mm |','|---|---:|---|---|---|---:|---:|']
for c in chains:
 shift=c.get('basket_shift_mm',{})
 lines.append('|%s|%d|%s|%s|%s|%s|%s|'%(c['gripper'],c['seed'],c['planning_failure'] or '进入执行',c['legacy_success'],c['n4_style_success'],round(shift.get('transport',0),3) if shift else '—',round(shift.get('release',0),3) if shift else '—'))
lines+=['','| 夹爪 | seed | 外力 +X N | 前置有效 | 2s最大相对滑移mm | 保持判据 |','|---|---:|---:|---|---:|---|']
for p in pulls:lines.append('|%s|%d|%g|%s|%s|%s|'%(p['gripper'],p['seed'],p['force_N'],p['valid_preload'],round(p['slip_mm'],3) if p['slip_mm'] is not None else '—',p['held']))
(out/'数值汇总.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'processes':len(status),'pairs':checks,'chain':[{k:v for k,v in c.items() if k not in ['phases','phase_end']} for c in chains],'pull':pulls},ensure_ascii=False,indent=2))
