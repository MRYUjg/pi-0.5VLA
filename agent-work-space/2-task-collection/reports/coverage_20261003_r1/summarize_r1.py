"""Aggregate recorded evidence without changing acceptance rules."""
import hashlib,json,math,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[2]
B=ROOT/'runs/coverage_train_val_r1'
def read(p):return json.loads(p.read_text())
def yaw(q):
    w,x,y,z=q
    return math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))
def wrap(x):return (x+180)%360-180
m=read(B/'dataset_manifest.json');pre=read(ROOT/'runs/coverage_preflight_r1/result.json')
rows=[]
for row in m['records']:
    ep=B/('can_s'+str(row['seed']));a=read(ep/'audit.json');r=read(ep/'replay.json');s=a['stats'];scene=s['scene'];obj=s['settled_object_states']
    rows.append({'scene_id':scene['id'],'split':scene['split'],'seed':row['seed'],'accepted':a['accepted'],'teacher_success':a['teacher_numeric_audit']['success'],'frames':s['frames'],'h5_bytes':(ep/'trajectory.h5').stat().st_size,'target_xy_mm':[v*1000 for v in obj['alphabet_soup_1']['position'][:2]],'basket_xy_mm':[v*1000 for v in obj['basket_1']['position'][:2]],'target_yaw_offset_configured_deg':scene['target_yaw_offset_deg'],'target_world_yaw_settled_deg':yaw(obj['alphabet_soup_1']['quaternion_wxyz']),'preparation_penetration_mm':s['preparation_penetration_mm'],'capture_exit':read(ep.with_suffix('.status.json'))['exit_code'],'replay_exit':read(ep/'replay.status.json')['exit_code'],'replay':r})
points=[r['target_xy_mm'] for r in rows]
pre_rows=[]
for r in pre['records']:
    q=r['settled_objects']['alphabet_soup_1']['quaternion_wxyz'];native=r['initialization']['native_sampler_before_override']['alphabet_soup_1']['quat'];actual=wrap(yaw(q)-yaw(native))
    pre_rows.append({'scene_id':r['scene']['id'],'configured_yaw_offset_deg':r['scene']['target_yaw_offset_deg'],'settled_yaw_offset_deg':actual,'yaw_error_deg':wrap(actual-r['scene']['target_yaw_offset_deg'])})
splits={s:read(B/(s+'_manifest.json')) for s in ('train','validation')}
ids={s:{r['seed'] for r in v['episodes']} for s,v in splits.items()}
reserved={r['scene']['seed'] for r in pre['records'] if r['scene']['split']=='test'}
assert not ids['train']&ids['validation'] and not (ids['train']|ids['validation'])&reserved
summary={'attempts':m['attempts'],'task_success':sum(r['teacher_success'] for r in rows),'accepted':m['accepted'],'split_counts':{s:len(v) for s,v in ids.items()},'total_actions':sum(r['frames'] for r in rows),'total_observations':sum(r['frames']+1 for r in rows),'total_h5_bytes':sum(r['h5_bytes'] for r in rows),'unique_target_centers_1mm':len({tuple(round(v) for v in p) for p in points}),'target_xy_span_mm':[max(p[i] for p in points)-min(p[i] for p in points) for i in (0,1)],'reserved_test_episodes_collected':len([x for x in reserved if (B/('can_s'+str(x))).exists()]),'initialization_preflight_pass':pre['pass'],'preflight_unique_target_centers_1mm':pre['unique_target_centers_1mm'],'max_preflight_yaw_error_deg':max(abs(r['yaw_error_deg']) for r in pre_rows),'yaw_checks':pre_rows,'rows':rows,'scope':'controlled initialization collection; no policy evaluation or training'}
with (pathlib.Path(__file__).parent/'summary.json').open('x') as f:json.dump(summary,f,indent=2)
print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','yaw_checks')},indent=2))
