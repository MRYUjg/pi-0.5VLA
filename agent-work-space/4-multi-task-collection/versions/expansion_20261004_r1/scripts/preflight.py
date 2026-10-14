"""All registered scenes: initialization validity, physical coverage, and image previews."""
import argparse,json,pathlib,os
import numpy as np
from PIL import Image,ImageDraw
from collect import T,render
import scenes

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=pathlib.Path(a.out);out.mkdir(exist_ok=False)
 os.environ['EXTRA_JOINT_PD_PROFILE']='legacy';os.environ['ZHIXING_GRIPPER_SPEED']='0.01'
 records=[]
 for scene in json.loads(scenes.MANIFEST.read_text())['scenes']:
  env,task=T.D.build('libero_object',0,seed=scene['seed']);env.reset();before=scenes.object_states(env)
  sim,m,d,gid=T.N.ids(env);initial=scenes.contact_check(env);max_pen=initial['non_support_penetration_mm'];step=sim.step
  def watch():
   nonlocal max_pen
   step();max_pen=max(max_pen,scenes.contact_check(env)['non_support_penetration_mm'])
  sim.step=watch;T.N4.quiet_hold(env,120,-1)
  after=scenes.object_states(env);delta=float(np.linalg.norm(np.array(after['alphabet_soup_1']['position'])[:2]-scene['target_xy_m']))
  errors=[]
  if delta>.005:errors.append('initial target drift >5mm')
  if max_pen>2:errors.append('non-support penetration >2mm')
  if np.any(d.warning.number) or not np.isfinite(d.qpos).all():errors.append('physics invalid')
  if env.check_success():errors.append('goal already true')
  record={'scene':scene,'reset_objects':before,'settled_objects':after,'non_support_penetration_mm':max_pen,'target_xy_drift_m':delta,'pass':not errors,'errors':errors,'initialization':env.coverage_record}
  images=render(env);im=Image.new('RGB',(512,280),'#222');im.paste(Image.fromarray(images['image']),(0,24));im.paste(Image.fromarray(images['wrist_image']),(256,24));ImageDraw.Draw(im).text((4,5),scene['id']+' '+str(scene['target_xy_m'])+' yaw '+str(scene['target_yaw_offset_deg']),fill='white');im.save(out/(scene['id']+'.jpg'))
  env.close();records.append(record);print(scene['id'],'valid',not errors,errors,flush=True)
 points=np.array([r['settled_objects']['alphabet_soup_1']['position'][:2] for r in records]);pair=np.linalg.norm(points[:,None]-points[None,:],axis=-1);pair+=np.eye(len(pair))*100
 result={'records':records,'pass':all(r['pass'] for r in records),'unique_target_centers_1mm':len({tuple(np.round(p,3)) for p in points}),'target_xy_span_mm':(np.ptp(points,axis=0)*1000).tolist(),'min_pair_distance_mm':float(pair.min()*1000),'scene_manifest_sha256':records[0]['initialization']['manifest_sha256'],'scope':'Initialization check only; test scenes never run through teacher or included in training.'}
 (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'}));return int(not result['pass'])
if __name__=='__main__':raise SystemExit(main())
