"""Three held-out validation initializations, native-goal stability diagnostic; no teacher actions."""
import argparse,base64,collections,json,os,pathlib,sys,urllib.request
import h5py,numpy as np
from PIL import Image,ImageDraw
ROOT=pathlib.Path(__file__).resolve().parents[2];REPO=ROOT.parents[1];C=REPO/'agent-work-space/2-task-collection'
sys.path.insert(0,str(C/'versions/coverage_20261003_r1/scripts'))
from collect import T,snapshot,render
import scenes
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--port',type=int,default=8871);a=p.parse_args();assert a.seed in (55200,55201,55202);a.out.mkdir(parents=True,exist_ok=False)
os.environ['EXTRA_JOINT_PD_PROFILE']='legacy';os.environ['EXTRA_JOINT_PD_KP']='';os.environ['ZHIXING_GRIPPER_SPEED']='0.01'
env,task=T.D.build('libero_object',0,seed=a.seed);T.EX.install_generic_pad_geoms(print);T.EX.install_generic_state_gripper(env,print);env.reset()
with h5py.File(C/f'runs/coverage_train_val_r1/can_s{a.seed}/trajectory.h5','r') as f:
 for action in f['preparation/actions']:env.step(action)
 initial_error=float(np.max(abs(snapshot(env)['qpos']-f['observations/qpos'][0])));assert initial_error<=1e-8
 initial_rgb=render(env);assert all(np.array_equal(initial_rgb[k],f['observations/'+k][0]) for k in initial_rgb)
os.environ['EXTRA_JOINT_PD_PROFILE']='eco65_no_extra_pd'
queue=collections.deque();rows=[];images=[];window=[];success=False;clipped=0;max_pen=0.;native_any=False
try:
 for i in range(650):
  state=snapshot(env)
  if i%100==0:images.append((i,render(env)))
  if not queue:
   payload={'state':state['state'].tolist(),'prompt':task.language,**{k:base64.b64encode(v.tobytes()).decode() for k,v in render(env).items()}}
   req=urllib.request.Request(f'http://127.0.0.1:{a.port}/infer',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
   raw=np.asarray(json.loads(urllib.request.urlopen(req,timeout=300).read())['actions']);assert raw.shape==(10,7) and np.isfinite(raw).all();queue.extend(raw[:5])
  raw=queue.popleft();action=np.clip(raw,-1,1);clipped+=int(np.any(action!=raw));env.env.done=False;env.step(action)
  after=snapshot(env);native=bool(after['native_success']);native_any|=native
  geometry=T.S.geometry(env);sim,m,d,gid=T.N.ids(env);obj=set(geometry['object_geoms']);basket=set(geometry['basket_geoms']);gripper=set(geometry['gripper_geoms'])
  support=any((int(c.geom1) in obj and int(c.geom2) in basket) or (int(c.geom2) in obj and int(c.geom1) in basket) for c in d.contact[:d.ncon])
  grip_contact=any((int(c.geom1) in obj and int(c.geom2) in gripper) or (int(c.geom2) in obj and int(c.geom1) in gripper) for c in d.contact[:d.ncon])
  contact=scenes.contact_check(env);max_pen=max(max_pen,contact['non_support_penetration_mm'])
  row={'step':i,'raw_action':raw.tolist(),'executed_action':action.tolist(),'native_success':native,'basket_contact':support,'gripper_object_contact':grip_contact,'non_support_penetration_mm':contact['non_support_penetration_mm'],'object_center':geometry['object'].mean(0).tolist()};rows.append(row)
  stable=native and support and not grip_contact and action[-1]<0
  window=(window+[row['object_center']])[-20:] if stable else []
  if len(window)==20 and np.linalg.norm(np.asarray(window[-1])-window[0])<.005:success=True;break
  if i%50==0:print('POLICY STEP',a.seed,i,'native',native,flush=True)
  if np.any(d.warning.number):break
 images.append((i+1,render(env)))
 result={'seed':a.seed,'steps':len(rows),'initial_qpos_error':initial_error,'initial_rgb_exact':True,'native_success_seen':native_any,'released_supported_native_stable20':success,'action_frames_clipped':clipped,'max_sampled_non_support_penetration_mm':max_pen,'physics_warnings':d.warning.number.tolist(),'scope':'preliminary policy closed-loop diagnostic; step-boundary contact checks, not full teacher r6 physical certification','replan_steps':5,'max_steps':650,'controller_profile':'eco65_no_extra_pd','scene':scenes.spec_for(a.seed)}
 (a.out/'result.json').write_text(json.dumps(result,indent=2));(a.out/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 sheet=Image.new('RGB',(256*len(images),540),'#222222');draw=ImageDraw.Draw(sheet)
 for col,(step,views) in enumerate(images):
  draw.text((256*col+5,5),f'{a.seed} step {step}',fill='white')
  for row,k in enumerate(('image','wrist_image')):sheet.paste(Image.fromarray(views[k]),(256*col,25+256*row))
 sheet.save(a.out/'preview.jpg');print('ROLLOUT RESULT',result,flush=True)
finally:env.close()
