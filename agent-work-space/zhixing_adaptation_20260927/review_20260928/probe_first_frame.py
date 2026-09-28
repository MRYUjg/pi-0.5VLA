import os,sys,json,pathlib
import numpy as np
import imageio
os.environ['ZHIXING_GRIPPER_SPEED']='0.01'
sys.path.insert(0,'/app/agent-work-space/zhixing_adaptation_20260927/scripts')
import n4_collect as C
import n3_grasp_test as N
O=C.ROOT/'review_20260928'
e,t=C.build('libero_object',0,seed=2);ep=C.Episode(e);e.seed(2);ep.start()
s,m,d,g=N.ids(e);on,rn=C.goal_objects(e);bid,bname=C.find_body(s,on)
sample=C.sample_object_init(e,bid,on,bname,rn,np.random.default_rng(100002))
C.quiet_hold(e,45,-1.)
fresh=e.env._get_observations(force_update=True)
res={'sample':sample,'observation_age_control_steps':45,'camera_differences':{}}
for key,short in [('agentview_image','agentview'),('robot0_eye_in_hand_image','wrist')]:
 a=ep.obs[key];b=fresh[key]
 res['camera_differences'][short]={'mean_abs_pixel_diff':float(np.abs(a.astype(float)-b.astype(float)).mean()),'changed_pixels_fraction':float(np.any(a!=b,axis=-1).mean())}
 imageio.imwrite(str(O/('first_frame_'+short+'_stale_vs_actual.png')),np.concatenate([a[::-1,::-1],b[::-1,::-1]],axis=1))
res['cached_eef_pos']=ep.obs['robot0_eef_pos'].tolist();res['actual_eef_pos']=fresh['robot0_eef_pos'].tolist()
res['cached_gripper_qpos']=ep.obs['robot0_gripper_qpos'].tolist();res['actual_gripper_qpos']=fresh['robot0_gripper_qpos'].tolist()
(O/'first_frame_probe.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2));e.close()
