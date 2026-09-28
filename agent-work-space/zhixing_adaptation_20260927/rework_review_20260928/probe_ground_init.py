import sys,json,pathlib,hashlib
import numpy as np
sys.path.insert(0,'/app/agent-work-space/zhixing_adaptation_20260927/scripts')
import n4_collect as C
import n3_grasp_test as N
from libero.libero import benchmark,get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
O=C.ROOT/'rework_review_20260928'
t=benchmark.get_benchmark_dict()['libero_object']().get_task(0)
b=pathlib.Path(get_libero_path('bddl_files'))/t.problem_folder/t.bddl_file
rows=[]
for robot,gripper in [('Panda','PandaGripper'),('ECO65','PandaGripper'),('ECO65','ZhixingGripper')]:
 e=ControlEnv(bddl_file_name=b,robots=[robot],gripper_types=gripper,use_camera_obs=False,has_offscreen_renderer=False);e.seed(7);e.reset();r=e.env.robots[0]
 s,m,d,g=N.ids(e)
 rows.append({'robot':robot,'gripper':gripper,'robot_class':type(r.robot_model).__name__,'q_init':r.init_qpos.tolist(),'grip_pos':d.site_xpos[g].tolist(),'controller_type':r.controller_config['type'],'default_gripper_speed':r.gripper.speed})
 e.close()
out={'poses':rows,'eco_panda_vs_reference_mm':float(np.linalg.norm(np.array(rows[1]['grip_pos'])-rows[0]['grip_pos'])*1000)}
(O/'ground_init_probe.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2),flush=True)
