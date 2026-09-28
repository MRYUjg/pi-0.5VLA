import sys,json,pathlib
import numpy as np
sys.path.insert(0,'/app/agent-work-space/zhixing_adaptation_20260927/scripts')
import n4_audit as A
import n4_collect as C
O=C.ROOT/'rework_review_20260928'
p=C.ROOT/'raw/success/n4v5_libero_object_t0_s4_0004.npz'
original=np.load(p,allow_pickle=False)
class Modified:
 files=original.files
 def __getitem__(self,k):
  value=original[k]
  if k=='actions':
   value=value.copy();value[100:,:6]=0;value[100:,6]=-1
  if k=='wrist_image':value=np.roll(value,1,axis=0)
  return value
A.load=lambda _:Modified()
e,_=C.build('libero_object',0)
r=A.check_episode(p,e,True,None)
r['negative_control']={'source':str(p),'mutations_in_memory_only':['actions[100:]=zero arm + open gripper','wrist_image shifted by 1 frame'],'actual_replay_final_success':bool(e.check_success()),'original_file_modified':False}
(O/'audit_negative_control.json').write_text(json.dumps(r,indent=2,ensure_ascii=False));print(json.dumps({'pass':r['pass'],'checks':r['checks'],'negative_control':r['negative_control'],'replay_max_err_by_kind':r['replay_max_err_by_kind']},indent=2),flush=True);e.close()
