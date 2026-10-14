"""Real pinned API integration: bounded writing, writer reload, reader/chunk correctness."""
import argparse,gc,json,pathlib,resource
import h5py,numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from resume_writer import writer

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
 features={'image':{'dtype':'image','shape':(256,256,3),'names':['height','width','channel']},'wrist_image':{'dtype':'image','shape':(256,256,3),'names':['height','width','channel']},'state':{'dtype':'float32','shape':(8,),'names':['tcp_x_m','tcp_y_m','tcp_z_m','rotvec_x_rad','rotvec_y_rad','rotvec_z_rad','Left_1_Joint_rad','Right_1_Joint_rad']},'actions':{'dtype':'float32','shape':(7,),'names':['osc_dx','osc_dy','osc_dz','osc_drx','osc_dry','osc_drz','gripper_close_command']},'next.done':{'dtype':'bool','shape':(1,),'names':None}}
 root=a.out/'datasets/test';repo='local/writer_probe_20261007';source=pathlib.Path(__file__).resolve().parents[3]/'runs/formal_20261005_r1/task03_s620300/trajectory.h5'
 a.out.mkdir(parents=True,exist_ok=False);states=[];actions=[];pixels=[];rss=[]
 ds=writer(repo,root,features)
 with h5py.File(source,'r') as f:
  for ep in range(3):
   if ep==2:
    ds.stop_image_writer();del ds;gc.collect();ds=writer(repo,root,features)
   for i in range(8):
    j=ep*8+i;frame={'image':f['observations/image'][j],'wrist_image':f['observations/wrist_image'][j],'state':f['observations/state'][j].astype(np.float32),'actions':f['transitions/actions'][j].astype(np.float32),'next.done':np.array([i==7],dtype=bool),'task':str(f.attrs['language'])}
    states.append(frame['state'].copy());actions.append(frame['actions'].copy());pixels.append(frame['image'].copy());ds.add_frame(frame)
   ds.save_episode();ds.hf_dataset=ds.create_hf_dataset();gc.collect();assert len(ds.hf_dataset)==0 and ds.meta.total_episodes==ep+1
   rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 ds.stop_image_writer();del ds;gc.collect()
 reader=LeRobotDataset(repo_id=repo,root=root,delta_timestamps={'actions':[i/20 for i in range(10)]},video_backend='pyav')
 assert len(reader)==24 and reader.num_episodes==3
 for index in range(24):
  item=reader[index];i=index%8;ep=index//8
  assert np.array_equal(item['state'].numpy(),states[index])
  assert np.array_equal(item['image'].numpy(),pixels[index].transpose(2,0,1).astype(np.float32)/np.float32(255))
  inds=np.minimum(np.arange(i,i+10),7)+ep*8
  assert np.array_equal(item['actions'].numpy(),np.asarray(actions)[inds])
  assert np.array_equal(item['actions_is_pad'].numpy(),np.arange(i,i+10)>=8)
  assert int(item['index'])==index and int(item['episode_index'])==ep and int(item['frame_index'])==i
  assert bool(item['next.done'].item())==(i==7)
 result={'pass':True,'episodes':3,'frames':24,'writer_resumed_after_episode':2,'complete_pixel_numeric_index_terminal_and_chunk_equal':True,'in_memory_saved_dataset_empty_each_episode':True,'max_rss_samples_kib':rss}
 (a.out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
