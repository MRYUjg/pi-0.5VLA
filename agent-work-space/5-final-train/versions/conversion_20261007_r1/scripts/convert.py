"""CPU-only, local LeRobot v2.1 image dataset conversion. No training / Hub upload."""
import argparse,gc,importlib.metadata,os,pathlib,sys,resource
import h5py,numpy as np,torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset,CODEBASE_VERSION
from common import *
from resume_writer import resume_prefix,writer
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--limit-each',type=int);a=p.parse_args()
    out=pathlib.Path(a.out);require(not out.exists(),'output already exists');require(a.limit_each is None or a.limit_each>0,'limit must be positive')
    selected,fingerprints=inspect_sources();torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',{'version':VERSION.name,'lerobot_commit':COMMIT,'format':CODEBASE_VERSION,'fps':20,'source_batch':str(BATCH),'source_hashes':fingerprints,'source_code_hashes':{str(p.relative_to(VERSION)):sha(p) for p in VERSION.rglob('*') if p.is_file() and '__pycache__' not in str(p)},'limit_each':a.limit_each,'training_run':False,'gpu_used':False,'packages':{n:importlib.metadata.version(n) for n in ('torch','torchvision','datasets','pyarrow','h5py','numpy','pillow')}})
    features={'image':{'dtype':'image','shape':(256,256,3),'names':['height','width','channel']},'wrist_image':{'dtype':'image','shape':(256,256,3),'names':['height','width','channel']},'state':{'dtype':'float32','shape':(8,),'names':['tcp_x_m','tcp_y_m','tcp_z_m','rotvec_x_rad','rotvec_y_rad','rotvec_z_rad','Left_1_Joint_rad','Right_1_Joint_rad']},'actions':{'dtype':'float32','shape':(7,),'names':['osc_dx','osc_dy','osc_dz','osc_drx','osc_dry','osc_drz','gripper_close_command']},'next.done':{'dtype':'bool','shape':(1,),'names':None}}
    results={}
    for split,records in selected.items():
        if a.limit_each:records=records[:a.limit_each]
        repo=REPO_IDS[split];root=out/'datasets'/repo
        prefix=resume_prefix(split,root,records,features,fingerprints)
        ds=writer(repo,root,features)
        mapping=[];offset=0
        try:
            for episode_idx,r in enumerate(records):
                ep=COLLECTION/r['source_relative']
                with h5py.File(ep/'trajectory.h5','r') as f:
                    n=len(f['transitions/actions']);language=str(f.attrs['language']);require(bool(language.strip()),'empty language')
                    if episode_idx>=prefix:
                        for i in range(n):
                            ds.add_frame({'image':f['observations/image'][i],'wrist_image':f['observations/wrist_image'][i],'state':f['observations/state'][i].astype(np.float32),'actions':f['transitions/actions'][i].astype(np.float32),'next.done':np.array([i==n-1],dtype=bool),'task':language})
                        ds.save_episode()
                        # Write-only use: persisted metadata keeps episode/global indices.
                        # Drop in-memory concatenated image bytes after each episode.
                        ds.hf_dataset=ds.create_hf_dataset();gc.collect()
                    mapping.append({'episode_index':episode_idx,'source_seed':r['seed'],'scene':r['stats']['scene'],'source_relative':str(ep.relative_to(COLLECTION)),'source_h5_sha256':fingerprints[str((ep/'trajectory.h5').relative_to(COLLECTION))],'length':n,'index_from':offset,'index_to':offset+n,'language':language,'source_sim_time_start':float(f['observations/time'][0]),'source_terminal_observation_index':n,'terminal_observation_in_lerobot':False})
                    offset+=n
                print(split,r['seed'],n,'reused' if episode_idx<prefix else 'saved','max_rss_kib',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
        finally:ds.stop_image_writer()
        provenance={'split':split,'repo_id':repo,'fps':20,'episodes':mapping,'frames':offset,'action_contract':'normalized OSC; xyz scale 0.05m, rotation 0.5rad; gripper -1 open/+1 close; no extra delta transform','image_contract':'raw HDF5 RGB unchanged; lossless embedded PNG','terminal_contract':'next.done means recorded episode end, not early environment done; final o_T remains in source HDF5','stats_scope':split+' only; never merge validation stats into train'}
        write(root/'provenance.json',provenance);results[split]={'repo_id':repo,'episodes':len(mapping),'frames':offset}
        del ds
    for rel,want in fingerprints.items():require(sha(COLLECTION/rel)==want,'source modified during conversion')
    for p in out.rglob('resume_prefix.json'):
        for rel,want in read(p)['source_parquet_sha256'].items():require(sha(COLLECTION/rel)==want,'old export changed while resuming')
    write(out/'conversion.json',{'completed':True,'datasets':results,'original_sources_unchanged':True,'acceptance_pending_validation':True})
if __name__=='__main__':main()
