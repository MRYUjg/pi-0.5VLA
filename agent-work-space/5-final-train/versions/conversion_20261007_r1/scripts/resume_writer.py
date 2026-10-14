"""Copy mutable metadata; hard-link immutable completed Parquet; bounded write-only API."""
import json,os,pathlib,shutil
import h5py,numpy as np,pyarrow.parquet as pq
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset,LeRobotDatasetMetadata
from common import *
OLD=COLLECTION/'lerobot_exports/spatial_formal_r2'
OLD_IDS={s:f'local/eco65_zhixing_spatial_formal_r2_{s}' for s in REPO_IDS}
def resume_prefix(split,root,records,features,fingerprints):
    old=OLD/'datasets'/OLD_IDS[split]
    if not old.exists():return 0
    protocol=read(OLD/'protocol.json')
    require(protocol['source_hashes']==fingerprints,'old export raw source drift')
    require(protocol['lerobot_commit']==COMMIT and protocol['fps']==20 and protocol['limit_each'] is None,'old conversion protocol mismatch')
    info=read(old/'meta/info.json');n=info['total_episodes']
    require(0<n<=len(records),'invalid complete prefix')
    episodes=[json.loads(s) for s in (old/'meta/episodes.jsonl').read_text().splitlines()]
    stats=[json.loads(s) for s in (old/'meta/episodes_stats.jsonl').read_text().splitlines()]
    require(len(episodes)==len(stats)==n,'incomplete metadata prefix')
    require(all(ep['episode_index']==i for i,ep in enumerate(episodes)),'nonsequential prefix')
    tasks={row['task_index']:row['task'] for row in [json.loads(s) for s in (old/'meta/tasks.jsonl').read_text().splitlines()]}
    root.mkdir(parents=True,exist_ok=False);shutil.copytree(old/'meta',root/'meta')
    sources={str(p.relative_to(COLLECTION)):sha(p) for p in (old/'meta').rglob('*') if p.is_file()};offset=0
    for i,(r,epmeta) in enumerate(zip(records[:n],episodes)):
        rel=info['data_path'].format(episode_chunk=i//info['chunks_size'],episode_index=i)
        source=old/rel;dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        table=pq.read_table(source,columns=['state','actions','frame_index','episode_index','index','timestamp','next.done','task_index'])
        with h5py.File(COLLECTION/r['source_relative']/'trajectory.h5','r') as f:
            length=len(f['transitions/actions']);require(table.num_rows==length==epmeta['length'],'prefix episode length')
            arrays={k:np.asarray(table[k].to_pylist()) for k in table.column_names}
            require(np.array_equal(arrays['state'],f['observations/state'][:-1].astype(np.float32)),'prefix state')
            require(np.array_equal(arrays['actions'],f['transitions/actions'][...].astype(np.float32)),'prefix action')
            require(np.array_equal(arrays['frame_index'].reshape(-1),np.arange(length)),'prefix frame indices')
            require(np.all(arrays['episode_index']==i),'prefix episode indices')
            require(np.array_equal(arrays['index'].reshape(-1),np.arange(offset,offset+length)),'prefix global indices')
            require(np.array_equal(arrays['timestamp'].reshape(-1),np.arange(length,dtype=np.float32)/np.float32(20)),'prefix time')
            done=np.zeros(length,dtype=bool);done[-1]=True;require(np.array_equal(arrays['next.done'].reshape(-1),done),'prefix terminal')
            require(all(tasks[int(t)]==str(f.attrs['language']) for t in arrays['task_index'].reshape(-1)),'prefix language')
        os.link(source,dest);sources[str(source.relative_to(COLLECTION))]=sha(source);offset+=length
    require(offset==info['total_frames'],'prefix frame metadata mismatch')
    write(root/'resume_prefix.json',{'episodes_reused':n,'frames_reused':offset,'source_parquet_sha256':sources,
        'mutable_metadata_copied':True,'immutable_parquet_hardlinked':True,'uncommitted_tail_excluded':True,
        'pixel_equality_pending_full_reader_validation':True})
    return n

def writer(repo,root,features):
    if not root.exists():
        return LeRobotDataset.create(repo_id=repo,root=root,robot_type='eco65_zhixing',fps=20,features=features,
            use_videos=False,image_writer_threads=2,image_writer_processes=0,video_backend='pyav')
    # Match pinned LeRobot.create fields; avoid loading an entire completed prefix for write-only use.
    ds=LeRobotDataset.__new__(LeRobotDataset);ds.meta=LeRobotDatasetMetadata(repo_id=repo,root=root)
    require(ds.meta.fps==20 and ds.meta.info['robot_type']=='eco65_zhixing','resumed writer metadata')
    require(all(json.loads(json.dumps(ds.meta.features[k]))==json.loads(json.dumps(v)) for k,v in features.items()),'resumed writer features changed')
    ds.repo_id=repo;ds.root=root;ds.revision=None;ds.tolerance_s=1e-4;ds.image_writer=None
    ds.episodes=None;ds.image_transforms=None;ds.delta_timestamps=None;ds.delta_indices=None
    ds.episode_data_index=None;ds.video_backend='pyav'
    ds.hf_dataset=ds.create_hf_dataset();ds.episode_buffer=ds.create_episode_buffer();ds.start_image_writer(num_threads=2)
    return ds
