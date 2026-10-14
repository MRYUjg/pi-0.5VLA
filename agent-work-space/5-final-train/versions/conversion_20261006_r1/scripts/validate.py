"""Independent full-frame readback through actual LeRobot API, with episode-boundary checks."""
import argparse,copy,pathlib
import h5py,numpy as np,torch
from PIL import Image,ImageDraw
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from common import *
def array(v):return v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else np.asarray(v)
def compare_frame(item,f,i,n,ep_idx,global_idx,language):
    require(np.array_equal(array(item['state']),f['observations/state'][i].astype(np.float32)),'state mismatch')
    require(np.array_equal(array(item['actions']),f['transitions/actions'][i].astype(np.float32)),'action mismatch')
    for key in ('image','wrist_image'):
        actual=array(item[key]);raw=f['observations/'+key][i];expected=raw.transpose(2,0,1).astype(np.float32)/np.float32(255)
        require(actual.shape==(3,256,256) and np.array_equal(actual,expected),'image mismatch: '+key)
    require(item['task']==language,'language mismatch')
    require(int(item['frame_index'])==i and int(item['episode_index'])==ep_idx and int(item['index'])==global_idx,'frame index mismatch')
    require(float(item['timestamp'])==float(np.float32(i/20)),'timestamp mismatch')
    require(bool(item['next.done'].item())==(i==n-1),'terminal mismatch')
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=pathlib.Path(a.out)
    require(not (out/'validation.json').exists(),'validation output already exists')
    torch.set_num_threads(1);selected,fingerprints=inspect_sources();protocol=read(out/'protocol.json')
    require(protocol['source_hashes']==fingerprints,'source provenance drift')
    totals={};frame_count=0;chunk_count=0;all_seeds=set();max_cast_error=0.;negatives=[]
    for split,repo in REPO_IDS.items():
        root=out/'datasets'/repo;prov=read(root/'provenance.json');records=selected[split]
        if protocol['limit_each']:records=records[:protocol['limit_each']]
        require([r['seed'] for r in records]==[r['source_seed'] for r in prov['episodes']],'dataset source selection mismatch')
        ds=LeRobotDataset(repo_id=repo,root=root,video_backend='pyav');chunks=LeRobotDataset(repo_id=repo,root=root,delta_timestamps={'actions':[i/20 for i in range(50)]},video_backend='pyav')
        require(ds.fps==20 and len(ds)==prov['frames'] and ds.num_episodes==len(records),'dataset counts/fps')
        require(ds.meta.info['robot_type']=='eco65_zhixing','robot type');require(not ds.meta.video_keys,'expected lossless image dataset')
        stats_source={'state':[],'actions':[]};ep_results=[]
        for mapping in prov['episodes']:
            seed=mapping['source_seed'];require(seed not in all_seeds,'split leakage');all_seeds.add(seed)
            ep=COLLECTION/mapping['source_relative'];n=mapping['length'];start=mapping['index_from'];idx=mapping['episode_index']
            require(int(ds.episode_data_index['from'][idx])==start and int(ds.episode_data_index['to'][idx])==start+n,'episode boundaries')
            with h5py.File(ep/'trajectory.h5','r') as f:
                require(n==len(f['transitions/actions']) and len(f['observations/state'])==n+1,'source T+1 contract')
                actions=f['transitions/actions'][...].astype(np.float32);state=f['observations/state'][:-1].astype(np.float32)
                stats_source['actions'].append(actions);stats_source['state'].append(state)
                cast_error=float(np.max(abs(f['transitions/actions'][...]-actions.astype(np.float64))));max_cast_error=max(max_cast_error,cast_error)
                for i in range(n):compare_frame(ds[start+i],f,i,n,idx,start+i,mapping['language'])
                for i in sorted({0,n//2,max(0,n-50),n-2,n-1}):
                    item=chunks[start+i];inds=np.minimum(np.arange(i,i+50),n-1)
                    require(np.array_equal(array(item['actions']),actions[inds]),'action chunk crossed episode or mismatch')
                    require(np.array_equal(array(item['actions_is_pad']),np.arange(i,i+50)>=n),'action chunk pad mask mismatch');chunk_count+=1
                if not negatives:
                    baseline=ds[start];compare_frame(baseline,f,0,n,idx,start,mapping['language'])
                    for case,expected in [('action','action mismatch'),('state','state mismatch'),('image','image mismatch: image'),('time','timestamp mismatch'),('terminal','terminal mismatch'),('language','language mismatch')]:
                        bad=copy.deepcopy(baseline)
                        if case=='action':bad['actions'][0]+=.1
                        elif case=='state':bad['state'][0]+=.1
                        elif case=='image':bad['image'][0,0,0]=1-bad['image'][0,0,0]
                        elif case=='time':bad['timestamp']+=.05
                        elif case=='terminal':bad['next.done']=torch.tensor([True])
                        else:bad['task']='wrong task'
                        try:compare_frame(bad,f,0,n,idx,start,mapping['language'])
                        except ValueError as e:require(str(e)==expected,'unexpected rejection');negatives.append({'case':case,'rejected':True,'error':str(e)})
                        else:raise AssertionError('corruption accepted: '+case)
                if idx==0:
                    sheet=Image.new('RGB',(256*3,540),'#222222');draw=ImageDraw.Draw(sheet)
                    for col,i in enumerate((0,n//2,n-1)):
                        item=ds[start+i]
                        for row,key in enumerate(('image','wrist_image')):
                            pixels=np.rint(array(item[key]).transpose(1,2,0)*255).astype(np.uint8);sheet.paste(Image.fromarray(pixels),(col*256,25+row*256))
                        draw.text((col*256+5,5),f'{split} seed {seed} frame {i}',fill='white')
                    sheet.save(out/(split+'_decoded_preview.png'))
            frame_count+=n;ep_results.append({'seed':seed,'frames':n,'all_numeric_and_rgb_frames_equal':True,'float64_to_float32_action_max_error':cast_error});print(split,seed,'all',n,'frames verified',flush=True)
        numeric_stats={}
        for key,parts in stats_source.items():
            values=np.concatenate(parts).astype(np.float64);observed=ds.meta.stats[key]
            expected={'mean':values.mean(0),'std':values.std(0),'min':values.min(0),'max':values.max(0)}
            require(int(observed['count'][0])==len(values),'stats count leak')
            for stat,want in expected.items():require(np.allclose(observed[stat],want,atol=1e-5,rtol=1e-5),'numeric stats mismatch: '+key+'/'+stat)
            numeric_stats[key]={'count':len(values),**{k:v.tolist() for k,v in expected.items()},'max_mean_error':float(np.max(abs(observed['mean']-expected['mean'])))}
        loader=torch.utils.data.DataLoader(chunks,batch_size=4,num_workers=0,shuffle=False);batch=next(iter(loader))
        require(batch['actions'].shape==(4,50,7) and batch['state'].shape==(4,8) and batch['image'].shape==(4,3,256,256),'CPU DataLoader shape')
        require(batch['actions'].device.type=='cpu','CPU expected')
        totals[split]={'episodes':ep_results,'frames':len(ds),'numeric_stats':numeric_stats,'dataloader_batch_shapes':{k:list(batch[k].shape) for k in ('actions','state','image','wrist_image','actions_is_pad')},'stats_scope':split+' only'}
        del ds,chunks,loader,batch
    negative_inputs=[]
    base=copy.deepcopy(selected['train'][0])
    for case,record,seen,expected in [('unaccepted',{**base,'accepted':False},set(),'source not accepted'),('duplicate',base,{base['seed']},'duplicate or cross-split seed'),('wrong_split',{**base,'stats':{**base['stats'],'scene':{**base['stats']['scene'],'split':'test'}}},set(),'split mismatch')]:
        try:check_records([record],'train',seen)
        except ValueError as e:require(str(e)==expected,'wrong input rejection');negative_inputs.append({'case':case,'rejected':True,'error':str(e)})
        else:raise AssertionError('bad source accepted')
    for rel,want in fingerprints.items():require(sha(COLLECTION/rel)==want,'source changed during validation')
    write(out/'validation.json',{'pass':True,'frames_verified':frame_count,'rgb_images_verified':frame_count*2,'raw_terminal_observations_excluded':len(all_seeds),'max_rgb_pixel_error':0,'max_action_float32_cast_error':max_cast_error,'action_chunks_checked':chunk_count,'action_horizon':50,'splits':totals,'negative_frame_tests':negatives,'negative_input_tests':negative_inputs,'source_unchanged':True,'training_executed':False,'gpu_used':False,'openpi_scope':'LeRobot public reader + CPU torch DataLoader verified; existing OpenPI LIBERO key/shape contract retained; full OpenPI training/model pipeline not executed'})
    write(out/'dataset_files_sha256.json',{str(p.relative_to(out)):sha(p) for p in (out/'datasets').rglob('*') if p.is_file()})
    print('VALIDATION PASSED',frame_count,'frames',flush=True)
if __name__=='__main__':main()
