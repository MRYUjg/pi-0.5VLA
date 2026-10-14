import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import subprocess
import sys
import numpy as np
import pyarrow.parquet as pq
from openpi.shared import normalize
from configuration import *
from training_utils import write
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();out=a.out
    assert json.loads((EXPORT/'validation.json').read_text())['pass']
    assert json.loads((ROOT/'runs/formal_merged_20261006_r1/summary.json').read_text())['pass']
    hashes=json.loads((EXPORT/'dataset_files_sha256.json').read_text())
    for rel,want in hashes.items():assert sha(EXPORT/rel)==want,rel
    train=json.loads((EXPORT/'datasets'/TRAIN_ID/'provenance.json').read_text())
    val=json.loads((EXPORT/'datasets'/VAL_ID/'provenance.json').read_text())
    train_seeds={r['source_seed'] for r in train['episodes']};val_seeds={r['source_seed'] for r in val['episodes']}
    assert train_seeds.isdisjoint(val_seeds)
    out.mkdir(parents=True,exist_ok=False)
    stats={}
    for key in ('state','actions'):
        rows=[np.asarray(pq.read_table(f,columns=[key])[key].to_pylist()) for f in sorted((EXPORT/'datasets'/TRAIN_ID/'data').rglob('*.parquet'))]
        values=np.concatenate(rows);assert len(values)==train['frames'] and np.isfinite(values).all()
        running=normalize.RunningStats();running.update(values);stats[key]=running.get_statistics()
    normalize.save(out/'assets'/TRAIN_ID,stats)
    sources={str(p.relative_to(REPO)):sha(p) for d in (VERSION,REPO/'src/openpi') for p in d.rglob('*.py')}
    sources['scripts/train.py']=sha(REPO/'scripts/train.py')
    tasks=sorted({r['scene']['task_id'] for r in train['episodes']})
    assert tasks==sorted({r['scene']['task_id'] for r in val['episodes']})
    write(out/'protocol.json',{'version':VERSION.name,'source_hashes':sources,'dataset_hashes':hashes,
        'base_checkpoint':str(BASE),'base_hashes':{str(p.relative_to(BASE)):sha(p) for p in BASE.rglob('*') if p.is_file()},
        'train_id':TRAIN_ID,'validation_id':VAL_ID,'tasks':tasks,'train_frames':train['frames'],'validation_frames':val['frames'],
        'train_episodes':len(train['episodes']),'validation_episodes':len(val['episodes']),
        'batch_size':BATCH,'steps':STEPS,'action_horizon':10,'save_every':SAVE_EVERY,
        'trainable':'LoRA only; all vision and non-LoRA parameters frozen',
        'sampling':'round-robin task balance, uniform episode then uniform frame within task',
        'normalization':'train only, used unchanged for validation and inference',
        'state_contract':'8D proprioception retained by dataset/interface; base pi05_libero uses discrete_state_input=False, so model conditioning remains image+language',
        'selection':'minimum fixed validation loss among scheduled saved checkpoints; no test set access',
        'seed':42,'acceptance':{'finite_losses_gradients':True,'lora_changed':True,'non_lora_unchanged':True,
            'independent_checkpoint_reload_prediction_max_error':1e-5,'policy_success':'measured separately; no target assumed from loss'},
        'environment':{'python':sys.version,'packages':{n:importlib.metadata.version(n) for n in ('jax','jaxlib','flax','orbax-checkpoint','torch','datasets','lerobot')},
            'gpu':subprocess.check_output(['nvidia-smi'],text=True)}})
    print('PREPARE PASS',tasks,train['frames'],val['frames'],flush=True)
