import argparse,hashlib,json,pathlib,importlib.metadata,subprocess,sys
import numpy as np,pyarrow.parquet as pq
from openpi.shared import normalize
from configuration import *
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=False)
 hashes=json.loads((EXPORT/'dataset_files_sha256.json').read_text())
 for n,h in hashes.items():assert sha(EXPORT/n)==h,n
 stats={}
 for key in ('state','actions'):
  rows=[np.asarray(pq.read_table(f,columns=[key])[key].to_pylist()) for f in sorted((EXPORT/'datasets'/TRAIN_ID/'data').rglob('*.parquet'))]
  values=np.concatenate(rows);assert len(values)==5226
  running=normalize.RunningStats();running.update(values);stats[key]=running.get_statistics()
 normalize.save(out/'assets'/TRAIN_ID,stats)
 source={str(p.relative_to(REPO)):sha(p) for d in (VERSION,REPO/'src/openpi') for p in d.rglob('*.py') if '__pycache__' not in str(p)}
 source['scripts/train.py']=sha(REPO/'scripts/train.py')
 base={str(p.relative_to(BASE)):sha(p) for p in BASE.rglob('*') if p.is_file()}
 write(out/'protocol.json',{'version':VERSION.name,'source_hashes':source,'dataset_hashes':hashes,'base_checkpoint':str(BASE),'base_hashes':base,'batch_size':1,'action_horizon':10,'trainable':'lora parameters only; vision and all non-lora frozen','overfit_steps':100,'short_finetune_steps':200,'norm_scope':'train only 5226 rows','train_id':TRAIN_ID,'validation_id':VAL_ID,'seed':42,'acceptance':{'finite_loss_and_gradients':True,'lora_weights_changed':True,'frozen_weights_unchanged':True,'fixed_overfit_eval_loss_relative_reduction_target':0.2,'checkpoint_reload_prediction_max_error':1e-5,'validation_loss':'report only; not a task-success claim'},'environment':{'python':sys.version,'packages':{n:importlib.metadata.version(n) for n in ('jax','jaxlib','flax','orbax-checkpoint','torch','datasets','lerobot')},'gpu':subprocess.check_output(['nvidia-smi'],text=True)}})
 print('PREPARE PASS train-only norm and source/base/data hashes',flush=True)
if __name__=='__main__':main()
