import argparse,functools,json,pathlib,time,hashlib
import flax.nnx as nnx
import jax,jax.numpy as jnp,numpy as np,torch
from openpi.training import sharding,checkpoints
from configuration import *
from training_utils import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();out=a.out;assert not (out/'training_started.json').exists()
 torch.set_num_threads(1);jax.config.update('jax_compilation_cache_dir',str(out/'jax_cache'))
 write(out/'training_started.json',{'time':time.time(),'source_hashes':{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in VERSION.rglob('*') if p.is_file() and '__pycache__' not in str(p)}})
 cfg=make(out);core=load_core();train,dc=make_data(cfg,'train');val,_=make_data(cfg,'validation')
 assert len(train)==5226 and len(val)==1721
 over_indices=np.linspace(0,570,8,dtype=int).tolist();val_indices=[]
 for rec in json.loads((EXPORT/'datasets'/VAL_ID/'provenance.json').read_text())['episodes']:
  val_indices += (rec['index_from']+np.linspace(0,rec['length']-10,4,dtype=int)).tolist()
 write(out/'evaluation_indices.json',{'overfit_train':over_indices,'validation':val_indices,'fixed_eval_rng_base':1000,'inference_rng':777,'training_rng':42})
 over_batches=[batch(train,i) for i in over_indices];val_batches=[batch(val,i) for i in val_indices]
 print('DATA READY',len(train),len(val),flush=True)
 mesh=sharding.make_mesh(1);state,state_sharding=core.init_train_state(cfg,jax.random.key(42),mesh,resume=False);jax.block_until_ready(state)
 before=fingerprint(state.params);write(out/'params_before.json',before)
 trainable=[k for k in before if 'lora' in k];assert trainable
 counts={'total':sum(r['elements'] for r in before.values()),'trainable':sum(before[k]['elements'] for k in trainable),'trainable_leaves':trainable}
 write(out/'parameter_counts.json',counts);print('PARAMETERS',counts['total'],counts['trainable'],flush=True)
 graphdef=state.model_def
 @jax.jit
 def evaluate(params,obs,actions,rng):
  m=nnx.merge(graphdef,params);m.eval();return jnp.mean(m.compute_loss(rng,obs,actions,train=False))
 def eval_set(batches):
  return [float(evaluate(state.params,o,ac,jax.random.key(1000+i))) for i,(o,ac) in enumerate(batches)]
 initial_over=eval_set(over_batches);initial_val=eval_set(val_batches)
 write(out/'baseline_eval.json',{'overfit':initial_over,'validation':initial_val})
 print('BASELINE',np.mean(initial_over),np.mean(initial_val),flush=True)
 step_fn=jax.jit(functools.partial(core.train_step,cfg),donate_argnums=(1,))
 rng=jax.random.key(42);sampler=np.random.default_rng(42);metrics=[]
 with (out/'metrics.jsonl').open('x') as log:
  for step in range(300):
   idx=over_indices[step%8] if step<100 else int(sampler.integers(len(train)))
   obs,actions=over_batches[step%8] if step<100 else batch(train,idx)
   start=time.time();state,info=step_fn(rng,state,(obs,actions));jax.block_until_ready(state)
   row={'step':step+1,'stage':'overfit' if step<100 else 'short_finetune','source_frame_index':idx,'seconds':time.time()-start,**{k:float(v) for k,v in info.items()}}
   assert np.isfinite(list(row[k] for k in ('loss','grad_norm','param_norm'))).all(),row
   metrics.append(row);log.write(json.dumps(row)+'\n');log.flush()
   if step==0 or (step+1)%10==0:print('STEP',step+1,'loss',row['loss'],'grad',row['grad_norm'],'seconds',row['seconds'],flush=True)
   if step==99:
    over_eval=eval_set(over_batches);mid_val=eval_set(val_batches)
    write(out/'overfit_eval.json',{'overfit':over_eval,'validation':mid_val,'relative_reduction':1-np.mean(over_eval)/np.mean(initial_over)})
    print('OVERFIT EVAL',np.mean(over_eval),'VAL',np.mean(mid_val),flush=True)
 final_over=eval_set(over_batches);final_val=eval_set(val_batches)
 after=fingerprint(state.params);write(out/'params_after.json',after)
 changed=[k for k in before if before[k]['sha256']!=after[k]['sha256']];frozen_changed=[k for k in changed if 'lora' not in k]
 assert not frozen_changed and changed
 manager,_=checkpoints.initialize_checkpoint_dir(out/'checkpoint',keep_period=300,overwrite=False,resume=False)
 checkpoints.save_state(manager,state,Holder(dc),300);manager.wait_until_finished();manager.close();print('CHECKPOINT SAVED',flush=True)
 @jax.jit
 def predict(params,obs):
  m=nnx.merge(graphdef,params);m.eval();return m.sample_actions(jax.random.key(777),obs,num_steps=10)
 prediction=np.asarray(predict(state.params,val_batches[0][0]));assert prediction.shape==(1,10,32) and np.isfinite(prediction).all();np.save(out/'prediction_before_reload.npy',prediction)
 reduction=1-float(np.mean(over_eval))/float(np.mean(initial_over));passed=not frozen_changed and bool(changed) and reduction>=.2 and any(r['grad_norm']>0 for r in metrics)
 write(out/'training_result.json',{'pass':passed,'steps':int(state.step),'overfit_relative_loss_reduction':reduction,'overfit_target':.2,'baseline_overfit_mean':float(np.mean(initial_over)),'after_overfit_mean':float(np.mean(over_eval)),'final_overfit_mean':float(np.mean(final_over)),'baseline_validation_mean':float(np.mean(initial_val)),'after_overfit_validation_mean':float(np.mean(mid_val)),'final_validation_mean':float(np.mean(final_val)),'all_gradients_finite':True,'changed_lora_leaves':len(changed),'changed_non_lora_leaves':frozen_changed,'checkpoint':'checkpoint/300','prediction_shape':list(prediction.shape),'gpu_memory_stats':jax.devices()[0].memory_stats(),'scope':'training mechanics + limited-data loss; not task success acceptance'})
 print('TRAINING RESULT',passed,'overfit reduction',reduction,flush=True)
 return int(not passed)
if __name__=='__main__':raise SystemExit(main())
