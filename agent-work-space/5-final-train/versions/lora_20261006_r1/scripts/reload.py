"""Independent full-state restore, fixed-noise prediction and unsaved resume check."""
import argparse,functools,json,pathlib
import flax.nnx as nnx
import jax,numpy as np,torch
from openpi.training import sharding,checkpoints
from configuration import *
from training_utils import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();out=a.out
 assert not (out/'reload_result.json').exists()
 assert json.loads((out/'training_result.json').read_text())['pass']
 torch.set_num_threads(1);jax.config.update('jax_compilation_cache_dir',str(out/'jax_cache'))
 cfg=make(out);core=load_core();train,dc=make_data(cfg,'train');val,_=make_data(cfg,'validation')
 selected=json.loads((out/'selected_checkpoint.json').read_text());step=selected['step']
 shape,_=core.init_train_state(cfg,jax.random.key(42),sharding.make_mesh(1),resume=True)
 manager,_=checkpoints.initialize_checkpoint_dir(out/'checkpoint',keep_period=SAVE_EVERY,overwrite=False,resume=True)
 state=checkpoints.restore_state(manager,shape,Holder(dc),step);manager.close();jax.block_until_ready(state);assert int(state.step)==step
 restored=fingerprint(state.params);expected=json.loads((out/f'params_{step}.json').read_text());assert restored==expected,'restored parameter bytes differ'
 index=json.loads((out/'evaluation_indices.json').read_text())['validation'][0]['index']
 o,ac=batch(val,index);graph=state.model_def
 @jax.jit
 def predict(params,obs):
  m=nnx.merge(graph,params);m.eval();return m.sample_actions(jax.random.key(777),obs,num_steps=10)
 prediction=np.asarray(predict(state.params,o));want=np.load(out/f'prediction_{step}.npy')
 error=float(np.max(abs(prediction-want)));assert np.isfinite(prediction).all() and error<=1e-5,error
 from train import multi_batch
 step_fn=jax.jit(functools.partial(core.train_step,cfg),donate_argnums=(1,))
 state,info=step_fn(jax.random.key(42),state,multi_batch(train,list(range(BATCH))));jax.block_until_ready(state)
 info={k:float(v) for k,v in info.items()};assert int(state.step)==step+1 and np.isfinite(list(info.values())).all()
 write(out/'reload_result.json',{'pass':True,'independent_process':True,'all_restored_parameter_bytes_equal':True,
  'fixed_noise_prediction_max_error':error,'prediction_shape':list(prediction.shape),'restored_step':step,
  'resume_test_step':step+1,'resume_metrics':info,'resume_test_saved_over_original':False,
  'scope':'checkpoint mechanics; closed-loop policy performance evaluated separately'})
 print('RELOAD PASS',step,error,flush=True)
if __name__=='__main__':main()
