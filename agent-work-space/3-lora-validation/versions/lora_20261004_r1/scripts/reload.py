import argparse,functools,json,pathlib
import flax.nnx as nnx
import jax,numpy as np,torch
from openpi.training import sharding,checkpoints
from configuration import *
from training_utils import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();out=a.out;assert not (out/'reload_result.json').exists()
 torch.set_num_threads(1);cfg=make(out);core=load_core();train,dc=make_data(cfg,'train');val,_=make_data(cfg,'validation')
 shape,_=core.init_train_state(cfg,jax.random.key(42),sharding.make_mesh(1),resume=True)
 manager,_=checkpoints.initialize_checkpoint_dir(out/'checkpoint',keep_period=300,overwrite=False,resume=True)
 state=checkpoints.restore_state(manager,shape,Holder(dc),300);manager.close();jax.block_until_ready(state);assert int(state.step)==300
 restored=fingerprint(state.params);expected=json.loads((out/'params_after.json').read_text());assert restored==expected,'restored parameter bytes differ'
 o,ac=batch(val,0);graph=state.model_def
 @jax.jit
 def predict(params,obs):
  m=nnx.merge(graph,params);m.eval();return m.sample_actions(jax.random.key(777),obs,num_steps=10)
 prediction=np.asarray(predict(state.params,o));want=np.load(out/'prediction_before_reload.npy');error=float(np.max(abs(prediction-want)));assert np.isfinite(prediction).all() and error<=1e-5,error
 step_fn=jax.jit(functools.partial(core.train_step,cfg),donate_argnums=(1,));state,info=step_fn(jax.random.key(42),state,batch(train,0));jax.block_until_ready(state);info={k:float(v) for k,v in info.items()};assert int(state.step)==301 and np.isfinite(list(info.values())).all()
 write(out/'reload_result.json',{'pass':True,'independent_process':True,'all_restored_parameter_bytes_equal':True,'fixed_noise_prediction_max_error':error,'prediction_shape':list(prediction.shape),'restored_step':300,'resume_test_step':301,'resume_metrics':info,'resume_test_saved_over_original':False,'gpu_memory_stats':jax.devices()[0].memory_stats()})
 print('RELOAD PASS',error,'RESUME STEP',int(state.step),flush=True)
if __name__=='__main__':main()
