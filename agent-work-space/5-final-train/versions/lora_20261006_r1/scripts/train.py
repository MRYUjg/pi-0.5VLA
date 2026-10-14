import argparse
import collections
import functools
import hashlib
import json
import pathlib
import time
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from openpi.training import sharding,checkpoints
from openpi.models import model as model_lib
from configuration import *
from training_utils import *

def multi_batch(ds,indices):
    items=[ds[int(i)] for i in indices]
    item=jax.tree.map(lambda *xs:np.stack([np.asarray(x) for x in xs]),*items)
    return model_lib.Observation.from_dict(item),item['actions']

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();out=a.out
    result_name='resource_smoke_result.json' if a.smoke else 'training_result.json'
    assert not (out/result_name).exists()
    if not a.smoke:assert json.loads((out/'resource_smoke_result.json').read_text())['pass']
    torch.set_num_threads(1);jax.config.update('jax_compilation_cache_dir',str(out/'jax_cache'))
    protocol=json.loads((out/'protocol.json').read_text())
    for rel,want in protocol['source_hashes'].items():assert hashlib.sha256((REPO/rel).read_bytes()).hexdigest()==want,rel
    cfg=make(out);core=load_core();train,dc=make_data(cfg,'train');val,_=make_data(cfg,'validation')
    assert len(train)==protocol['train_frames'] and len(val)==protocol['validation_frames']
    train_prov=json.loads((EXPORT/'datasets'/TRAIN_ID/'provenance.json').read_text())
    val_prov=json.loads((EXPORT/'datasets'/VAL_ID/'provenance.json').read_text())
    by_task=collections.defaultdict(list)
    for ep in train_prov['episodes']:by_task[ep['scene']['task_id']].append(ep)
    tasks=sorted(by_task);evaluation=[]
    for ep in val_prov['episodes']:
        for i in np.linspace(0,max(0,ep['length']-10),4,dtype=int):evaluation.append({'task_id':ep['scene']['task_id'],'index':ep['index_from']+int(i)})
    if not a.smoke:
        write(out/'training_started.json',{'time':time.time(),'source_hashes':protocol['source_hashes']})
        write(out/'evaluation_indices.json',{'validation':evaluation,'fixed_eval_rng_base':1000,'inference_rng':777,'training_rng':42})
    mesh=sharding.make_mesh(1);state,_=core.init_train_state(cfg,jax.random.key(42),mesh,resume=False);jax.block_until_ready(state)
    before=fingerprint(state.params)
    if not a.smoke:
        write(out/'params_before.json',before)
        write(out/'parameter_counts.json',{'total':sum(r['elements'] for r in before.values()),
            'trainable':sum(r['elements'] for k,r in before.items() if 'lora' in k)})
    graph=state.model_def
    @jax.jit
    def evaluate(params,obs,actions,rng):
        m=nnx.merge(graph,params);m.eval();return jnp.mean(m.compute_loss(rng,obs,actions,train=False))
    @jax.jit
    def predict(params,obs):
        m=nnx.merge(graph,params);m.eval();return m.sample_actions(jax.random.key(777),obs,num_steps=10)
    eval_batches=[batch(val,r['index']) for r in evaluation] if not a.smoke else []
    def evaluate_fixed():
        values=[float(evaluate(state.params,o,ac,jax.random.key(1000+i))) for i,(o,ac) in enumerate(eval_batches)]
        assert np.isfinite(values).all()
        per={str(t):float(np.mean([v for v,r in zip(values,evaluation) if r['task_id']==t])) for t in tasks}
        return {'task_balanced_mean':float(np.mean(list(per.values()))),'per_task':per,'values':values}
    if not a.smoke:
        baseline=evaluate_fixed();write(out/'baseline_eval.json',baseline)
        manager,_=checkpoints.initialize_checkpoint_dir(out/'checkpoint',keep_period=SAVE_EVERY,overwrite=False,resume=False)
    step_fn=jax.jit(functools.partial(core.train_step,cfg),donate_argnums=(1,))
    rng=jax.random.key(42);sampler=np.random.default_rng(42);losses=[];best=None
    total=3 if a.smoke else STEPS
    with (out/('resource_smoke_metrics.jsonl' if a.smoke else 'metrics.jsonl')).open('x') as log:
        for step in range(total):
            tids=[tasks[(step*BATCH+j)%len(tasks)] for j in range(BATCH)];indices=[]
            for tid in tids:
                ep=by_task[tid][int(sampler.integers(len(by_task[tid])))];indices.append(ep['index_from']+int(sampler.integers(ep['length'])))
            obs,actions=multi_batch(train,indices);start=time.time()
            state,info=step_fn(rng,state,(obs,actions));jax.block_until_ready(state)
            row={'step':step+1,'source_frame_indices':indices,'task_ids':tids,'seconds':time.time()-start,**{k:float(v) for k,v in info.items()}}
            assert np.isfinite([row[k] for k in ('loss','grad_norm','param_norm')]).all(),row
            losses.append(row['loss']);log.write(json.dumps(row)+'\n');log.flush()
            if step==0 or (step+1)%100==0:print('STEP',step+1,'loss',row['loss'],'grad',row['grad_norm'],'seconds',row['seconds'],flush=True)
            if not a.smoke and (step+1)%SAVE_EVERY==0:
                ev=evaluate_fixed();write(out/f'eval_{step+1}.json',ev)
                after=fingerprint(state.params);changed=[k for k in before if before[k]['sha256']!=after[k]['sha256']]
                assert changed and all('lora' in k for k in changed),'non-LoRA drift or no learning'
                write(out/f'params_{step+1}.json',after)
                checkpoints.save_state(manager,state,Holder(dc),step+1);manager.wait_until_finished()
                pred=np.asarray(predict(state.params,eval_batches[0][0]));assert pred.shape==(1,10,32) and np.isfinite(pred).all()
                np.save(out/f'prediction_{step+1}.npy',pred)
                candidate={'step':step+1,'validation_task_balanced_loss':ev['task_balanced_mean']}
                if best is None or candidate['validation_task_balanced_loss']<best['validation_task_balanced_loss']:best=candidate
                print('CHECKPOINT',step+1,ev['task_balanced_mean'],flush=True)
    if a.smoke:
        write(out/result_name,{'pass':True,'steps':total,'batch_size':BATCH,'losses':losses,'gpu_memory_stats':jax.devices()[0].memory_stats(),'smoke_weights_saved':False})
    else:
        manager.close();assert int(state.step)==STEPS and best is not None
        write(out/'selected_checkpoint.json',best)
        write(out/result_name,{'pass':True,'steps':STEPS,'batch_size':BATCH,'tasks':tasks,'selected':best,
            'baseline_validation_loss':baseline['task_balanced_mean'],'last_window_mean_loss':float(np.mean(losses[-100:])),
            'all_loss_gradients_finite':True,'non_lora_parameters_unchanged':True,
            'gpu_memory_stats':jax.devices()[0].memory_stats(),'scope':'formal LoRA training/checkpoint selection; policy success still requires closed-loop evaluation'})
    return 0

if __name__=='__main__':raise SystemExit(main())
