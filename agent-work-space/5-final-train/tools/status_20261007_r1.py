"""Read-only shutdown-recovery progress, complete attempt ledger and actual process liveness."""
import collections,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return None
def alive(pid,expected):
 try:return expected in pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
 except OSError:return False
if __name__=='__main__':
 active=read(ROOT/'reports/active_pipeline.json') or {'name':'pipeline_20261005_r1'}
 state=read(ROOT/'runs'/active['name']/'status.json')
 report={'active':active,'pipeline':state,'pipeline_process_alive':alive(active.get('pid'),active['name']),
  'collection_process_alive':alive(active.get('resume_collection_pid'),'validation_20261006_r1/scripts/batch.py'),'batches':{}}
 for name,folder in [('formal','formal_20261005_r1'),('supplement','supplement_20261006_r1')]:
  batch=ROOT/'runs'/folder;protocol=read(batch/'protocol.json');counts=collections.defaultdict(lambda:{'recorded':0,'accepted':0,'rejected':0,'interrupted':0,'train_accepted':0,'validation_accepted':0})
  for p in batch.glob('task*.summary.json'):
   r=read(p)
   if r is None:continue
   c=counts[str(r['scene']['task_id'])];c['recorded']+=1;c['accepted' if r['accepted'] else 'rejected']+=1;c['interrupted']+=int(r.get('interrupted',False))
   if r['accepted']:c[r['scene']['split']+'_accepted']+=1
  final=read(batch/'summary.json')
  report['batches'][name]={'registered_attempts':len(protocol['jobs']) if protocol else 0,'counts':dict(sorted(counts.items())),'release_pass':final.get('pass') if final else None}
 merged=read(ROOT/'runs/formal_merged_20261006_r1/summary.json')
 report['merged_release']={k:v for k,v in merged.items() if k!='records'} if merged else None
 train=ROOT/'runs'/active.get('training_run','spatial_lora_20261006_r1')
 report['training']={name:read(train/(name+'.json')) for name in ('resource_smoke_result','training_result','selected_checkpoint','reload_result')}
 export=ROOT/'lerobot_exports'/active.get('dataset_export','spatial_formal_r2')
 report['conversion']={'complete':read(export/'conversion.json'),'validation':read(export/'validation.json'),'datasets':{}}
 for p in export.glob('datasets/local/*/meta/info.json'):
  info=read(p)
  if info:report['conversion']['datasets'][p.parents[1].name]={'episodes':info['total_episodes'],'frames':info['total_frames'],'resume_prefix_verified':read(p.parents[1]/'resume_prefix.json') is not None}
 metrics=train/'metrics.jsonl'
 if metrics.exists():
  with metrics.open('rb') as f:f.seek(max(0,metrics.stat().st_size-8192));lines=f.read().decode(errors='replace').splitlines()
  for line in reversed(lines):
   try:report['last_training_step']=json.loads(line);break
   except ValueError:pass
 print(json.dumps(report,ensure_ascii=False,indent=2))
