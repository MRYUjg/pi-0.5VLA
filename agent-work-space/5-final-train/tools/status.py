"""Read-only progress summary; no Docker or ML dependencies required."""
import collections,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return None
if __name__=='__main__':
 pipeline=read(ROOT/'runs/pipeline_20261005_r1/status.json')
 report={'pipeline':pipeline,'batches':{}}
 for stage in ('regression','formal'):
  batch=ROOT/f'runs/{stage}_20261005_r1';protocol=read(batch/'protocol.json');summary=read(batch/'summary.json')
  counts=collections.defaultdict(lambda:{'completed':0,'passed':0,'rejected':0,'train_accepted':0,'validation_accepted':0})
  for p in batch.glob('task*.summary.json'):
   r=read(p)
   if r is None:continue
   c=counts[str(r['scene']['task_id'])];c['completed']+=1
   ok=r.get('development_pass') if stage=='regression' else r.get('accepted')
   c['passed' if ok else 'rejected']+=1
   if r.get('accepted') and r['scene']['split'] in ('train','validation'):c[r['scene']['split']+'_accepted']+=1
  report['batches'][stage]={'registered_attempts':len(protocol['jobs']) if protocol else 0,'counts':dict(counts),'final_summary':summary}
 train=ROOT/'runs/spatial_lora_20261005_r1'
 report['training']={name:read(train/(name+'.json')) for name in ('resource_smoke_result','training_result','selected_checkpoint','reload_result')}
 metrics=train/'metrics.jsonl'
 if metrics.exists():
  with metrics.open('rb') as f:
   f.seek(max(0,metrics.stat().st_size-8192));lines=f.read().decode(errors='replace').splitlines()
  for line in reversed(lines):
   try:report['last_training_step']=json.loads(line);break
   except ValueError:pass
 print(json.dumps(report,ensure_ascii=False,indent=2))
