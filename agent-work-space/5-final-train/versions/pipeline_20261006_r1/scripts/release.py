"""Merge complete attempt ledgers without replacing failures or admission criteria."""
import collections,hashlib,json,pathlib
V=pathlib.Path(__file__).resolve().parents[1];ROOT=V.parents[1]
read=lambda p:json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=ROOT/'runs/formal_merged_20261006_r1';out.mkdir(parents=True,exist_ok=False)
 protocols=[];records=[];rows=[];seen=set();hashes={};dups=[]
 for name,n in [('formal_20261005_r1',300),('supplement_20261006_r1',10)]:
  b=ROOT/'runs'/name;protocol=read(b/'protocol.json');summary=read(b/'summary.json')
  assert len(protocol['jobs'])==len(summary['records'])==n,'incomplete attempt ledger'
  assert not summary['duplicate_initializations']
  registered={s['seed']:s for s in protocol['jobs']};assert len(registered)==n
  protocols.extend(protocol['jobs'])
  for rel in summary['records']:
   ep=ROOT/rel;r=read(ep.with_suffix('.summary.json'));s=r['scene']
   assert s==registered[s['seed']] and s['seed'] not in seen;seen.add(s['seed']);rows.append(r);records.append(rel)
   if r['accepted']:
    audit=read(ep/'audit.json');assert audit['accepted'] and audit['integrity_pass'] and audit['initialization_pass'] and audit['teacher_success'] and audit['physical_audit']['audit_pass'] and not audit['errors']
    for filename,want in audit['files'].items():assert sha(ep/filename)==want
    assert read(ep.with_suffix('.status.json'))['exit_code']==read(ep/'audit.status.json')['exit_code']==read(ep/'replay.status.json')['exit_code']==0
    key=(s['task_id'],audit['settled_qpos_rounded8_sha256'])
    if key in hashes:dups.append([hashes[key],s['seed']])
    hashes[key]=s['seed']
 counts={}
 for tid in [0,2,3,5,6]:
  counts[str(tid)]={split:{'attempted':sum(r['scene']['task_id']==tid and r['scene']['split']==split for r in rows),'accepted':sum(r['scene']['task_id']==tid and r['scene']['split']==split and r['accepted'] for r in rows)} for split in ('train','validation')}
 passed=not dups and all(v['train']['accepted']>=20 and v['validation']['accepted']>=4 for v in counts.values())
 summary={'pass':passed,'stage':'formal_merged','eligible_tasks':[0,2,3,5,6],'excluded_tasks':[8],'attempts':len(rows),'counts':counts,'duplicate_initializations':dups,'admission_threshold':'at least 20 train and 4 validation admitted per task; unchanged','records':records,'original_attempts':300,'supplemental_validation_attempts':10,'interrupted_attempts':sum(r.get('interrupted',False) for r in rows),'source_batches':['formal_20261005_r1','supplement_20261006_r1']}
 (out/'protocol.json').write_text(json.dumps({'jobs':protocols,'retries':0,'scope':'original and independent preregistered supplemental attempts; all failures retained'},indent=2))
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True);return int(not passed)
if __name__=='__main__':raise SystemExit(main())
