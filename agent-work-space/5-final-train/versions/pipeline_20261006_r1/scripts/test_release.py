import hashlib,importlib.util,json,pathlib,tempfile,contextlib,io
script=pathlib.Path(__file__).with_name('release.py')
spec=importlib.util.spec_from_file_location('release',script);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def fixture(root,extra_ok=True,duplicate=False):
 def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
 for name in ('formal_20261005_r1','supplement_20261006_r1'):
  jobs=[];records=[]
  if name.startswith('formal'):
   for t in [0,2,3,5,6]:
    for split,n,offset in [('train',50,620000),('validation',10,630000)]:
     jobs.extend({'task_id':t,'split':split,'seed':offset+t*100+i} for i in range(n))
  else:jobs=[{'task_id':2,'split':'validation','seed':650200+i} for i in range(10)]
  b=root/'runs'/name
  for s in jobs:
   seed=s['seed'];i=seed%100
   accepted=(i<20 if s['split']=='train' else i<(3 if s['task_id']==2 else 4)) if name.startswith('formal') else extra_ok and i==0
   ep=b/f'seed{seed}';rel=str(ep.relative_to(root));records.append(rel)
   row={'scene':s,'accepted':accepted,'interrupted':not accepted and seed%100==20}
   write(ep.with_suffix('.summary.json'),row)
   if accepted:
    ep.mkdir(exist_ok=True);(ep/'dummy').write_bytes(b'unchanged')
    audit={'accepted':True,'integrity_pass':True,'initialization_pass':True,'teacher_success':True,'physical_audit':{'audit_pass':True},'errors':[], 'files':{'dummy':hashlib.sha256(b'unchanged').hexdigest()},'settled_qpos_rounded8_sha256':'duplicated' if duplicate and s['task_id']==0 else str(seed)}
    write(ep/'audit.json',audit)
    for p in (ep.with_suffix('.status.json'),ep/'audit.status.json',ep/'replay.status.json'):write(p,{'exit_code':0})
  write(b/'protocol.json',{'jobs':jobs});write(b/'summary.json',{'records':records,'duplicate_initializations':[]})
for case,kw,want in [('admit_independent_extra',{},0),('reject_insufficient_validation',{'extra_ok':False},1),('reject_duplicate',{'duplicate':True},1)]:
 with tempfile.TemporaryDirectory() as tmp:
  root=pathlib.Path(tmp);fixture(root,**kw);m.ROOT=root
  with contextlib.redirect_stdout(io.StringIO()):code=m.main()
  assert code==want,(case,code)
  summary=json.loads((root/'runs/formal_merged_20261006_r1/summary.json').read_text());assert summary['attempts']==310 and summary['interrupted_attempts']==5
  print('PASS',case)
