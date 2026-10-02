#!/usr/bin/env python3
"""Seal local native-object sources, reports and evidence without touching N0."""
import ast,hashlib,io,json,pathlib,tarfile,datetime
P=pathlib.Path('/mnt/data/vla-policy/openpi/agent-work-space/zhixing_gripper_solve');out=P/'maintenance/real_objects_20261002_r1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):
 with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2)
lock=json.loads((out/'candidate_lock_r2.json').read_text());root=P.parent.parent
for path,digest in lock['files'].items():assert sha(root/path)==digest,path
source=set([P/'README.md'])
for pattern in ('真实碗罐*.md','scripts/*real*object*.py','scripts/*real*geometry*.py','scripts/probe_real_failure_r1.py'):
 source.update(P.glob(pattern))
source.update(P/'scripts'/f for f in ('safe_teacher_r8.py','loaded_diagnostics_r5.py','eco65_expert_validation.py'))
manifest={'version':'real_objects_20261002_r1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'teacher':'real_object_teacher_r10','files':{str(p.relative_to(P)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(source)},'upstream':'n0_20260930_r1 remains unchanged; separate verification preserved'}
dump(out/'source_manifest.json',manifest)
with tarfile.open(out/'source_snapshot.tar.gz','x:gz') as tf:
 for p in sorted(source):tf.add(p,arcname=str(p.relative_to(P)),recursive=False)
restored=[]
with tarfile.open(out/'source_snapshot.tar.gz') as tf:
 for name,info in manifest['files'].items():assert hashlib.sha256(tf.extractfile(name).read()).hexdigest()==info['sha256'];restored.append(name)
evidence=sorted(p for p in (P/'evidence').glob('real*') if p.is_file())
for d in sorted((P/'evidence').glob('real*')):
 if d.is_dir():evidence.extend(p for p in d.rglob('*') if p.is_file())
em={str(p.relative_to(P)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(evidence)}
dump(out/'evidence_manifest.json',{'files':em,'note':'Includes development failures, r9 first validation, r10 regression and fresh validation; no replacement or pooling.'})
funcs={}
for ver in (8,9,10):
 t=ast.parse((P/f'scripts/real_object_teacher_r{ver}.py').read_text());funcs[ver]={n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,ast.FunctionDef) and n.name in ('occupied_intervals','plan_grasp')}
assert funcs[8]==funcs[9]==funcs[10]
dump(out/'verification.json',{'source_manifest':'PASS','archive_restore':'PASS','source_count':len(source),'evidence_files':len(em),'evidence_bytes':sum(v['bytes'] for v in em.values()),'candidate_lock':'PASS','geometry_test_functions_identical_r8_r9_r10':True,'upstream':json.loads((out/'upstream_verification.json').read_text())['verdict'],'scope':'Source/evidence integrity only; task success is in the validation summary.'})
print('FREEZE PASS',len(source),'sources;',len(em),'evidence files')
