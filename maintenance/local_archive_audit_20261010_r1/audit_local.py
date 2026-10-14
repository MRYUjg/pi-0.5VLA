"""Read-only audit of local project mirrors against preserved server manifests."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import os
import tarfile

ROOT = Path('/Users/yuyu/Documents/openpi')
OUT = Path(__file__).resolve().parent
SKIP = {'.git', 'node_modules', '__pycache__', '.venv'}
CACHE = {}

def digest(path):
    path = Path(path)
    if path not in CACHE:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        CACHE[path] = h.hexdigest()
    return CACHE[path]

def walk(base):
    for parent, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in SKIP and not (Path(parent)/d).is_symlink()]
        for name in files:
            p = Path(parent)/name
            if not p.is_symlink() and p.is_file():
                yield p

def verify(entries, base, alternatives=False):
    rows=[]
    for name, spec in entries.items():
        expected = spec.get('sha256') if isinstance(spec,dict) else spec
        if not isinstance(expected,str) or len(expected)!=64:
            continue
        p=base/name
        row={'path':name,'expected_sha256':expected}
        if not p.is_file():
            row['status']='missing'
        else:
            actual=digest(p)
            row.update(actual_sha256=actual,status='exact' if actual==expected else 'different')
            if actual!=expected and p.suffix in {'.py','.xml','.md','.json','.sh','.txt','.yaml','.yml','.toml','.bddl','.cfg','.ini'}:
                normalized=hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                row['matches_after_CRLF_to_LF']=normalized==expected
                if normalized==expected:row['status']='line_endings_only'
        if alternatives and row['status'] in {'missing','different'}:
            alt=ROOT/'.publish/pi-0.5VLA'/name
            if alt.is_file():
                actual=digest(alt)
                normalized=hashlib.sha256(alt.read_bytes().replace(b'\r\n',b'\n')).hexdigest() if alt.suffix in {'.py','.xml','.md','.json','.sh','.txt','.yaml','.yml','.toml','.bddl','.cfg','.ini'} else actual
                if actual==expected or normalized==expected:
                    row['alternate_copy']=str(alt.relative_to(ROOT))
                    row['alternate_status']='exact' if actual==expected else 'line_endings_only'
        rows.append(row)
    return {'counts':dict(Counter(x['status'] for x in rows)),
            'alternate_matches':sum('alternate_copy' in x for x in rows),'entries':rows}

inventory=[]
for name in ['agent-work-space','gpt-rpent-control','robosuite','third_party','src','examples','scripts','packages','my-robot','ECO_data','libreo_raw_data','.publish','.n3work']:
    base=ROOT/name
    files=[]
    for p in walk(base):
        st=p.stat()
        files.append({'path':str(p.relative_to(ROOT)),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns})
    inventory.append({'root':name,'files':len(files),'bytes':sum(x['bytes'] for x in files),'entries':files})

checks={}
envpath=ROOT/'agent-work-space/5-final-train/versions/rollout_20261008_r1/environment_manifest.json'
env=json.loads(envpath.read_text())
checks['latest_simulation_environment']=verify(env['files'],ROOT,True)
critical_names=['robosuite/models/grippers/zhixing_gripper.py','robosuite/robots/single_arm.py',
    'robosuite/models/robots/robot_model.py','robosuite/models/assets/robots/ECO65-B/robot.xml',
    'third_party/libero/libero/libero/envs/robots/mounted_eco65.py','src/openpi/training/config.py']
checks['critical_simulator_files']=[x for x in checks['latest_simulation_environment']['entries'] if x['path'] in critical_names]

fifth=ROOT/'agent-work-space/5-final-train'
for mf in sorted((fifth/'releases').glob('*/manifest.json')):
    data=json.loads(mf.read_text())
    if isinstance(data.get('files'),dict):
        checks['stage5_'+mf.parent.name]=verify(data['files'],fifth)

fourth=ROOT/'agent-work-space/4-multi-task-collection'
mf=fourth/'archive/final_20261005_r1/manifest.json'
checks['stage4_final_archive']=verify(json.loads(mf.read_text())['files'],fourth)

second=ROOT/'agent-work-space/2-task-collection'
for mf in sorted((second/'releases').glob('*/manifest.json')):
    data=json.loads(mf.read_text())
    entries=data.get('files',{})
    if isinstance(entries,dict):
        own={k:v for k,v in entries.items() if k.startswith('agent-work-space/2-task-collection/versions/')}
        checks['stage2_source_'+mf.parent.name]=verify(own,ROOT)

sixth=ROOT/'agent-work-space/6-model-inference'
for v in ['a100_closedloop_20261008_r1','a100_comparison_20261008_r1','trt_t4_20261007_r1']:
    mf=sixth/'releases'/v/'manifest.json'
    if not mf.exists():continue
    data=json.loads(mf.read_text())
    entries=data.get('files_relative_to_6_model_inference',data.get('files',{}))
    checks['stage6_'+v]=verify(entries,sixth)
for mf in sorted((sixth/'weights').glob('*/bundle_manifest.json')):
    checks['weight_bundle_'+mf.parent.name]=verify(json.loads(mf.read_text())['files'],mf.parent)

gpt=ROOT/'gpt-rpent-control'
lock=gpt/'reports/eco65_zhixing_deploy_20261007_r1/candidate_lock_r5.json'
checks['rpent_adapter_r5']=verify(json.loads(lock.read_text()),gpt)
for mf in sorted((gpt/'releases').glob('*/manifest.json')):
    data=json.loads(mf.read_text())
    checks['rpent_release_'+mf.parent.name]=verify(data['files_relative_to_gpt_rpent_control'],gpt)

summary=ROOT/'agent-work-space/project-summary-evidence/summary_20261008_r1/summary_manifest.json'
checks['project_summary']=verify(json.loads(summary.read_text())['files'],ROOT/'agent-work-space')

archives=[]
for base in [second,ROOT/'agent-work-space/3-lora-validation',ROOT/'agent-work-space/zhixing_gripper_solve']:
    for p in base.rglob('*.tar.gz'):
        with tarfile.open(p,'r:gz') as tf:
            members=[x.name for x in tf.getmembers() if x.isfile()]
        archives.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'member_count':len(members),
                         'contains_model_or_dataset_payload':any(x.endswith(('.npz','.hdf5','.parquet','.safetensors','.pt')) for x in members)})

dataset_manifest=json.loads((fifth/'lerobot_exports/spatial_formal_r3/dataset_files_sha256.json').read_text())
checkpoints=[]
for p in sorted((fifth/'runs/spatial_lora_20261007_r1/checkpoint').iterdir()):
    checkpoints.append({'checkpoint':p.name,'params_exists':(p/'params').exists(),'train_state_exists':(p/'train_state').exists(),
                        'files':[str(x.relative_to(ROOT)) for x in walk(p)]})
gaps={'scope':'Inspected workspace, its .publish mirror, and usual OpenPI/HuggingFace cache locations; no remote recovery attempted.',
      'formal_dataset_manifest_entries':len(dataset_manifest),
      'formal_dataset_parquet_expected':sum(k.endswith('.parquet') for k in dataset_manifest),
      'formal_dataset_parquet_found_in_stage5':len(list(fifth.rglob('*.parquet'))),
      'formal_original_trajectory_npz_found_in_stage5':len(list(fifth.rglob('*.npz'))),
      'checkpoints':checkpoints,
      'full_RPent_repository_found_in_workspace':False,
      'model_file_suffixes_found_in_workspace':{ext:sum(p.suffix==ext for p in walk(ROOT)) for ext in ['.safetensors','.pt','.pth','.ckpt','.engine','.plan']},
      'usual_caches':{str(p):p.exists() for p in [Path('/Users/yuyu/.cache/openpi'),Path('/Users/yuyu/.cache/huggingface')]}}

counts={}
for base in [ROOT/'agent-work-space',gpt]:
    files=list(walk(base))
    counts[str(base.relative_to(ROOT))]={'files':len(files),'bytes':sum(p.stat().st_size for p in files),
         'python_files':sum(p.suffix=='.py' for p in files),'mp4_files':sum(p.suffix=='.mp4' for p in files)}

result={'audit_date_beijing':'2026-10-10','workspace':str(ROOT),'claim':'Local mirrors exist but are not a complete server backup.',
        'inventory':inventory,'manifest_checks':checks,'source_archives':archives,'known_gaps':gaps,'counts':counts}
target=OUT/'inventory_and_verification.json'
with target.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({'report':str(target),'counts':counts,
   'checks':{k:{x:y for x,y in v.items() if x!='entries'} for k,v in checks.items() if isinstance(v,dict)},
   'critical_simulator_files':checks['critical_simulator_files'],'known_gaps':gaps,'source_archives':archives},ensure_ascii=False,indent=2))
