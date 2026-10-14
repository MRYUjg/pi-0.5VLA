"""Read-only evidence that calibration stayed outside all registered formal scenes."""
import collections
import json
import pathlib
from common import ROOT,sha,write_json

run=ROOT/'runs/trt_t0_t2_20261007_r1'
dataset=run/'calibration_r1/dataset.json'
data=json.loads(dataset.read_text())['observations']
ledger=pathlib.Path('/app/agent-work-space/5-final-train/versions/collection_20261006_r1/configs/test_seeds.json')
reserved=json.loads(ledger.read_text())['tasks']
formal={(int(t),s) for t,ss in reserved.items() for s in ss}
calibration={(r['task'],r['seed']) for r in data}
assert not calibration&formal
assert len({r['image_sha256'] for r in data})==2000
counts=collections.Counter(r['task'] for r in data)
assert set(counts.values())=={200}
phases=collections.Counter((r['task'],r['phase']) for r in data)
for task in range(10):
    registration=json.loads((run/f'panda_calibration_r1/task{task:02d}/registration.json').read_text())
    indices={r['index'] for r in data if r['task']==task}
    assert indices.isdisjoint(registration['reserved_formal_indices'])
    assert indices.issubset(registration['developer_indices'])
    assert all(phases[(task,p)]>=60 for p in ['early','middle','late'])
for r in data:assert sha(r['path'])==r['file_sha256']
write_json(run/'calibration_audit_r1.json',{'passed':True,'observations':2000,
    'unique_image_pairs':2000,'per_task':dict(counts),
    'phases':{f'{t}_{p}':v for (t,p),v in phases.items()},
    'ECO65_formal_overlap':0,'Panda_formal_overlap':0,'ECO65_formal_ledger_sha256':sha(ledger),
    'dataset_sha256':sha(dataset),'frames_are_temporally_correlated':True,'independent_trial_count_claimed':False})
print('CALIBRATION AUDIT PASS')
