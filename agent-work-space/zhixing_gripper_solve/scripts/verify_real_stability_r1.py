"""Read-only supplemental check of the teacher's claimed stage dwell, no gate changes."""
import argparse, json, pathlib
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('--batch', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
records = []
for p in sorted(pathlib.Path(a.batch).glob('*/result.json')):
    r = json.loads(p.read_text())
    if r['case'] != 'bowl':
        continue
    rows = [json.loads(s) for s in p.with_name('steps.jsonl').read_text().splitlines()]
    stages = {}
    for phase, arrival in r['plan'].get('stage_arrivals', {}).items():
        if not arrival['ok']:
            continue
        tail = [x for x in rows if x['phase'] == phase][-5:]
        checks = []
        for x in tail:
            pos = float(np.linalg.norm(np.array(x['geometry']['tcp']) - arrival['target']))
            angle = float(np.arccos(np.clip((np.trace(np.array(arrival['target_rotation']).T @ np.array(x['geometry']['R'])) - 1) / 2, -1, 1)))
            displacement = float(np.linalg.norm(np.array(x['after']['grip_site']) - x['before']['grip_site']))
            checks.append({'position_error_m': pos, 'angle_rad': angle, 'displacement_m': displacement,
                           'satisfies_claim': pos < arrival['position_tolerance_m'] and angle < .03 and displacement < .0005})
        stages[phase] = {'samples': checks, 'claim_verified': len(tail) == 5 and all(x['satisfies_claim'] for x in checks)}
    records.append({'case': r['case'], 'seed': r['seed'], 'task_success': r['success'], 'stages': stages,
                    'claims_verified': all(s['claim_verified'] for s in stages.values())})
result = {'version': 'verify_real_stability_r1', 'records': records,
          'all_reported_arrival_claims_verified': bool(records) and all(x['claims_verified'] for x in records),
          'scope': 'Supplemental read-only check of teacher dwell claims. Original task acceptance gates unchanged.'}
with pathlib.Path(a.out).open('x') as f:
    json.dump(result, f, indent=2)
print('STABILITY CLAIMS', result['all_reported_arrival_claims_verified'])
raise SystemExit(0 if result['all_reported_arrival_claims_verified'] else 1)
