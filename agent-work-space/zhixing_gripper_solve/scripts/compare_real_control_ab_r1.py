"""Check paired initial states and unchanged contracts before interpreting PD A/B."""
import argparse,json,pathlib,collections
ap=argparse.ArgumentParser();ap.add_argument('--batch',required=True);ap.add_argument('--audit',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();batch=pathlib.Path(a.batch);audit=json.loads(pathlib.Path(a.audit).read_text());groups=collections.defaultdict(dict)
for p in batch.glob('*/result.json'):
 r=json.loads(p.read_text());profile=r['runtime_profile']['extra_joint_pd_profile'];groups[(r['case'],r['seed'])][profile]=r
pairs=[]
for (case,seed),g in sorted(groups.items()):
 base=g['legacy'];candidate=g['eco65_no_extra_pd'];matched=base['initial_state_sha256']==candidate['initial_state_sha256'];contract=base['control_contract']==candidate['control_contract'];sources=base['sources']==candidate['sources'];profile_ok=base['runtime_profile']['extra_joint_pd_kp']==3 and candidate['runtime_profile']['extra_joint_pd_kp']==0
 pairs.append({'case':case,'seed':seed,'initial_state_matched':matched,'same_source_fingerprints':sources,'same_control_contract':contract,'expected_profile_values':profile_ok,'baseline_success':base['success'],'candidate_success':candidate['success'],'baseline_failure':base['failure'],'candidate_failure':candidate['failure'],'baseline_profile':base['runtime_profile'],'candidate_profile':candidate['runtime_profile']})
result={'version':'compare_real_control_ab_r1','pairs':pairs,'numeric_audit_pass':audit['pass'],'comparison_valid':audit['pass'] and all(p['initial_state_matched'] and p['same_source_fingerprints'] and p['same_control_contract'] and p['expected_profile_values'] for p in pairs),'baseline_success':sum(p['baseline_success'] for p in pairs),'candidate_success':sum(p['candidate_success'] for p in pairs),'total_pairs':len(pairs),'scope':'Paired evidence for these cases only; not proof that PD explains every failure or changes production defaults.'}
result['eligible_for_fresh_validation']=result['comparison_valid'] and all(p['candidate_success'] for p in pairs)
with pathlib.Path(a.out).open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False,indent=2))
