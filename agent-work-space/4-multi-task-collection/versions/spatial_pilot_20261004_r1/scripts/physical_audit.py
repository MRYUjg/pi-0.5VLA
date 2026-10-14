"""Run unchanged r6 physical evidence audit; does not grant dataset acceptance."""
import sys,pathlib,json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'vendor'))
import verify_real_object_r6 as audit
p=pathlib.Path(sys.argv[1])
try:r=audit.audit(p/'teacher')
except Exception as e:r={'audit_pass':False,'success':False,'errors':[type(e).__name__+': '+str(e)]}
with (p/'physical_audit.json').open('x') as f:json.dump(r,f,indent=2,default=lambda x:x.item())
print(json.dumps({k:r.get(k) for k in ('audit_pass','success','errors','failure')}),flush=True)
raise SystemExit(int(not r['audit_pass']))
