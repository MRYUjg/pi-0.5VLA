"""Host replay + audit launcher, preserves failures and real process exits."""
import argparse,json,pathlib,subprocess,time
from run import REPO,COLLECTION,container_command

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--batch',required=True);a=ap.parse_args();batch=COLLECTION/'runs'/a.batch
    p=json.loads((batch/'protocol.json').read_text())
    for seed in p['seeds']:
        ep=batch/('can_s'+str(seed));s=json.loads(ep.with_suffix('.status.json').read_text())
        # r6 audit expects the teacher's enclosing process status next to its evidence directory.
        with (ep/'teacher.status.json').open('x') as f:json.dump({**s,'job':{'profile':'eco65_no_extra_pd','seed':seed,'case':'can'},'source_status':str(ep.with_suffix('.status.json'))},f,indent=2)
        rel='/app/'+str(ep.relative_to(REPO))
        cmd=container_command('replay.py',['--episode',rel,'--out',rel+'/replay.json'])
        start=time.time()
        with (ep/'replay.log').open('x') as f:
            try:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900).returncode
            except subprocess.TimeoutExpired:code=-999
        with (ep/'replay.status.json').open('x') as f:json.dump({'exit_code':code,'wall_seconds':time.time()-start,'command':cmd},f,indent=2)
        print(seed,'replay exit',code,flush=True)
    cmd=container_command('audit.py',['--batch','/app/'+str(batch.relative_to(REPO))])
    # Frozen r6 auditor resolves /app source paths through this host-root path.
    cmd[2:2]=['-v',str(REPO)+':'+str(REPO)+':ro']
    with (batch/'audit.log').open('x') as f:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900).returncode
    with (batch/'audit.status.json').open('x') as f:json.dump({'exit_code':code,'command':cmd},f,indent=2)
    print('AUDIT EXIT',code,flush=True);return code
if __name__=='__main__':raise SystemExit(main())
