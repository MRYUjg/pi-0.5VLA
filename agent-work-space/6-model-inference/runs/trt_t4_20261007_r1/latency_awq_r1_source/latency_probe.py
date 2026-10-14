"""One registered scene, one client; excluded from the 30-trial quality score."""
import argparse
import json
import pathlib

from closedloop import episode


def main(out,scenes,variant,port):
    out.mkdir(parents=True,exist_ok=False)
    rows=json.loads(scenes.read_text())
    row=next(r for r in rows if r['task']==0 and r['init_index']==0)
    reference=json.loads((scenes.parent/'controller.json').read_text());reference.pop('XML_sha256')
    result=episode(out,row,scenes.parent,variant,port,reference)
    with (out/'summary.json').open('x') as f:json.dump({'episode':result,'single_simulation_client':True,
        'duplicate_scene_excluded_from_quality_denominator':True,'purpose':'single-client full-camera pipeline latency'},f,indent=2)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True)
    p.add_argument('--scenes',type=pathlib.Path,required=True);p.add_argument('--variant',required=True)
    p.add_argument('--port',type=int,default=18879);a=p.parse_args();main(a.out,a.scenes,a.variant,a.port)
