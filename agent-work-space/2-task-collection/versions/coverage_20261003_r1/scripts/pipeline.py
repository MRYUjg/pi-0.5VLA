"""Public entry: verify sealed inputs, collect a unique batch, replay and audit."""
import argparse,pathlib,subprocess,sys
from freeze import verify

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--batch',required=True);ap.add_argument('--scene-ids',default='');ap.add_argument('--workers',type=int,choices=[1,2],default=2);a=ap.parse_args()
    verify();scripts=pathlib.Path(__file__).resolve().parent
    code=subprocess.call([sys.executable,str(scripts/'run_coverage.py'),'--batch',a.batch,'--scene-ids',a.scene_ids,'--workers',str(a.workers)])
    if code:return code
    return subprocess.call([sys.executable,str(scripts/'validate.py'),'--batch',a.batch])
if __name__=='__main__':raise SystemExit(main())
