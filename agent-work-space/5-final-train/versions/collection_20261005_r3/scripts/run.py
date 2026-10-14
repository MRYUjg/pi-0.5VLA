"""Host-only isolated collection launcher; stdout/logs retained by the supervisor."""
import argparse
import os
import pathlib
import subprocess

VERSION=pathlib.Path(__file__).resolve().parents[1]
ROOT=VERSION.parents[1]
REPO=ROOT.parents[1]

def command(script,args):
    rel=lambda p:'/app/'+str(p.relative_to(REPO))
    runtime=REPO/'agent-work-space/1.3-zhixing_gripper_solve/runtime/py310_mj326/bin/python'
    return ['docker','run','--rm','--network','none','--cpus','2','--memory','4g',
        '-e','EXTRA_JOINT_PD_PROFILE=eco65_no_extra_pd','-e','EXTRA_JOINT_PD_KP=',
        '-e','MUJOCO_GL=osmesa','-e','PYTHONDONTWRITEBYTECODE=1',
        '-e','OPENBLAS_NUM_THREADS=1','-e','OMP_NUM_THREADS=1','-e','LP_NUM_THREADS=1',
        '-e','PYTHONPATH=/app:/app/packages/openpi-client/src:/app/third_party/libero',
        '-v',str(REPO)+':/app:ro','-v',str(REPO)+':'+str(REPO)+':ro',
        '-v',str(ROOT)+':'+rel(ROOT),'-w','/app','--entrypoint',rel(runtime),
        'libero:latest',rel(VERSION/'scripts'/script)]+list(args)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('script');p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args()
    assert pathlib.Path(a.script).name==a.script and a.script.endswith('.py')
    raise SystemExit(subprocess.run(command(a.script,a.args)).returncode)
