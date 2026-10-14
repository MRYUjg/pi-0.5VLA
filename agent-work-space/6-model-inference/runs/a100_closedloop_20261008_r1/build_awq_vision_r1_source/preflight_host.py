"""CPU simulation fingerprint registration; compare to frozen 4090 protocol."""
import hashlib,json,pathlib,shutil,subprocess,time
REPO=pathlib.Path('/mnt/data/vla-policy/openpi');ROOT=REPO/'agent-work-space/6-model-inference'
V='a100_closedloop_20261008_r1';RUN=ROOT/'runs'/V
folder=RUN/'preflight_r1_source';folder.mkdir(exist_ok=False)
for p in (ROOT/'versions'/V).iterdir():
    if p.is_file():shutil.copy2(p,folder/p.name);(folder/p.name).chmod(0o444)
source='/work/runs/'+V+'/'+folder.name
cmd=['docker','run','--rm','--name','pi05-a100-preflight-20261008-r1','--cpus','2','--memory','3g',
    '-e','PYTHONPATH=/app:/app/packages/openpi-client/src:/app/third_party/libero',
    '-e','PYTHONDONTWRITEBYTECODE=1','-e','MUJOCO_GL=osmesa','-e','LP_NUM_THREADS=1',
    '-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','EXTRA_JOINT_PD_PROFILE=legacy',
    '-v',str(REPO)+':/app:ro','-v',str(ROOT)+':/work:ro','-v',str(RUN)+':/newrun',
    '-w','/app','--entrypoint','/app/agent-work-space/1.3-zhixing_gripper_solve/runtime/py310_mj326/bin/python',
    'sha256:1ac69906b0b877c9fe1c2964794aa5b90112850da01967a445658e41a958f284',
    source+'/preflight.py','--out','/newrun/preflight_r1']
start=time.time()
with (RUN/'preflight_r1.log').open('x') as f:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
with (RUN/'preflight_r1_exit.json').open('x') as f:json.dump({'command':cmd,'exit_code':code,'seconds':time.time()-start},f,indent=2)
assert code==0,code
old=ROOT/'runs/trt_t4_20261007_r1/preflight_r1';new=RUN/'preflight_r1'
for name in ['scenes.json','controller.json']:
    assert json.loads((old/name).read_text())==json.loads((new/name).read_text()),name
a=json.loads((old/'environment.json').read_text());b=json.loads((new/'environment.json').read_text())
assert a==b,'simulation source/packages changed'
with (RUN/'simulation_alignment_r1.json').open('x') as f:json.dump({'passed':True,
    'identical_scene_registration_controller_and_simulation_environment':True,
    'reference':'runs/trt_t4_20261007_r1/preflight_r1','new_scene_indices':[0,1,2],
    'scene_sha256':hashlib.sha256((new/'scenes.json').read_bytes()).hexdigest()},f,indent=2)
print('A100 CPU SIMULATION REGISTRATION ALIGNED WITH FROZEN T4',flush=True)
