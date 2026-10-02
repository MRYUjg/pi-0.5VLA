"""Analytic checks: distinguish hollow walls, handle rotated boxes and overlaps."""
import json,pathlib,sys,mujoco,numpy as np
from real_object_teacher_r8 import occupied_intervals
checks=[]
def run(name,xml,origin,axis,expected):
 m=mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+xml+'</worldbody></mujoco>');d=mujoco.MjData(m);mujoco.mj_forward(m,d)
 got=occupied_intervals(m,d,range(m.ngeom),np.array(origin),np.array(axis));actual=np.array([x[:2] for x in got]);want=np.array(expected)
 assert actual.shape==want.shape and np.allclose(actual,want,atol=1e-8),(name,got,expected)
 checks.append({'name':name,'pass':True,'intervals':got,'expected':expected})
run('hollow walls preserve cavity','<geom type="box" pos="-.04 0 0" size=".002 .02 .02"/><geom type="box" pos=".04 0 0" size=".002 .02 .02"/>',[0,0,0],[1,0,0],[[-.042,-.038],[.038,.042]])
run('rotated thin vertex-free slice','<geom type="box" euler="0 0 45" size=".01 .01 .02"/>',[0,0,.006],[1,0,0],[[-.01*np.sqrt(2),.01*np.sqrt(2)]])
run('overlap forms one occupied interval','<geom type="box" pos="0 0 0" size=".02 .02 .02"/><geom type="box" pos=".03 0 0" size=".02 .02 .02"/>',[0,0,0],[1,0,0],[[-.02,.05]])
run('ray misses elevated object','<geom type="box" pos="0 0 .10" size=".02 .02 .02"/>',[0,0,0],[1,0,0],[])
p=pathlib.Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True)
with p.open('x') as f:json.dump({'pass':True,'checks':checks},f,indent=2)
print('GEOMETRY TESTS PASS',len(checks))
