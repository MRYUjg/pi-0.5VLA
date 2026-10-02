"""Plot exact native collision-box cross-sections and selected grasp ray."""
import json,pathlib,sys,itertools,numpy as np,mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import real_object_teacher_r15 as T
batch=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=False)
fig,axs=plt.subplots(1,2,figsize=(12,5));signs=np.array(list(itertools.product((-1,1),repeat=3)))
for ax,case in zip(axs,('can','bowl')):
 p=next(p for p in [p for p in sorted(batch.glob('*/result.json')) if json.loads(p.read_text())['case']==case] if json.loads(p.read_text())['success']);r=json.loads(p.read_text());plan=r['plan']['grasp_selected'];env,_=T.D.build(r['suite'],0,seed=r['seed']);env.reset();sim,m,d,gid=T.N.ids(env);d.qpos[:]=r['initial_qpos'];mujoco.mj_forward(m,d);geo=T.S.geometry(env);z=plan['pin'][2];origin=np.mean(geo['object'],axis=0)
 for g in geo['object_geoms']:
  assert m.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX
  verts=(signs*m.geom_size[g])@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];cross=[]
  for i in range(8):
   for j in range(i+1,8):
    if np.count_nonzero(signs[i]!=signs[j])!=1:continue
    za,zb=verts[i,2],verts[j,2]
    if abs(zb-za)>1e-12 and min(za,zb)<=z<=max(za,zb):cross.append(verts[i]+(verts[j]-verts[i])*((z-za)/(zb-za)))
  if len(cross)<3:continue
  xy=(np.array(cross)[:,:2]-origin[:2])*1000
  hull=ConvexHull(xy);ax.fill(xy[hull.vertices,0],xy[hull.vertices,1],color='steelblue',alpha=.5,linewidth=.3,edgecolor='navy')
 R=np.array(geo['R']);Rc=np.array(plan['R']);closing=Rc@(R.T@T.EX.closing_axis_world_generic(env));pin=np.array(plan['pin']);ends=pin+np.array([-.5,.5])[:,None]*plan['opening_m']*closing
 xy=(ends[:,:2]-origin[:2])*1000;ax.plot(xy[:,0],xy[:,1],'k--',label='open jaw span');xy=(pin[:2]-origin[:2])*1000;ax.plot(xy[0],xy[1],'ro',label='grasp point')
 ax.set(title=f'{case}: native collision slice\nmaterial width {plan["material_width_m"]*1000:.2f} mm; open span {plan["opening_m"]*1000:.2f} mm',xlabel='X relative to object center (mm)',ylabel='Y relative to object center (mm)');ax.axis('equal');ax.grid(alpha=.2);ax.legend(fontsize=8);env.close()
fig.suptitle('Measured collision geometry at planned grasp height (original assets)');fig.tight_layout();fig.savefig(out/'native_geometry.png',dpi=180);fig.savefig(out/'native_geometry.pdf');print('GEOMETRY PLOT',out)
