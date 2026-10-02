"""Standalone evidence plot: same diagnostic seed, measured old/new trajectories."""
import pathlib,json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=pathlib.Path('/app/agent-work-space/zhixing_gripper_solve')
out=root/'evidence/safe_path_figure_20261002_r1';out.mkdir(exist_ok=False)
old=root/'evidence/loaded_matrix_20261001_r3/ZhixingGripper_chain_s47100_f0'
new=root/'evidence/safe_path_20261002_r8/ZhixingGripper_chain_s47100_f0'
def read(p):return [json.loads(s) for s in (p/'steps.jsonl').read_text().splitlines()]
a,b=read(old),read(new);plan=json.loads((new/'safe_path.json').read_text())['plan'];rim=plan['measured_loaded_geometry']['basket'][1][2]
fig,ax=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
for rows,color,label in [(a,'#b44b42','Previous teacher'),(b,'#237a65','Geometry-aware teacher')]:
 rows=[r for r in rows if r['phase'] in ('hold','clearance_lift','transport','place','release','post_release_window')]
 pos=np.array([r['after']['object_pos'] for r in rows]);rot=np.array([r['after']['object_rotation'] for r in rows])
 bottom=pos[:,2]-np.abs(rot[:,2,:])@np.array([.03,.015,.035])
 ax[0].plot(pos[:,1]*1000,bottom*1000,color=color,label=label,lw=2)
 transport=[r for r in rows if r['phase']=='transport'][0]['before']['time'];t=np.array([r['after']['time']-transport for r in rows])
 # Basket freejoint from immutable model names (recorded model joint addresses).
 model=json.loads(((old if label.startswith('Previous') else new)/'result.json').read_text())
 j=model['joint_names'].index('basket_1_joint0');q=int(model['joint_qposadr'][j]);origin=np.array(rows[0]['before']['qpos'][q:q+3])
 drift=np.array([np.linalg.norm(np.array(r['after']['qpos'][q:q+3])-origin)*1000 for r in rows])
 ax[1].plot(t,drift,color=color,lw=2,label=label)
lo,hi=np.array(plan['measured_loaded_geometry']['basket']);ax[0].plot([lo[1]*1000,hi[1]*1000],[rim*1000,rim*1000],'k--',label='Basket rim height')
ax[0].axvspan(lo[1]*1000,hi[1]*1000,color='grey',alpha=.08)
ax[0].set(xlabel='Object center Y (mm)',ylabel='Object lowest point Z (mm)',title='Measured load path, seed 47100')
ax[1].axvline(0,color='grey',ls=':',lw=1);ax[1].set(xlabel='Time from horizontal transport start (s)',ylabel='Basket translation (mm)',title='Basket movement during carry and release')
for panel in ax:panel.grid(alpha=.2);panel.legend(fontsize=8)
fig.savefig(out/'path_comparison.png',dpi=180);fig.savefig(out/'path_comparison.pdf');plt.close(fig)
(out/'provenance.json').write_text(json.dumps({'old':str(old),'new':str(new),'seed':47100,'object_half_extents_m':[.03,.015,.035],'note':'YZ projection of measured box lowest point, not rendered simulation. Old/new use same initial scene.'},indent=2))
