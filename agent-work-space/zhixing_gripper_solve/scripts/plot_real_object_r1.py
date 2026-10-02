"""Export measured geometry/force plots from numeric evidence (no synthetic images)."""
import argparse,json,pathlib,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('--batch',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();batch=pathlib.Path(a.batch);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=False)
colors=plt.get_cmap('tab20').colors
for case in ('can','bowl'):
 paths=sorted(batch.glob(case+'_*/result.json'));p=next(p for p in paths if json.loads(p.read_text())['success']);r=json.loads(p.read_text());rows=[json.loads(s) for s in p.with_name('steps.jsonl').read_text().splitlines()];times=np.array([x['after']['time'] for x in rows]);times-=times[0]
 tcp=np.array([x['geometry']['tcp'] for x in rows]);obj=np.array([np.mean(x['geometry']['object'],axis=0) for x in rows]);bottom=np.array([x['geometry']['object'][0][2] for x in rows]);gripper=np.array([x['geometry']['gripper_outer'][0][2] for x in rows]);force=np.array([x['normal_force_mean_N'] for x in rows]);phases=list(dict.fromkeys(x['phase'] for x in rows))
 fig,axs=plt.subplots(2,2,figsize=(13,8));fig.suptitle(f'Native {case} | seed {r["seed"]} | measured teacher execution')
 for i,ph in enumerate(phases):
  idx=np.array([x['phase']==ph for x in rows]);axs[0,0].plot(tcp[idx,0],tcp[idx,1],color=colors[i],label=ph,linewidth=2)
 axs[0,0].plot(obj[:,0],obj[:,1],color='black',alpha=.4,linestyle=':',label='payload center')
 for box,label,col in [(r['initial_geometry']['object'],'initial object','black'),(r['initial_geometry']['basket'],'recipient','tab:green')]:
  lo,hi=np.array(box);axs[0,0].add_patch(plt.Rectangle(lo[:2],hi[0]-lo[0],hi[1]-lo[1],fill=False,color=col));axs[0,0].text(lo[0],hi[1],label,fontsize=8)
 axs[0,0].set(xlabel='World X (m)',ylabel='World Y (m)',title='Actual TCP / payload XY path');axs[0,0].axis('equal');axs[0,0].legend(fontsize=6,ncol=2)
 axs[0,1].plot(times,tcp[:,2],label='TCP');axs[0,1].plot(times,bottom,label='payload minimum Z');axs[0,1].plot(times,gripper,label='full gripper minimum Z');axs[0,1].axhline(r['initial_geometry']['basket'][1][2],color='r',linestyle='--',label='recipient top');axs[0,1].set(xlabel='Time (s)',ylabel='World Z (m)',title='Measured lower envelopes');axs[0,1].legend(fontsize=7)
 axs[1,0].plot(times,force[:,0],label='left');axs[1,0].plot(times,force[:,1],label='right');axs[1,0].set(xlabel='Time (s)',ylabel='Normal force (N)',title='Finger / object contact force');axs[1,0].legend()
 axs[1,1].plot(times,[x['bilateral_fraction'] for x in rows],label='bilateral fraction');axs[1,1].plot(times,[x['after']['native_success'] for x in rows],label='native goal');axs[1,1].set(xlabel='Time (s)',ylabel='Fraction / predicate',title='Contact and native task outcome');axs[1,1].legend()
 for i,ph in enumerate(phases):
  ids=[j for j,x in enumerate(rows) if x['phase']==ph]
  for ax in (axs[0,1],axs[1,0],axs[1,1]):ax.axvspan(times[ids[0]],times[ids[-1]],color=colors[i],alpha=.08)
 for ax in axs.flat:ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(out/(case+'.png'),dpi=170);fig.savefig(out/(case+'.pdf'));plt.close(fig)
print('PLOTS',out)
