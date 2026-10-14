import json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
ROOT=pathlib.Path('/app/agent-work-space/2-task-collection')
r=json.loads((ROOT/'runs/coverage_preflight_r1/result.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,6.3),layout='constrained')
colors={'train':'#0072b2','validation':'#d55e00','test':'#555555'}
markers={'train':'o','validation':'^','test':'x'}
for ax,obj,label in zip(axes,['alphabet_soup_1','basket_1'],['Target can','Receiving basket']):
 for split in colors:
  records=[x for x in r['records'] if x['scene']['split']==split];xy=np.array([x['settled_objects'][obj]['position'][:2] for x in records])*1000
  ax.scatter(xy[:,0],xy[:,1],c=colors[split],marker=markers[split],s=65,label=split+' (%d)'%len(records),zorder=3)
 grouped={}
 for rec in r['records']:
  pos=np.asarray(rec['settled_objects'][obj]['position'][:2])*1000
  name=rec['scene']['id'];split=rec['scene']['split'];short={'train':'T','validation':'V','test':'H'}[split]+str(int(name[-2:]))
  key=tuple(np.round(pos,3));grouped.setdefault(key,[]).append(short)
 for pos,names in grouped.items():ax.annotate('/'.join(names),pos,xytext=(4,5),textcoords='offset points',fontsize=8,color='#333333')
 ax.set(xlabel='World X (mm)',ylabel='World Y (mm)',title=label+' after 120 settling steps');ax.grid(alpha=.2);ax.set_aspect('equal',adjustable='box');ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.13),ncol=3)
axes[0].add_patch(Rectangle((-145,-265),50,50,fill=False,ls='--',lw=1,color='#999999'))
axes[0].set_xlim(-150,-90);axes[0].set_ylim(-270,-210)
axes[1].set_xlim(-11,11);axes[1].set_ylim(248,272)
fig.suptitle('Validated initialization coverage: 9 train / 3 validation / 3 reserved test\nNative assets and goal preserved; controlled initialization variant',fontsize=12)
out=ROOT/'reports/coverage_20261003_r1';fig.savefig(out/'initialization_coverage_r2.png',dpi=160);fig.savefig(out/'initialization_coverage_r2.pdf');plt.close(fig)
