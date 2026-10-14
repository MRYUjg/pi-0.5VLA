"""Standalone plot from recorded metrics; does not read checkpoint or touch training."""
import json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[2]
run=ROOT/'runs/smoke_train_r2';out=ROOT/'reports'
rows=[json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()]
r=json.loads((run/'training_result.json').read_text())
x=np.array([a['step'] for a in rows]);y=np.array([a['loss'] for a in rows])
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
ax=axes[0];ax.plot(x,y,color='#bac6d5',lw=.8,label='Stochastic training loss')
ax.plot(x[19:],np.convolve(y,np.ones(20)/20,'valid'),color='#235b92',lw=2,label='20-step moving average')
ax.axvline(100,color='#666',ls='--',lw=1);ax.set(xlabel='Optimizer step',ylabel='Flow-matching loss',title='300 actual GPU updates');ax.legend(fontsize=9);ax.grid(alpha=.15)
ax=axes[1]
ax.plot([0,100,300],[r[k] for k in ('baseline_overfit_mean','after_overfit_mean','final_overfit_mean')],marker='o',color='#235b92',label='Fixed 8 train samples')
ax.plot([0,100,300],[r[k] for k in ('baseline_validation_mean','after_overfit_validation_mean','final_validation_mean')],marker='o',color='#bf6639',label='Fixed 12 validation samples')
ax.set(xlabel='Optimizer step',ylabel='Fixed-noise evaluation loss',title='Same samples and RNG at each evaluation',ylim=(0,.42));ax.grid(alpha=.15);ax.legend(fontsize=9)
fig.suptitle('ECO65 + ZhiXing | pi0.5 LoRA smoke validation | RTX 3090 24 GB',fontsize=14)
fig.savefig(out/'training_curve_20261004_r1.png',dpi=160)
fig.savefig(out/'training_curve_20261004_r1.svg')
