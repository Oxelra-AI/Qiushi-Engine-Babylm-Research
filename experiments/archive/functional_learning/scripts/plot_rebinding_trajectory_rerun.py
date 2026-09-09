#!/usr/bin/env python3
"""Plot research typed rebinding trajectory rerun."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INP = Path('experiments/archive/functional_learning/data/rebinding_trajectory_typed/results.json')
OUT = Path('experiments/archive/functional_learning/figures/rebinding_trajectory_rerun.png')
with open(INP) as f: d=json.load(f)
agg=d['aggregate']; arms=['ident_full','ident_blocked','unpaired_src']
labels={'ident_full':'IDENT_FULL','ident_blocked':'IDENT_BLOCKED','unpaired_src':'UNPAIRED_SRC'}
colors={'ident_full':'C0','ident_blocked':'C1','unpaired_src':'C2'}
eps=sorted([int(e) for e in agg.keys()])
fig,axes=plt.subplots(2,3,figsize=(15,8))
fig.suptitle('research typed-cue rerun: acquisition peak, then old-binding retention without robust rebinding',fontsize=13,fontweight='bold')
for arm in arms:
    oldnll=[agg[str(e)][arm]['T_old_scored.rwt_nll']['mean'] for e in eps]
    oldmrr=[agg[str(e)][arm]['T_old_scored.rwt_mrr']['mean'] for e in eps]
    Tm=[agg[str(e)][arm]['rebinding.T_logit_old_minus_new']['mean'] for e in eps]
    Sm=[agg[str(e)][arm]['rebinding.S_logit_new_minus_old']['mean'] for e in eps]
    pair=[agg[str(e)][arm]['rebinding.rebinding_pair_success']['mean'] for e in eps]
    snew=[agg[str(e)][arm]['rebinding.S_new_nll']['mean'] for e in eps]
    axes[0,0].plot(eps,oldnll,'-o',color=colors[arm],label=labels[arm])
    axes[0,1].plot(eps,oldmrr,'-o',color=colors[arm],label=labels[arm])
    axes[0,2].plot(eps,Tm,'-o',color=colors[arm],label=labels[arm])
    axes[1,0].plot(eps,Sm,'-o',color=colors[arm],label=labels[arm])
    axes[1,1].plot(eps,pair,'-o',color=colors[arm],label=labels[arm])
    axes[1,2].plot(eps,snew,'-o',color=colors[arm],label=labels[arm])
axes[0,0].set_title('Old-target held-out NLL: best around epoch 50')
axes[0,0].set_ylabel('RWT(old) NLL')
axes[0,1].set_title('Rank signal modestly improves while NLL worsens')
axes[0,1].set_ylabel('RWT(old) MRR')
axes[0,2].set_title('Original context margin')
axes[0,2].set_ylabel('logit old - logit new in T')
axes[1,0].set_title('Swapped context margin')
axes[1,0].set_ylabel('logit new - logit old in S')
axes[1,1].set_title('Pairwise rebinding success')
axes[1,1].set_ylabel('fraction (both margins > 0)')
axes[1,2].set_title('New target after swap deteriorates')
axes[1,2].set_ylabel('RWT(new) NLL in S')
for ax in axes.ravel():
    ax.set_xlabel('epoch'); ax.grid(alpha=.25); ax.axhline(0,color='black',lw=.6) if 'margin' in ax.get_ylabel().lower() else None
axes[1,1].axhline(0.1,color='gray',lw=.8,ls=':',label='~random top candidate scale')
axes[0,0].legend(fontsize=8)
plt.tight_layout()
OUT.parent.mkdir(parents=True,exist_ok=True)
plt.savefig(OUT,dpi=180,bbox_inches='tight')
print(json.dumps({'status':'ok','figure':str(OUT)}))
