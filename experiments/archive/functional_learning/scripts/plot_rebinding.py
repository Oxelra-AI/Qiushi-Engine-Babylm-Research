#!/usr/bin/env python3
"""Plot research trajectory and rebinding summaries."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INP = Path('experiments/archive/functional_learning/data/rebinding_analysis/results.json')
OUT = Path('experiments/archive/functional_learning/figures/rebinding_trajectory.png')
with open(INP) as f:
    d = json.load(f)
cur = d['curve_summary']['cue_typed']
arms = ['ident_full','ident_blocked','unpaired_src']
labels = {'ident_full':'IDENT_FULL','ident_blocked':'IDENT_BLOCKED','unpaired_src':'UNPAIRED_SRC'}
colors = {'ident_full':'C0','ident_blocked':'C1','unpaired_src':'C2'}

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('research: research models learn late overconfident heuristics more than robust rebinding', fontsize=13, fontweight='bold')

# Typed trajectories: absolute NLL and MRR
for arm in arms:
    eps = sorted([int(x) for x in cur.keys()])
    nll = [cur[str(e)][arm]['rwt_nll']['mean'] for e in eps]
    wnll = [cur[str(e)][arm]['rwt_wnll']['mean'] for e in eps]
    mrr = [cur[str(e)][arm]['rwt_mrr']['mean'] for e in eps]
    axes[0,0].plot(eps, nll, marker='o', color=colors[arm], label=labels[arm])
    axes[0,1].plot(eps, wnll, marker='o', color=colors[arm], label=labels[arm])
    axes[0,2].plot(eps, mrr, marker='o', color=colors[arm], label=labels[arm])
axes[0,0].set_title('Held-out correct RWT NLL rises after epoch 50')
axes[0,0].set_ylabel('T RWT NLL (lower is better)')
axes[0,1].set_title('Deterioration is within-family overconfidence')
axes[0,1].set_ylabel('T within-RWT NLL')
axes[0,2].set_title('Rank signal remains modest while probabilities collapse')
axes[0,2].set_ylabel('T within-RWT MRR')
for ax in axes[0]:
    ax.set_xlabel('Epoch')
    ax.grid(alpha=.25)
axes[0,0].legend(fontsize=8)

# Typed source-specific interaction by epoch
for m, ax, title in [('rwt_nll', axes[1,0], 'Old-target T/U interaction D on NLL'),
                     ('rwt_wnll', axes[1,1], 'Old-target T/U interaction D within RWT')]:
    for ia, style in [('D_ident_full','-o'),('D_ident_blocked','--s')]:
        eps = sorted([int(x) for x in cur.keys()])
        vals = [cur[str(e)]['interactions'][ia][m]['mean'] for e in eps]
        ax.plot(eps, vals, style, label=ia, alpha=.9)
    ax.axhline(0, color='black', lw=.7)
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('D = (I_T-UP_T)-(I_U-UP_U)')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)

# Rebinding final margins
agg = d['rebinding_aggregate']
ax = axes[1,2]
x = np.arange(len(arms))
width = .36
for ci, cue in enumerate(['cue_typed','cue_uninformative']):
    vals = [agg[cue][arm]['binding_margin_sum']['mean'] for arm in arms]
    errs = [agg[cue][arm]['binding_margin_sum']['std_across_seeds'] for arm in arms]
    ax.bar(x + (ci-.5)*width, vals, width, yerr=errs, label=cue.replace('cue_',''), alpha=.75)
ax.axhline(0, color='black', lw=.7)
ax.set_xticks(x)
ax.set_xticklabels([labels[a] for a in arms], rotation=25, ha='right')
ax.set_title('Final rebinding margin sum: old in T + new after swap')
ax.set_ylabel('mean logit margin sum (>0 supports rebinding)')
ax.grid(axis='y', alpha=.25)
ax.legend(fontsize=8)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=180, bbox_inches='tight')
print(json.dumps({'status':'ok','figure':str(OUT)}))
