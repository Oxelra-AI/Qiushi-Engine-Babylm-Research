#!/usr/bin/env python3
"""Generate research causal intervention figure."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = json.load(open('experiments/archive/functional_learning/data/causal_intervention/results.json'))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: Direction accuracy + redirect rate comparison
models = ['prep', 'direct_full', 'static_1over17']
labels = ['Preparation', 'Direct full\n(collapsed)', 'Static w=1/17\n(preserved)']
colors = ['#2196F3', '#F44336', '#4CAF50']

for pidx, (metric_name, title, ylabel) in enumerate([
    ('dir_acc', 'Direction accuracy at L1\n(prep-learned direction)', 'Accuracy'),
    ('iv_train', 'Train-entity L1 redirect rate', 'Redirect rate'),
    ('cross_model', 'Cross-model transplant\n(preserving → collapsed, L1)', 'Redirect rate'),
]):
    ax = axes[pidx]
    
    if metric_name == 'dir_acc':
        for si, sk in enumerate(['43', '100']):
            sd = d['seeds'][sk]
            vals = [sd['dir_acc'][m].get('1', sd['dir_acc'][m].get(1, 0)) for m in models]
            x = np.arange(len(models)) + si * 0.25 - 0.125
            ax.bar(x, vals, 0.22, color=[colors[i] for i in range(len(models))],
                   alpha=0.7 if si == 0 else 0.4, label=f'Seed {sk}' if si < 1 else None,
                   edgecolor='k', linewidth=0.5)
        ax.axhline(0.25, color='gray', linestyle='--', linewidth=0.8, label='Chance')
        ax.set_xticks(range(len(models))); ax.set_xticklabels(labels, fontsize=9)
        ax.legend(fontsize=8)
        
    elif metric_name == 'iv_train':
        bar_groups = []
        for si, sk in enumerate(['43', '100']):
            sd = d['seeds'][sk]
            for mi, m in enumerate(models):
                iv = sd['iv_train'].get(m, {})
                d_only = iv.get('L1_d_only', {}).get('redirect_rate', 0)
                orth = iv.get('L1_orth_only', {}).get('redirect_rate', 0)
                bar_groups.append((si, mi, m, d_only, orth))
        
        x_base = np.arange(len(models))
        width = 0.15
        for si, sk in enumerate(['43', '100']):
            for mode_i, (mode, alpha) in enumerate([('d_only', 1.0), ('orth_only', 0.3)]):
                vals = []
                for mi, m in enumerate(models):
                    iv = d['seeds'][sk]['iv_train'].get(m, {})
                    vals.append(iv.get(f'L1_{mode}', {}).get('redirect_rate', 0))
                offset = si * (2*width + 0.02) + mode_i * width - width * 1.5
                bars = ax.bar(x_base + offset, vals, width, 
                             color=[colors[mi] for mi in range(len(models))],
                             alpha=alpha, edgecolor='k', linewidth=0.5)
        
        ax.axhline(0.25, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xticks(range(len(models))); ax.set_xticklabels(labels, fontsize=9)
        # Custom legend
        from matplotlib.patches import Patch
        ax.legend([Patch(facecolor='gray', alpha=1.0), Patch(facecolor='gray', alpha=0.3)],
                  ['d_only (direction)', 'orth_only (control)'], fontsize=8, loc='upper left')
        
    elif metric_name == 'cross_model':
        modes_xm = ['L1_full', 'L1_d_only']
        mode_labels = ['Full swap', 'd_only swap']
        x = np.arange(len(modes_xm))
        for si, sk in enumerate(['43', '100']):
            xm = d['seeds'][sk].get('cross_model', {})
            vals = [xm.get(mode, {}).get('redirect_rate', 0) for mode in modes_xm]
            ax.bar(x + si*0.3 - 0.15, vals, 0.25, color=['#FF9800', '#9C27B0'][si],
                   alpha=0.8, label=f'Seed {sk}', edgecolor='k', linewidth=0.5)
        ax.axhline(0.25, color='gray', linestyle='--', linewidth=0.8, label='Chance')
        ax.set_xticks(range(len(modes_xm))); ax.set_xticklabels(mode_labels, fontsize=10)
        ax.legend(fontsize=8)
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(0, 1.08)

fig.suptitle('research: Causal activation intervention for query-match selection',
             fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('experiments/archive/functional_learning/figures/causal_intervention.png',
            dpi=200, bbox_inches='tight')
print(f"Saved figure: experiments/archive/functional_learning/figures/causal_intervention.png")
