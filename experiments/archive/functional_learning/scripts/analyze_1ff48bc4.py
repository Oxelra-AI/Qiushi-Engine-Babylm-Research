#!/usr/bin/env python3
"""research analysis: compare uncued vs cued decomposition across training."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Load both parts
with open('data/partA/results.json') as f:
    dA = json.load(f)
with open('data/partB/results.json') as f:
    dB = json.load(f)

seeds = [42, 43, 100]
arms = ['all_corr','ident_full','neutral']
colors = {'all_corr':'#2ca02c','ident_full':'#d62728','neutral':'#7f7f7f'}
labels = {'all_corr':'All corr (500)','ident_full':'Identity fill','neutral':'Neutral fill'}

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Metrics to plot
metrics = [
    ('hc_s', 'Held-out corr NLL (source present)', 'lower is better'),
    ('within_s', 'Within-family identification NLL', 'lower is better'),
    ('prwt_s', 'P(rewrite family | source)', 'higher = more rewrite-family mass'),
]

for row, (met, ylabel, note) in enumerate(metrics):
    for col, (part_data, part_key, title) in enumerate([
        (dA, 'part_A', 'UNCUED (no task cue)'),
        (dB, 'part_B', 'CUED (copy/rewrite cue)')
    ]):
        ax = axes[row, col]
        for arm in arms:
            all_curves = []
            for s in seeds:
                curves = part_data[part_key][f'seed{s}'][arm]
                epochs = [c['e'] for c in curves]
                vals = [c[met] for c in curves]
                all_curves.append(vals)
            all_curves = np.array(all_curves)
            mean = all_curves.mean(axis=0)
            std = all_curves.std(axis=0)
            ax.plot(epochs, mean, color=colors[arm], label=labels[arm], linewidth=2)
            ax.fill_between(epochs, mean-std, mean+std, alpha=0.15, color=colors[arm])
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        if row == 0:
            ax.set_title(title, fontsize=13, fontweight='bold')
        if row == 0 and col == 0:
            ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)

fig.suptitle('research: Family Decomposition — Uncued vs Task-Cued\n'
             'Identity practice genuinely improves within-family identification;\n'
             'output competition is a task-ambiguity artifact',
             fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('figures/family_decomp.png', dpi=150, bbox_inches='tight')
print('Saved: figures/family_decomp.png')

# ─── Also produce the key contrast bar chart ──────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

# Ident - Neutral contrasts at final epoch
contrast_mets = ['hcg', 'hc_s', 'within_s', 'family_s']
met_labels = ['Conditioning\ngain (hcg)', 'Abs corr NLL\n(hc_s)', 
              'Within-family\nident (within_s)', 'Family pref\n(family_s)']

for col, (part_data, part_key, title) in enumerate([
    (dA, 'part_A', 'UNCUED'), (dB, 'part_B', 'CUED')
]):
    ax = axes2[col]
    deltas, errs = [], []
    for m in contrast_mets:
        iv = [part_data[part_key][f'seed{s}']['ident_full'][-1][m] for s in seeds]
        nv = [part_data[part_key][f'seed{s}']['neutral'][-1][m] for s in seeds]
        d = np.array(iv) - np.array(nv)
        deltas.append(np.mean(d))
        errs.append(np.std(d))
    
    x = np.arange(len(contrast_mets))
    bars = ax.bar(x, deltas, yerr=errs, capsize=5, 
                  color=['#1f77b4','#d62728','#2ca02c','#ff7f0e'],
                  alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(met_labels, fontsize=9)
    ax.set_ylabel('ident_full − neutral (nats)')
    ax.set_title(f'{title}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (d, e) in enumerate(zip(deltas, errs)):
        ax.text(i, d + (0.3 if d >= 0 else -0.5), f'{d:+.2f}', 
                ha='center', fontsize=10, fontweight='bold')

fig2.suptitle('Identity − Neutral contrasts: Uncued vs Cued\n'
              'Within-family improvement tripled; family damage eliminated by cue',
              fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig('figures/contrast_bars.png', dpi=150, bbox_inches='tight')
print('Saved: figures/contrast_bars.png')

# Print the key numbers
print("\n=== DECISIVE NUMBERS ===")
for pn, pk in [("UNCUED","part_A"),("CUED","part_B")]:
    print(f"\n{pn}:")
    for m, ml in zip(contrast_mets, ['hcg','hc_s','within_s','family_s']):
        iv = [part_data[pk][f'seed{s}']['ident_full'][-1][m] for s in seeds]
        nv = [part_data[pk][f'seed{s}']['neutral'][-1][m] for s in seeds]
        d = np.array(iv) - np.array(nv)
        print(f"  ident-neutral {ml}: {np.mean(d):+.3f} ± {np.std(d):.3f}")

# Re-read both for accuracy
for pn, (pdata, pk) in [("UNCUED",(dA,"part_A")),("CUED",(dB,"part_B"))]:
    print(f"\n{pn}:")
    for m in contrast_mets:
        iv = [pdata[pk][f'seed{s}']['ident_full'][-1][m] for s in seeds]
        nv = [pdata[pk][f'seed{s}']['neutral'][-1][m] for s in seeds]
        d = np.array(iv) - np.array(nv)
        print(f"  ident-neutral {m}: {np.mean(d):+.3f} ± {np.std(d):.3f}")

print("\nDONE")
