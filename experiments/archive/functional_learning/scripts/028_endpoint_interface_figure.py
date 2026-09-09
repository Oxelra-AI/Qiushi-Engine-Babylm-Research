#!/usr/bin/env python3
"""Make a compact figure for research/028 endpoint interface tests."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

P27 = Path('experiments/archive/functional_learning/data/clean_d_component_ablation/results.json')
P28 = Path('experiments/archive/functional_learning/data/held_fitted_direction_test/results.json')
OUT = Path('experiments/archive/functional_learning/figures/028_endpoint_interface.png')

D27 = json.load(open(P27))
D28 = json.load(open(P28))
arms = ['prep','direct_full','static_1over17','interleaved_ans_full']
labels = ['prep','direct','static 1/17','interleaved']
banks = ['train','held_query','train_in_held_ctx']
bank_labels = ['train query','held query','train query\nin held ctx']
colors = {'prep':'0.35','direct_full':'tab:red','static_1over17':'tab:blue','interleaved_ans_full':'tab:green'}

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

# Panel A: clean accuracy by bank
ax = axes[0,0]
x = np.arange(len(banks)); width = 0.18
for i, arm in enumerate(arms):
    vals = [D27['summary'][arm][b]['clean'] for b in banks]
    ax.bar(x+(i-1.5)*width, vals, width, color=colors[arm], label=labels[i])
ax.set_xticks(x); ax.set_xticklabels(bank_labels)
ax.set_ylim(0,1.05); ax.set_ylabel('clean answer accuracy')
ax.set_title('A. Endpoint behavior on intervention banks')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.25)

# Panel B: centered d-component accuracy (necessity-like)
ax = axes[0,1]
for i, arm in enumerate(arms):
    vals = [D27['summary'][arm][b]['center'] for b in banks]
    ax.bar(x+(i-1.5)*width, vals, width, color=colors[arm], label=labels[i])
ax.axhline(0.25, color='k', ls='--', lw=1, label='chance')
ax.set_xticks(x); ax.set_xticklabels(bank_labels)
ax.set_ylim(0,1.05); ax.set_ylabel('accuracy after centering d')
ax.set_title('B. Removing slot-specific d pattern damages answers')
ax.grid(axis='y', alpha=0.25)

# Panel C: rotated target rate
ax = axes[1,0]
for i, arm in enumerate(arms):
    vals = [D27['summary'][arm][b]['rotate_target'] for b in banks]
    ax.bar(x+(i-1.5)*width, vals, width, color=colors[arm], label=labels[i])
ax.axhline(0.25, color='k', ls='--', lw=1)
ax.set_xticks(x); ax.set_xticklabels(bank_labels)
ax.set_ylim(0,1.05); ax.set_ylabel('answer follows rotated d marker')
ax.set_title('C. Rotating d pattern redirects answer')
ax.grid(axis='y', alpha=0.25)

# Panel D: research held-fitted direction, two-held d-only redirection
ax = axes[1,1]
sources = ['train','held_one','two_held']
src_labels = ['fit train','fit one-held','fit two-held']
x2 = np.arange(len(sources))
for i, arm in enumerate(arms):
    vals = [D28['summary'][arm][src]['redir_two_held'] for src in sources]
    ax.bar(x2+(i-1.5)*width, vals, width, color=colors[arm], label=labels[i])
ax.axhline(0.25, color='k', ls='--', lw=1)
ax.set_xticks(x2); ax.set_xticklabels(src_labels)
ax.set_ylim(0,1.05); ax.set_ylabel('two-held d-only redirect')
ax.set_title('D. Held-fitted directions do not rescue direct-full two-held transfer')
ax.grid(axis='y', alpha=0.25)

fig.suptitle('research/028: endpoint causal interface, necessity-style ablation, and held-code test', y=1.02)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=180, bbox_inches='tight')
print(f'Saved {OUT}')
