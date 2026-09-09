"""Regenerate fig_data_structure.pdf with corrected insight text.
Fixes: 'most economical explanation' → 'increasing relative learning pressure on cross-view evidence'
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = "data/external/figures"

fig, axes = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw={'height_ratios': [1, 1.5]})

# Top panel: Packed row structure
ax = axes[0]
ax.set_xlim(0, 100)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title("Packed Qwen-Pair Training Row", fontsize=10, fontweight='bold', pad=8)

segments = [
    (2, 22, '#d5f4e6', '#27ae60', "Original 1"),
    (23, 47, '#d6eaf8', '#2980b9', "Rewrite 1"),
    (48, 68, '#d5f4e6', '#27ae60', "Original 2"),
    (69, 93, '#d6eaf8', '#2980b9', "Rewrite 2"),
]
for x0, x1, fc, ec, label in segments:
    rect = FancyBboxPatch((x0, 3), x1-x0, 4, boxstyle="round,pad=0.2",
                          facecolor=fc, edgecolor=ec, linewidth=1.2)
    ax.add_patch(rect)
    ax.text((x0+x1)/2, 5, label, ha='center', va='center', fontsize=8, fontweight='bold')

ax.annotate('View 1 (original)', xy=(12, 8.5), fontsize=8, color='#27ae60', fontweight='bold', ha='center')
ax.annotate('View 2 (rewrite)', xy=(35, 8.5), fontsize=8, color='#2980b9', fontweight='bold', ha='center')
ax.text(50, 1.0, "Typical row: 3-4 paired segments, ~150 words", 
        ha='center', va='center', fontsize=7, style='italic', color='#666')

# Bottom panel: Three masking conditions
ax2 = axes[1]
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 18)
ax2.axis('off')
ax2.set_title("Three Masking/Credit Strategies on the Same Row", fontsize=10, fontweight='bold', pad=8)

conditions = [
    (14.5, "Ordinary WWM (15%)", 
     "Random 15% of all positions masked and predicted\n(applied to non-pair rows and as ordinary-mask preservation target)",
     '#ecf0f1', '#7f8c8d'),
    (9.0, "Dense mask + sparse label (M,S)",
     "All detected content groups in View 2 are masked (132,283 groups)\nLoss computed only on a deterministic sparse subset (21,479 groups)\nView 1 (original) remains fully visible throughout",
     '#fdebd0', '#e67e22'),
    (3.5, "Dense mask + dense label (M,M)",
     "Same dense masking of View 2 content groups\nLoss computed on ALL masked positions (132,283 groups)",
     '#fadbd8', '#e74c3c'),
]

for y, title, desc, fc, ec in conditions:
    rect = FancyBboxPatch((3, y-1.2), 94, 3.8, boxstyle="round,pad=0.3",
                          facecolor=fc, edgecolor=ec, linewidth=1.0)
    ax2.add_patch(rect)
    ax2.text(6, y+1.0, title, va='center', fontsize=8.5, fontweight='bold', color=ec)
    ax2.text(6, y-0.2, desc, va='center', fontsize=7.5, color='#333')

# CORRECTED key mechanism note
ax2.text(50, 16.5, 
    "Dense masking removes within-view completion clues,\nincreasing the relative learning pressure on cross-view evidence\nwhile leaving other contextual cues possible",
    ha='center', va='center', fontsize=8, style='italic',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff3cd', edgecolor='#ffc107', alpha=0.9))

fig.tight_layout(pad=1.5)
fig.savefig(f"{OUT}/fig_data_structure.pdf", bbox_inches='tight')
plt.close()
print("fig_data_structure.pdf regenerated with corrected text")
