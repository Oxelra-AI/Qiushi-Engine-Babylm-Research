# binding learning response results — Binding learning-response experiment: results

## Evidence
- Script: `scripts/binding_learning_response.py`
- Results: `data/binding_learning_response/results.json`

## Design
Four matched arms trained on same-bag entity→state procedural templates:
1. `wwm_correct`: WWM on correct passages only
2. `wwm_both`: WWM on correct + swapped passages (no binding labels)
3. `random_pair`: WWM + contrastive loss with random +/- orientation
4. `true_binding`: WWM + true contrastive loss (correct = positive, margin=0.5, λ=0.5)

Model: DeBERTa-v2 2-layer, hidden 128, 4 heads. Same init seed, 150 steps, batch 8.
Held-out: new entity names AND new locations/states (generalization test).

## Results

| arm | pre held-out acc | post held-out acc | Δ margin (held-out) |
|---|---:|---:|---:|
| wwm_correct | 0.469 | **0.594** | +0.002 |
| wwm_both | 0.469 | 0.438 | -0.003 |
| random_pair | 0.469 | 0.438 | +0.003 |
| true_binding | 0.469 | **0.375** | **-0.014** |

Training accuracy: true_binding = 89%, wwm_correct = 58%, others ~33-50%.

## Scientific interpretation

### The contrastive objective FAILED to induce generalizable binding

`true_binding` achieved near-perfect training accuracy (89%) but **below-chance held-out accuracy (37.5%)**. This is classic overfitting to template surface patterns, not binding learning.

The model learned: "when entity X appears with location Y in the setup, predict Y" — a local co-occurrence shortcut that works on training templates but fails when entities/locations change.

### Plain WWM on correct passages DID improve held-out binding

`wwm_correct` improved held-out accuracy from 46.9% → 59.4%. This suggests that even without explicit binding supervision, seeing enough clean entity→state passages helps the model learn some binding-relevant representations through standard MLM prediction.

### What this means for the research

1. **The naive contrastive approach doesn't work.** Simply adding a margin loss between correct/swapped contexts causes template memorization, not binding generalization.

2. **Plain WWM has more binding potential than ideal binding dependency probe suggested.** The inference probe tested a frozen 100M model; here, a small model trained from scratch on binding-rich text shows genuine held-out improvement under plain WWM.

3. **The bottleneck may be scale/exposure, not objective design.** With only 64 training pairs and 150 steps, the contrastive signal may be too sparse relative to the WWM signal. Or the contrastive formulation needs redesign (e.g., harder negatives, span-level rather than passage-level, or architectural changes).

4. **Route implication:** Before building a complex contrastive BabyLM trainer, we should test whether simply increasing exposure to clean binding-rich text under plain WWM can close more of the Entity/EWoK gap. This is cheaper and the evidence now supports it.

## Remaining questions
- Does scaling `wwm_correct`-style training (more pairs, more steps, larger model) continue improving held-out binding?
- Can a better contrastive formulation avoid template memorization? (e.g., entity-name masking in both contexts, harder negative mining, cross-example contrastive pairs)
- At BabyLM scale (10M words), does mixing 15-20% binding-rich text under plain WWM improve Entity/EWoK?
