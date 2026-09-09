# binding learning response v2 results — Repaired binding learning-response (v2): results

## Critical fix applied

Target word now APPENDED to query. Downstream query target masked; setup target visible.
Verified in printed examples:
```
Correct:  Mark moved to the garden. Kate moved to the garage. To visit Mark go to the garden.
Masked:   Mark moved to the garden. Kate moved to the garage. To visit Mark go to the<mask>.
Target:   'garden' positions=[18]  ✓ Query target confirmed
```

## Design

4 arms × 3 seeds (42, 123, 789). DeBERTa-v2 2-layer/128-hidden. 150 steps, batch 8.
48 training pairs. Held-out splits: new_ent_train_state (24), train_ent_new_state (18), 
new_ent_new_state (18), heldout_template_family (8).

## Aggregate results (mean across 3 seeds)

| arm | train acc | new_ent_train_state | train_ent_new_state | new_ent_new_state | heldout_template |
|---|---:|---:|---:|---:|---:|
| wwm_correct | 0.632 | 0.555 | 0.593 | 0.611 | 0.458 |
| wwm_both | 0.562 | 0.472 | 0.648 | 0.481 | 0.583 |
| random_pair | 0.674 | 0.528 | 0.630 | 0.426 | 0.417 |
| true_binding | **1.000** | 0.444 | 0.537 | 0.426 | 0.708 |

Mean margins (all arms, all held-out splits): ≤ ±0.01, most ≤ 0.003.

## Scientific conclusions

### 1. Naive contrastive binding objective: CONFIRMED FAILURE

`true_binding` achieves 100% training accuracy but below-chance held-out accuracy on the 
critical new_ent_new_state split (0.426) and new_ent_train_state (0.444). Extreme seed 
variance on train_ent_new_state (margins: -0.027, +0.074, -0.016) shows instability rather 
than signal. Even with the corrected downstream target masking, the margin-based contrastive 
loss causes template/entity-name memorization, not generalizable binding.

### 2. Plain WWM on binding-rich text: NO stable improvement over controls

wwm_correct is NOT consistently better than wwm_both or random_pair across splits:
- new_ent_new_state: wwm_correct 0.611 > wwm_both 0.481 (looks positive)  
- BUT train_ent_new_state: wwm_correct 0.593 < wwm_both 0.648 (reversed)
- AND heldout_template: wwm_correct 0.458 < wwm_both 0.583 (reversed)

All margins are effectively zero (≤0.003). The binding learning response results "wwm_correct improvement" was noise
from single-seed measurement on the wrong target position.

### 3. Seed variance dominates the signal

wwm_correct heldout_template accuracies: [0.125, 0.75, 0.5] — ranges from far below to 
far above chance within the same arm. This shows the sample sizes and model capacity are 
insufficient to draw conclusions about binding learning.

## What this closes

- **Naive margin-based contrastive objective on synthetic templates**: closed. Does not 
  generalize binding even with corrected downstream target masking, 3 seeds, and multiple 
  held-out splits.
- **"Plain WWM learns binding from binding-rich text" claim from binding learning response results**: NOT supported.
  The corrected multi-seed experiment shows no stable separation from controls.

## Remaining viable routes

1. **Redesigned contrastive objective** with stronger anti-shortcut controls:
   - Entity-name masking in BOTH correct and swapped contexts
   - Cross-example negatives (different template instances)
   - Larger/more diverse training set
   - Unseen entities in BOTH training and evaluation
   
2. **Persistent entity-state architecture**: add recurrent memory or entity slots that 
   must track who-did-what across the sequence.

3. **Accept the binding mechanism limitation** and close the Overall gap through columns 
   where the protected model already exceeds the leader (Supplement +3.87, Reading +2.20)
   combined with GlobalPIQA gains from S1 shape, if SuperGLUE+AoA can contribute.
