# earlier analysis repaired recipient-only binding pilot

## Repair implemented

- Recipient-only contrast: UPDATE and RETAIN share source, new state, and final prediction frame; only update recipient changes.
- Query entity and source order are counterbalanced; state words rotate through query-source, distractor-source, and new-state roles.
- Evaluation uses one explicit final `[MASK]` and compares one-token candidates at the same mask position using tokenizer offsets.
- Training comparison uses the same number of update steps. Both arms use 15% WWM background corruption; focused additionally forces the final answer token to `[MASK]`.

## Dataset/token validation

- Train groups: 36 (72 training packets)
- Held-out groups: 12
- Tokenization issues: train=0, held=0
- Train query-side counts: {'A': 18, 'B': 18} ; source-order counts: {'AB': 18, 'BA': 18}
- Held query-side counts: {'A': 6, 'B': 6} ; source-order counts: {'AB': 6, 'BA': 6}

## Baseline coherent86

| split | U new>source | R source>new | N source>new | R-N | sensitivity | update | retain | neutral | recipient flip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | +0.100 | -0.140 | +0.249 | -0.389 | -0.040 | 18/36 | 17/36 | 20/36 | 0/36 |
| heldout | +0.111 | -0.154 | +0.259 | -0.413 | -0.044 | 7/12 | 5/12 | 6/12 | 0/12 |

## Training comparison

| arm | split | U new>source | R source>new | N source>new | R-N | sensitivity | update | retain | neutral | recipient flip | answer labels | answer visible | background targets |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| standard_wwm15 | train | +0.222 | -0.090 | -0.286 | +0.196 | +0.132 | 18/36 | 19/36 | 20/36 | 1/36 | 5327 | 533 | 185590 |
| standard_wwm15 | heldout | +0.182 | -0.230 | -0.183 | -0.047 | -0.048 | 6/12 | 5/12 | 7/12 | 0/12 | 5327 | 533 | 185590 |
| focused_answer_plus_same_bg | train | +0.822 | +1.259 | -0.698 | +1.957 | +2.081 | 25/36 | 33/36 | 12/36 | 22/36 | 36000 | 0 | 186179 |
| focused_answer_plus_same_bg | heldout | +0.113 | -0.200 | -0.890 | +0.689 | -0.087 | 7/12 | 3/12 | 4/12 | 0/12 | 36000 | 0 | 186179 |

## Interpretation

This repaired run should be read as a recipient-sensitive control, not as proof of a general principle by itself. The same new-state word is used in the target-update and distractor-update contexts, so a successful recipient flip must follow the updated entity rather than the state word alone. Absolute UPDATE, RETAIN, and NEUTRAL margins are reported because a change in R-N alone can be caused by a movement of either term. On held-out repaired groups, coherent86 baseline recipient-flip correctness is 0/12 with U=+0.11, R=-0.15, N=+0.26. After the fixed-budget standard arm it is 0/12 with U=+0.18, R=-0.23, N=-0.18; after focused answer-plus-same-background it is 0/12 with U=+0.11, R=-0.20, N=-0.89. The focused arm does not clearly outperform standard on the repaired held-out contrast, so the credit allocation binding pilot stronger conclusion should be withdrawn; the experience design or scoring frame still permits shortcuts or the intervention fails to install recipient-sensitive selection.

## Files

- summary_json: `experiments/archive/functional_learning/data/recipient_only_balanced_offset7_bg15_e500/recipient_only_binding_repair_summary.json`
- summary_md: `research/documents/functional_learning/data/recipient_only_balanced_offset7_bg15_e500/recipient_only_binding_repair_summary.md`
- train_groups: `experiments/archive/functional_learning/data/recipient_only_balanced_offset7_bg15_e500/train_groups.jsonl`
- heldout_groups: `experiments/archive/functional_learning/data/recipient_only_balanced_offset7_bg15_e500/heldout_groups.jsonl`
- train_packets: `experiments/archive/functional_learning/data/recipient_only_balanced_offset7_bg15_e500/train_packets.jsonl`
- heldout_packets: `experiments/archive/functional_learning/data/recipient_only_balanced_offset7_bg15_e500/heldout_packets.jsonl`
- tokenization_validation: `experiments/archive/functional_learning/data/recipient_only_balanced_offset7_bg15_e500/tokenization_validation.json`
