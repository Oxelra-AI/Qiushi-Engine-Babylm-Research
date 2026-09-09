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
| heldout | +0.061 | -0.011 | +0.331 | -0.342 | +0.050 | 6/12 | 6/12 | 7/12 | 0/12 |

## Training comparison

| arm | split | U new>source | R source>new | N source>new | R-N | sensitivity | update | retain | neutral | recipient flip | answer labels | answer visible | background targets |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| standard_wwm15 | train | +0.100 | -0.140 | +0.249 | -0.389 | -0.040 | 18/36 | 17/36 | 20/36 | 0/36 | 0 | 0 | 0 |
| standard_wwm15 | heldout | +0.061 | -0.011 | +0.331 | -0.342 | +0.050 | 6/12 | 6/12 | 7/12 | 0/12 | 0 | 0 | 0 |
| focused_answer_plus_same_bg | train | +6.448 | +6.952 | +5.542 | +1.410 | +13.400 | 34/36 | 33/36 | 32/36 | 31/36 | 36000 | 0 | 0 |
| focused_answer_plus_same_bg | heldout | +6.105 | +1.079 | +5.766 | -4.687 | +7.183 | 11/12 | 6/12 | 11/12 | 5/12 | 36000 | 0 | 0 |

## Interpretation

This repaired run should be read as a recipient-sensitive control, not as proof of a general principle by itself. The same new-state word is used in the target-update and distractor-update contexts, so a successful recipient flip must follow the updated entity rather than the state word alone. Absolute UPDATE, RETAIN, and NEUTRAL margins are reported because a change in R-N alone can be caused by a movement of either term. On held-out repaired groups, coherent86 baseline recipient-flip correctness is 0/12 with U=+0.06, R=-0.01, N=+0.33. After the fixed-budget standard arm it is 0/12 with U=+0.06, R=-0.01, N=+0.33; after focused answer-plus-same-background it is 5/12 with U=+6.10, R=+1.08, N=+5.77. The focused arm gives the stronger repaired recipient-sensitive learning signal, but the conclusion remains bounded to this controlled packet substrate and private-adapter short run. The next practical bridge should test whether an ALN-preserving mixed objective can keep this repaired readout while preserving cheap7 downstream competence.

## Files

- summary_json: `experiments/archive/functional_learning/data/recipient_only_balanced_answer_only_lr1e4_e500/recipient_only_binding_repair_summary.json`
- summary_md: `research/documents/functional_learning/data/recipient_only_balanced_answer_only_lr1e4_e500/recipient_only_binding_repair_summary.md`
- train_groups: `experiments/archive/functional_learning/data/recipient_only_balanced_answer_only_lr1e4_e500/train_groups.jsonl`
- heldout_groups: `experiments/archive/functional_learning/data/recipient_only_balanced_answer_only_lr1e4_e500/heldout_groups.jsonl`
- train_packets: `experiments/archive/functional_learning/data/recipient_only_balanced_answer_only_lr1e4_e500/train_packets.jsonl`
- heldout_packets: `experiments/archive/functional_learning/data/recipient_only_balanced_answer_only_lr1e4_e500/heldout_packets.jsonl`
- tokenization_validation: `experiments/archive/functional_learning/data/recipient_only_balanced_answer_only_lr1e4_e500/tokenization_validation.json`
