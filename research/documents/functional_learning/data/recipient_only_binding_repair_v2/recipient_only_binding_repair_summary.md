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
| train | +0.176 | -0.236 | +0.102 | -0.338 | -0.060 | 22/36 | 15/36 | 16/36 | 1/36 |
| heldout | +0.294 | -0.305 | -0.082 | -0.222 | -0.010 | 7/12 | 5/12 | 5/12 | 0/12 |

## Training comparison

| arm | split | U new>source | R source>new | N source>new | R-N | sensitivity | update | retain | neutral | recipient flip | answer labels | answer visible | background targets |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| standard_wwm15 | train | +0.235 | -0.232 | +0.040 | -0.273 | +0.003 | 20/36 | 16/36 | 17/36 | 0/36 | 1278 | 102 | 44524 |
| standard_wwm15 | heldout | +0.294 | -0.286 | -0.050 | -0.236 | +0.008 | 6/12 | 6/12 | 5/12 | 0/12 | 1278 | 102 | 44524 |
| focused_answer_plus_same_bg | train | +0.001 | +0.009 | +0.405 | -0.396 | +0.010 | 19/36 | 17/36 | 18/36 | 0/36 | 8640 | 0 | 44781 |
| focused_answer_plus_same_bg | heldout | +0.051 | -0.039 | +0.361 | -0.400 | +0.012 | 6/12 | 6/12 | 6/12 | 0/12 | 8640 | 0 | 44781 |

## Interpretation

This repaired run should be read as a recipient-sensitive control, not as proof of a general principle by itself. The same new-state word is used in the target-update and distractor-update contexts, so a successful recipient flip must follow the updated entity rather than the state word alone. Absolute UPDATE, RETAIN, and NEUTRAL margins are reported because a change in R-N alone can be caused by a movement of either term. On held-out repaired groups, coherent86 baseline recipient-flip correctness is 0/12 with U=+0.29, R=-0.30, N=-0.08. After the fixed-budget standard arm it is 0/12 with U=+0.29, R=-0.29, N=-0.05; after focused answer-plus-same-background it is 0/12 with U=+0.05, R=-0.04, N=+0.36. The focused arm does not clearly outperform standard on the repaired held-out contrast, so the credit allocation binding pilot stronger conclusion should be withdrawn; the experience design or scoring frame still permits shortcuts or the intervention fails to install recipient-sensitive selection.

## Files

- summary_json: `experiments/archive/functional_learning/data/recipient_only_binding_repair_v2/recipient_only_binding_repair_summary.json`
- summary_md: `research/documents/functional_learning/data/recipient_only_binding_repair_v2/recipient_only_binding_repair_summary.md`
- train_groups: `experiments/archive/functional_learning/data/recipient_only_binding_repair_v2/train_groups.jsonl`
- heldout_groups: `experiments/archive/functional_learning/data/recipient_only_binding_repair_v2/heldout_groups.jsonl`
- train_packets: `experiments/archive/functional_learning/data/recipient_only_binding_repair_v2/train_packets.jsonl`
- heldout_packets: `experiments/archive/functional_learning/data/recipient_only_binding_repair_v2/heldout_packets.jsonl`
- tokenization_validation: `experiments/archive/functional_learning/data/recipient_only_binding_repair_v2/tokenization_validation.json`
