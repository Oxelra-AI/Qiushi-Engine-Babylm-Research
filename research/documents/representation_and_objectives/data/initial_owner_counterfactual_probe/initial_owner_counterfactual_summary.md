# matched state and initial owner results initial-owner counterfactual probe

The original equivariant symmetry repair and macro context state rows always initialize the changed object with the non-final participant, so coherent state supervision can be solved by a weaker transfer-away rule. This probe evaluates the same trained matched arms on rows where the initial owner is controlled independently of the true final role.

- seeds: `[28110, 28111, 28112]`
- epochs: `50`
- arms: `['heldheld_only', 'aligned_matched', 'inverted_matched', 'neutral_matched']`
- no official evaluation, upload, or leaderboard interaction

## Counterfactual readouts

`cf_initial_true_final` means the changed object initially belongs to the same role slot that is the true final owner. A transfer-away-from-initial-owner rule should fail on the changed-object half of this suite. `cf_initial_opposite_true_final` is the original natural-transfer configuration: the object begins with the opposite participant and should move to the true final owner.

| arm | train row-label | paired ordinary true | ordinary changed | ordinary unchanged | cf true-final choice | cf true-final changed | cf true-final unchanged | cf opposite choice | cf opposite changed | cf opposite unchanged | mixed true stmt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heldheld_only | 1.000 | 0.565 | 0.609 | 0.521 | 0.434 | 0.354 | 0.513 | 0.570 | 0.643 | 0.497 | 0.506 |
| aligned_matched | 0.945 | 0.833 | 0.818 | 0.849 | 0.531 | 0.203 | 0.859 | 0.837 | 0.805 | 0.870 | 0.507 |
| inverted_matched | 0.938 | 0.840 | 0.753 | 0.927 | 0.591 | 0.250 | 0.932 | 0.852 | 0.768 | 0.935 | 0.510 |
| neutral_matched | 0.921 | 0.553 | 0.557 | 0.549 | 0.500 | 0.471 | 0.529 | 0.577 | 0.609 | 0.544 | 0.521 |

## Slot-controlled counterfactuals

| arm | initial slot0 changed | initial slot1 changed | initial true-final changed | initial opposite changed |
|---|---:|---:|---:|---:|
| heldheld_only | 0.503 | 0.495 | 0.354 | 0.643 |
| aligned_matched | 0.510 | 0.497 | 0.203 | 0.805 |
| inverted_matched | 0.521 | 0.497 | 0.250 | 0.768 |
| neutral_matched | 0.549 | 0.531 | 0.471 | 0.609 |

## Interpretation

If aligned/inverted state-trained arms score high on the ordinary/opposite-initial rows but near zero on true-final-initial changed rows, the state transfer is a non-initial-owner transition heuristic rather than a role-coordinate representation. If they remain high regardless of initial owner, the evidence supports a stronger final-role coordinate. Mixed relation orientation is reported to keep the state and relation conclusions separate.

## Files

- aggregate JSON: `counterfactual_aggregate.json`
- seed summaries: `counterfactual_seed_summaries.json`
- per-row predictions: `counterfactual_per_row_predictions.jsonl`
- construction: `counterfactual_construction.json`
