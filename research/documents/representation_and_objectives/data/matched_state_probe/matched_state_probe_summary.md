# matched state and initial owner results matched state-format probe

This learned probe keeps the repaired equivariant symmetry repair and macro context text surface but equalizes the state-format arms: each state arm has 192 held-held comparison rows plus 128 state-query rows and the same number of epochs/update passes. It tests whether complete orientation probe results's state-query transfer came from bridge polarity or from generic state-format/optimization effects.

- seeds: `[28100, 28101, 28102]`
- epochs: `50`
- arms: `['heldheld_only', 'heldheld_repeat_control', 'aligned_matched', 'inverted_matched', 'neutral_matched', 'random_label_matched']`
- no official evaluation, upload, or leaderboard interaction

## Training row construction

| arm | total rows | relation rows | state rows | state source |
|---|---:|---:|---:|---|
| heldheld_only | 192 | 192 | 0 | none |
| heldheld_repeat_control | 320 | 320 | 0 | none |
| aligned_matched | 320 | 192 | 128 | aligned_sparse_state_bridge:128 |
| inverted_matched | 320 | 192 | 128 | inverted_sparse_state_bridge:128 |
| neutral_matched | 320 | 192 | 128 | neutral_matched_seen_state_with_held_distractor:128 |
| random_label_matched | 320 | 192 | 128 | random_label_state_format:128 |

## Core readouts (mean over seeds)

`mixed_true_stmt` is true-assignment statement accuracy on mixed held-seen relation comparisons; `state_true_choice` is pair-choice accuracy on paired held-state rows; `state_inverted_choice` is the same rows scored against the globally inverted held-role orientation; `pair_both_true` requires changed and unchanged state choices to be simultaneously true.

| arm | train row-label stmt | mixed true stmt | mixed inverted stmt | paired state true choice | paired state inverted choice | pair_both true | pair_both inverted | exact train state row-label choice |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| heldheld_only | 1.000 | 0.498 | 0.502 | 0.543 | 0.350 | 0.292 | 0.102 | nan |
| heldheld_repeat_control | 1.000 | 0.473 | 0.527 | 0.544 | 0.354 | 0.289 | 0.109 | nan |
| aligned_matched | 0.941 | 0.490 | 0.510 | 0.776 | 0.594 | 0.552 | 0.318 | 0.885 |
| inverted_matched | 0.945 | 0.490 | 0.510 | 0.842 | 0.525 | 0.685 | 0.182 | 0.859 |
| neutral_matched | 0.966 | 0.516 | 0.484 | 0.509 | 0.483 | 0.271 | 0.221 | 0.979 |
| random_label_matched | 0.867 | 0.490 | 0.510 | 0.552 | 0.536 | 0.302 | 0.286 | 0.698 |

## Support-distance state choices (mean over seeds, true orientation)

| arm | exact aligned train surface | same train names/new objects | first train + second eval | first eval + second train | paired two-eval names | cross-template two-eval |
|---|---:|---:|---:|---:|---:|---:|
| heldheld_only | 0.568 | 0.526 | 0.609 | 0.557 | 0.543 | 0.544 |
| heldheld_repeat_control | 0.536 | 0.557 | 0.578 | 0.609 | 0.544 | 0.560 |
| aligned_matched | 0.885 | 0.792 | 0.750 | 0.781 | 0.776 | 0.776 |
| inverted_matched | 0.844 | 0.833 | 0.839 | 0.839 | 0.842 | 0.833 |
| neutral_matched | 0.516 | 0.526 | 0.526 | 0.464 | 0.509 | 0.513 |
| random_label_matched | 0.490 | 0.484 | 0.469 | 0.552 | 0.552 | 0.531 |

## Same support-distance rows scored against inverted orientation

| arm | exact aligned train surface | same train names/new objects | first train + second eval | first eval + second train | paired two-eval names | cross-template two-eval |
|---|---:|---:|---:|---:|---:|---:|
| heldheld_only | 0.359 | 0.370 | 0.380 | 0.380 | 0.350 | 0.362 |
| heldheld_repeat_control | 0.422 | 0.380 | 0.391 | 0.370 | 0.354 | 0.372 |
| aligned_matched | 0.583 | 0.646 | 0.635 | 0.625 | 0.594 | 0.589 |
| inverted_matched | 0.531 | 0.542 | 0.536 | 0.516 | 0.525 | 0.531 |
| neutral_matched | 0.547 | 0.536 | 0.526 | 0.536 | 0.483 | 0.497 |
| random_label_matched | 0.562 | 0.547 | 0.573 | 0.542 | 0.536 | 0.536 |

## Changed-state relation split on paired two-eval names

True-orientation choice accuracy by held relation and changed/unchanged query. Relations h0/h2 were directly bridged in aligned/inverted arms; h1/h3 were only connected through held-held comparison rows.

| arm | h0 changed | h1 changed | h2 changed | h3 changed | h0 unchanged | h1 unchanged | h2 unchanged | h3 unchanged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| heldheld_only | 0.688 | 0.646 | 0.729 | 0.708 | 0.396 | 0.406 | 0.396 | 0.375 |
| heldheld_repeat_control | 0.656 | 0.646 | 0.750 | 0.708 | 0.365 | 0.417 | 0.406 | 0.406 |
| aligned_matched | 0.708 | 0.656 | 0.698 | 0.667 | 0.865 | 0.865 | 0.875 | 0.875 |
| inverted_matched | 0.844 | 0.802 | 0.812 | 0.812 | 0.823 | 0.906 | 0.833 | 0.906 |
| neutral_matched | 0.615 | 0.479 | 0.542 | 0.469 | 0.469 | 0.458 | 0.552 | 0.490 |
| random_label_matched | 0.615 | 0.469 | 0.531 | 0.448 | 0.479 | 0.646 | 0.573 | 0.656 |

## Interpretation from this run

- Mixed held-seen relation orientation remains near chance in the polarity arms: aligned true-statement 0.490, inverted true-statement 0.490. This run therefore does not turn state evidence into a reusable relation coordinate.
- Paired held-state true-choice accuracy: aligned 0.776, inverted scored as true 0.842, inverted scored as inverted 0.525, neutral 0.509, random-label 0.552.
- Compare the exact train-state row-label fit and the support-distance tables to decide whether inverted evidence was locally learned, whether any effect follows training identities, and whether neutral/random state-format exposure reproduces held-state transfer.

## Files

- per-row logits: `per_row_predictions.jsonl`
- seed-level summaries: `seed_summaries.json`
- aggregate summaries: `aggregate_summary.json`
- train construction: `matched_train_construction.json`
