# matched state and initial owner results matched state-format probe

This learned probe keeps the repaired equivariant symmetry repair and macro context text surface but equalizes the state-format arms: each state arm has 192 held-held comparison rows plus 128 state-query rows and the same number of epochs/update passes. It tests whether complete orientation probe results's state-query transfer came from bridge polarity or from generic state-format/optimization effects.

- seeds: `[28100]`
- epochs: `3`
- arms: `['aligned_matched', 'inverted_matched']`
- no official evaluation, upload, or leaderboard interaction

## Training row construction

| arm | total rows | relation rows | state rows | state source |
|---|---:|---:|---:|---|
| aligned_matched | 320 | 192 | 128 | aligned_sparse_state_bridge:128 |
| inverted_matched | 320 | 192 | 128 | inverted_sparse_state_bridge:128 |

## Core readouts (mean over seeds)

`mixed_true_stmt` is true-assignment statement accuracy on mixed held-seen relation comparisons; `state_true_choice` is pair-choice accuracy on paired held-state rows; `state_inverted_choice` is the same rows scored against the globally inverted held-role orientation; `pair_both_true` requires changed and unchanged state choices to be simultaneously true.

| arm | train row-label stmt | mixed true stmt | mixed inverted stmt | paired state true choice | paired state inverted choice | pair_both true | pair_both inverted | exact train state row-label choice |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aligned_matched | 0.753 | 0.508 | 0.492 | 0.629 | 0.457 | 0.383 | 0.203 | 0.672 |
| inverted_matched | 0.750 | 0.500 | 0.500 | 0.668 | 0.488 | 0.438 | 0.219 | 0.766 |

## Support-distance state choices (mean over seeds, true orientation)

| arm | exact aligned train surface | same train names/new objects | first train + second eval | first eval + second train | paired two-eval names | cross-template two-eval |
|---|---:|---:|---:|---:|---:|---:|
| aligned_matched | 0.672 | 0.672 | 0.656 | 0.672 | 0.629 | 0.602 |
| inverted_matched | 0.672 | 0.625 | 0.672 | 0.656 | 0.668 | 0.641 |

## Same support-distance rows scored against inverted orientation

| arm | exact aligned train surface | same train names/new objects | first train + second eval | first eval + second train | paired two-eval names | cross-template two-eval |
|---|---:|---:|---:|---:|---:|---:|
| aligned_matched | 0.422 | 0.422 | 0.375 | 0.453 | 0.457 | 0.477 |
| inverted_matched | 0.484 | 0.500 | 0.609 | 0.531 | 0.488 | 0.516 |

## Changed-state relation split on paired two-eval names

True-orientation choice accuracy by held relation and changed/unchanged query. Relations h0/h2 were directly bridged in aligned/inverted arms; h1/h3 were only connected through held-held comparison rows.

| arm | h0 changed | h1 changed | h2 changed | h3 changed | h0 unchanged | h1 unchanged | h2 unchanged | h3 unchanged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aligned_matched | 0.688 | 0.594 | 0.750 | 0.656 | 0.625 | 0.625 | 0.500 | 0.594 |
| inverted_matched | 0.750 | 0.594 | 0.719 | 0.656 | 0.719 | 0.719 | 0.562 | 0.625 |

## Interpretation from this run

- Mixed held-seen relation orientation remains near chance in the polarity arms: aligned true-statement 0.508, inverted true-statement 0.500. This run therefore does not turn state evidence into a reusable relation coordinate.
- Paired held-state true-choice accuracy: aligned 0.629, inverted scored as true 0.668, inverted scored as inverted 0.488, neutral nan, random-label nan.
- Compare the exact train-state row-label fit and the support-distance tables to decide whether inverted evidence was locally learned, whether any effect follows training identities, and whether neutral/random state-format exposure reproduces held-state transfer.

## Files

- per-row logits: `per_row_predictions.jsonl`
- seed-level summaries: `seed_summaries.json`
- aggregate summaries: `aggregate_summary.json`
- train construction: `matched_train_construction.json`
