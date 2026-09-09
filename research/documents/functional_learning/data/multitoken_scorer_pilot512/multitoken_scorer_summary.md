# Multi-token scorer baseline

## Purpose

The earlier accepted state-update pilot rows usually contain multi-token answers. This scorer symmetrically masks the candidate span in the final use frame and compares answer versus foil by mean token log probability, while also recording summed scores and equal-token-length subsets.

## Validation

- Rows: 110; pairs: 55; pair issues: 0; scoring errors: 0
- Metadata role distribution: {'target_earlier': 52, 'target_later': 58}
- Literal target-position counts: {'target_earlier': 23, 'target_later': 26, 'distractor_found_target_missing': 2, 'target_found_distractor_missing': 4}
- Token length distribution: {'update_answer': {3: 19, 4: 5, 5: 8, 2: 9, 6: 8, 9: 1, 8: 1, 1: 4}, 'update_foil': {4: 8, 2: 14, 3: 9, 6: 4, 5: 8, 8: 5, 7: 1, 13: 1, 9: 1, 1: 4}, 'retain_answer': {4: 8, 2: 14, 3: 9, 6: 4, 5: 8, 8: 5, 7: 1, 13: 1, 9: 1, 1: 4}, 'retain_foil': {3: 19, 4: 5, 5: 8, 2: 9, 6: 8, 9: 1, 8: 1, 1: 4}, 'same_token_length_both_rows_n': 15}

## Baseline pair scores

All scored pairs: n=55, mean U=+1.9352, mean R=-1.9542, mean U+R=-0.0189, joint correct=3/55 by mean-token margin.

Equal-token-length subset: n=15, mean U+R (mean-margin)=+0.0235, joint correct by summed score=1/15.

By target source position:

- target_earlier: n=26, mean U=+2.9917, mean R=-3.3820, mean U+R=-0.3903, joint=0/26
- target_later: n=29, mean U=+0.9880, mean R=-0.6740, mean U+R=+0.3140, joint=3/29

## Interpretation

This is a zero-training readout of accepted pilot rows, not evidence that the rows are semantically clean or suitable for training. Its value is to validate the scorer, expose token-length and position effects, and define the pair-level contrast for the bounded ALN-preserving pilot conditional on cleaner accepted packets becoming available.
