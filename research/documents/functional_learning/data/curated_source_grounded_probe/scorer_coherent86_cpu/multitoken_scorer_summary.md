# Multi-token scorer baseline

## Purpose

The earlier accepted state-update pilot rows usually contain multi-token answers. This scorer symmetrically masks the candidate span in the final use frame and compares answer versus foil by mean token log probability, while also recording summed scores and equal-token-length subsets.

## Validation

- Rows: 12; pairs: 6; pair issues: 0; scoring errors: 0
- Metadata role distribution: {'manual_balanced_unknown': 12}
- Literal target-position counts: {'target_later': 1, 'target_earlier': 5}
- Token length distribution: {'update_answer': {3: 3, 4: 2, 5: 1}, 'update_foil': {7: 1, 4: 2, 2: 1, 1: 1, 3: 1}, 'retain_answer': {7: 1, 4: 2, 2: 1, 1: 1, 3: 1}, 'retain_foil': {3: 3, 4: 2, 5: 1}, 'same_token_length_both_rows_n': 0}

## Baseline pair scores

All scored pairs: n=6, mean U=+3.0251, mean R=-4.0724, mean U+R=-1.0473, joint correct=0/6 by mean-token margin.

Equal-token-length subset: n=0, mean U+R (mean-margin)=+nan, joint correct by summed score=0/0.

By target source position:

- manual_balanced_unknown: n=6, mean U=+3.0251, mean R=-4.0724, mean U+R=-1.0473, joint=0/6

## Interpretation

This is a zero-training readout of accepted pilot rows, not evidence that the rows are semantically clean or suitable for training. Its value is to validate the scorer, expose token-length and position effects, and define the pair-level contrast for the bounded ALN-preserving pilot conditional on cleaner accepted packets becoming available.
