# roberta stratified bridge and early local response RoBERTa pair-stratified local response

Created: `2026-09-02T02:37:03Z`

Positive `repeat_minus_compact_advantage` means the compact-trained model predicts the masked event better than the repeat-trained model. Interpret only with selected official-compatible trajectory results.

## chck_1M
- Overall local advantage: `-0.0` over `10` events / `14` pieces
- View/category advantages:
  - `compact|function_other`: advantage `-0.0`, events `1`
  - `compact|retained_content`: advantage `-0.0`, events `3`
  - `repeat|function_other`: advantage `-0.0`, events `4`
  - `repeat|retained_content`: advantage `-0.0`, events `2`
- Feature contrast highlights (positive = stronger compact local advantage in high/high-positive stratum):
  - `compact_content_fraction|repeat|retained_content`: `0.0`
  - `compact_source_absent_content_fraction_of_content|compact|retained_content`: `0.0`
  - `compact_source_absent_content_fraction_of_content|repeat|retained_content`: `0.0`
  - `compact_tail_content_coverage|repeat|function_other`: `0.0`
  - `compact_tail_content_coverage|repeat|retained_content`: `0.0`

This file is a mechanism bridge, not an endpoint score or causal intervention.
