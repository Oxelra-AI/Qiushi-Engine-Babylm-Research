# clean qwen control interpretation and next mechanism — Mechanism-control training and shuffled-correspondence audit

## Completed mechanism-control training


Both clean qwen control interpretation and next mechanism mechanism controls completed 100M-word training with the same DeBERTa-v2 8×480 / baseline16k / WWM / seed43022 recipe used by the first clean-Qwen seed.

### Runs

- `training/runs/selected_original_dup_all_16k_seed43022`
  - word exposure: 100,000,000
  - actual training steps: 2,518
  - loss: 9.8181 → 2.3406
  - params: 34,467,424
  - vocab: 16,384
  - saved checkpoints: 100
  - full `chck_1M..chck_100M` ladder present

- `training/runs/qwen_separated_pair_16k_seed43022`
  - word exposure: 100,000,000
  - actual training steps: 2,475
  - loss: 9.8067 → 2.7528
  - params: 34,467,424
  - vocab: 16,384
  - saved checkpoints: 100
  - full `chck_1M..chck_100M` ladder present

Training loss is not downstream competence evidence. The lower final loss of selected-original duplication likely reflects easier within-row exact repetition, not necessarily stronger transfer; the full official-style evaluation was pending in this record.

## Corpus mechanisms being tested

### `selected_original_dup_all`

Metadata: `data/selected_original_dup_all_control/selected_original_dup_all_metadata.json`.

- Keeps all 37,594 selected official original sentences.
- Duplicates every selected original once: `(original, original)`.
- Uses no Qwen words.
- Uses official text only.
- Preserves duplicate-pair boundaries.
- Matches clean-Qwen effective source totals exactly.
- Duplicate-pair words: 1,698,026 (fraction 0.169803), near the clean-Qwen pair-word budget 1,656,800.

This tests whether the clean-Qwen effect is mostly selected-original extraction/redundancy rather than generated second-view value.

### `qwen_separated_pair`

Metadata: `data/qwen_separated_pair_control/qwen_separated_pair_metadata.json`.

- Uses the same 37,594 selected original sentences and same Qwen rewrite multiset as the aligned treatment.
- Places originals and rewrites in separate rows/windows; `original_and_rewrite_in_same_row=false`.
- Pair words exactly match clean-Qwen: 1,656,800 (fraction 0.16568).
- Matches clean-Qwen effective source totals exactly.

This tests whether the clean-Qwen effect requires same-window attention-visible original--rewrite correspondence, or whether merely having both views somewhere in the corpus is enough.

## Selected-exposure invariant

Audit: `data/selected_exposure_audit.json`.

The audit shows that all compared arms have the same selected example-ID exposure total per 10M pool:

- clean aligned: 4,077,760 words
- shuffled control: 4,077,760 words
- separated-pair control: 4,077,760 words
- selected-original-dup-all control: 4,077,760 words
- old original-dup control: 4,077,760 words
- full official pool reference: 4,077,760 words

Thus the remaining selection explanation is not simple over-exposure of selected original rows beyond the official pool. It is whether extracting selected sentences and replacing their row context with pair/duplicate material changes the learnable signal.

## Shuffled residual-similarity audit

Audit: `data/shuffled_residual_similarity_audit.json`.

The shuffled control is a strong lexical/content correspondence break, despite shuffling only within source × rewrite-length bins:

- changed pairs: 37,594 / 37,594
- aligned content-Jaccard mean: 0.463335
- shuffled content-Jaccard mean: 0.005672
- aligned content-recall original→rewrite mean: 0.608036
- shuffled content-recall original→rewrite mean: 0.011009
- aligned entity recall mean: 0.688696
- shuffled entity recall mean: 0.035355
- aligned number recall mean: 0.998471
- shuffled number recall mean: 0.003996
- shuffled fraction content-Jaccard ≥ 0.4: 0.0
- high residual shuffled-overlap examples: none

This substantially strengthens the interpretation of `qwen_clean_aligned - qwen_shuffled_control = +0.7135`: the contrast is not just same generated-text register, and not a same-topic residual caused by within-source shuffling. It supports a local correspondence/coherence component, though not by itself an abstract semantic-alignment principle.

## Pending decisive evaluation

The full official-style evaluation for `selected_original_dup_all` and `qwen_separated_pair` was pending, with the following evaluation configuration:

- `scripts/full_eval_runner.py`
- `scripts/launch_mechanism_eval.sh`
- `scripts/summarize_mechanism_eval.py`

Interpretation once complete:

- If `qwen_clean_aligned - selected_original_dup_all` remains positive, generated Qwen second views add value beyond duplicating all selected originals.
- If it collapses, selected-original extraction/redundancy explains most of the effect and the route should pivot toward official-only selected exposure or quality-weighted curricula.
- If `qwen_clean_aligned - qwen_separated_pair` remains positive, same-window original--rewrite adjacency/correspondence is an active component.
- If it collapses, separated coexistence/augmentation is enough, and future optimization should focus on rewrite selection/dose rather than adjacent packing.
- If `qwen_separated_pair - qwen_shuffled_control` is positive but `qwen_clean_aligned - qwen_separated_pair` is small, correct pair identity matters in corpus exposure, but same-window adjacency may not be required.
