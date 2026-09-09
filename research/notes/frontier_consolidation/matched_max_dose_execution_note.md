# matched max dose execution note matched MAX dose execution note

## Scientific role

The current experiment tests the fixed-budget dose response of source+compact-view packets. It is designed to avoid reading the compact effect from a single checkpoint: the intended statistic is the exposure profile.

- Treatment-vs-clean: compare clean0 versus dose1/max view and repeat over the common 10M--80M ladder where clean0 exists.
- Compact-minus-repeat: compare view versus hash-rotated source repeat at dose1 and MAX over 10M--100M.
- MAX-minus-1x: compare whether increasing the restructured fraction changes the compact-minus-repeat term beyond the earlier analysis same-coordinate seed spread.

## Repairs made before GPU launch

Two dose response experiment design construction issues were repaired before H100 training:

1. The combined accepted-pair file was not nested with the inherited 12,155-pair 1x block as its prefix. A new selector preserves the inherited 1x selected block exactly as the inner set and then appends a distribution-matched increment.
2. The dose response experiment design prototype materializer displaced clean-Qwen base words at word-stream level. The validated 1x anchor used coherent 160-word row holdout, so the MAX materializer was rewritten to use row holdout with the same seed and to preserve the old 1x heldout rows as a subset.

## Matched MAX construction

- Selector: `experiments/archive/frontier_consolidation/scripts/dose_distribution_compare_and_select.py`
- Selector output: `experiments/archive/frontier_consolidation/data/dose_distribution_select/selected_matched_max_pairs.jsonl`
- Selector report: `research/documents/frontier_consolidation/data/dose_distribution_select/dose_distribution_comparison_and_selection.md`
- Raw all-available MAX: 2.7299x, but WAR expansion was shorter/denser.
- Matched MAX: 33,291 pairs, 1,118,587 pair words, dose 2.6412x before row rounding.
- Matched increment origins: 490,535 pair words from WAR expansion and 204,541 from unused medium accepted pairs.
- Complete matched MAX feature shifts versus old 1x are about 0.1 old-block standard deviations for source length, compression ratio, content density, and novel rewrite fraction.

## Row-holdout materialization

- Materializer: `experiments/archive/frontier_consolidation/scripts/materialize_matched_max_rowholdout.py`
- Metadata: `experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/dose2p64x_rowholdout_metadata.json`
- Changed block after row rounding: 1,118,720 words = 11.1872% of the 10M pool = 2.64148x the old 423,520-word changed block.
- Selected pair words: 1,118,587; neutral clean-Qwen topup: 133 words.
- Old 1x heldout rows subset of MAX: true, overlap 2,647/2,647.
- View/repeat/clean pool totals: exact 10M, row count 65,313 each.
- View/repeat 100M streams: exact 100M, 653,130 rows each; identical row-length/order and common filler.
- spatial repair route status tokenizer is fixed for all arms; no per-dose tokenizer refit.
- These arms are mechanism instruments, not leaderboard submissions, because their documented text spans the old tokenizer-fitting pool plus additional generated FineWeb views.

## Tokenization audit

- Script: `experiments/archive/frontier_consolidation/scripts/dose_tokenization_audit.py`
- Report: `research/documents/frontier_consolidation/data/dose_tokenization_audit/dose_tokenization_audit.md`
- Complete matched MAX weighted pair tokens/word: 1.4397 versus old 1x prefix 1.4354, shift +0.0043 (+0.30%).
- WAR rows are shorter in absolute pair tokens, but the matched MAX as a whole has a small subword-burden shift under the fixed tokenizer.

## H100 training and dependent evaluation

Training wrapper: `experiments/archive/frontier_consolidation/scripts/train_deberta_matched_max_dose.py`.

Dry-runs passed for both arms with exact 100M streams, parameter count 34,467,424, and tokenizer SHA `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`.

Launched training and planned dependent evaluation:

- MAX view training, run dir `experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022`
- MAX repeat training, run dir `experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022`
- stable-family ladder evaluator waiting for both MAX runs, output dir `experiments/archive/frontier_consolidation/data/dose_ladder_stable_eval`

Evaluation/readout scripts:

- Ladder evaluator: `experiments/archive/frontier_consolidation/scripts/dose_ladder_stable_eval.py`
- Profile reader: `experiments/archive/frontier_consolidation/scripts/dose_ladder_profile_readout.py`

The ladder evaluator reuses existing 1x seed43022 view/repeat rows from earlier analysis and clean rows from legal tokenizer clean control trajectory design where available, evaluates missing clean0 10M/30M/40M/50M/60M and all MAX 10M--100M checkpoints, and writes CSVs for treatment-vs-clean, compact-minus-repeat, and MAX-minus-1x contrasts. The profile reader should be run after the evaluator finishes.
