# wholeword copied control plan whole-word copied-content control for target-selective source-absent channel

## Why this run exists

target selective source absent result opened a causal channel inside the successful masked-denoising geometry: keeping all text/interface/masks/init fixed, removing all source-absent compact-content target labels (`rw_abs_content`, 7,649 pieces) made later fixed-event source-absent denoising worse than removing an equal number of copied target pieces. At 20M the piece-weighted `drop_abs_minus_drop_copied` on source-absent probe events was +0.119794 nats with pair-cluster bootstrap interval [+0.097717,+0.140308]; the effect persisted on source-absent probe words never selected by the two-epoch WWM schedule (+0.092402). This showed more than exact selected-event memorization.

The target selective source absent result comparator was imperfect: it selected copied token pieces independently, so 3,450/7,226 touched copied target groups were partially deleted. A copied word might still receive some supervision through remaining subword pieces. The strategist note therefore requires one stronger comparator before the source-absent channel becomes load-bearing.

## wholeword copied control plan selection audit

Script: `experiments/archive/representation_and_objectives/scripts/build_wholeword_copied_content_selection.py`.

Output: `experiments/archive/representation_and_objectives/data/wholeword_copied_content_selection/wholeword_copied_content_selection_audit.json`.

The selected control deletes whole copied-content target-word groups under the exact crossview v2 20M result own-visible 20M interface:

- Data SHA: `1123d5ab1b0127618d9f9fd72accf39c078ecef047ecf9213d0b9622dea7ccff`.
- Same tokenizer as triangle/crossview: `training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_1M`.
- Exact source-absent targets to match: 5,779 whole absent-content groups, 7,649 BPE pieces.
- Copied-content candidate pool: 26,999 whole groups, 38,000 BPE pieces.
- Selected copied-content control: 5,779 whole groups, 7,649 BPE pieces.
- Exact BPE-length matching for every selected group: fraction 1.0.
- Exact epoch and identity-exposure matching for every selected group: `match_class_counts = {"exact_len|same_epoch_exposure": 5779}`.
- Mean feature deltas selected minus absent: BPE length 0.0, support_log_mean +0.002011, support_log_min +0.001521, relative word position -0.001583, relative token midpoint -0.000577, sequence token midpoint -0.003203, row-order fraction +0.000153.

Precompute with `target_selective_wholeword_trainer.py --precompute_only` confirmed:

- original target counts: filler 4,134,298; source 110,478; rw_copied 65,723; rw_abs_content 7,649; rw_abs_other 2,537; all 4,320,685.
- kept target counts after whole-word copied deletion: filler 4,134,298; source 110,478; rw_copied 58,074; rw_abs_content 7,649; rw_abs_other 2,537; all 4,313,036.
- dropped copied-content pieces: 7,649.
- dropped source-absent pieces: 0.
- no unexpected positions.

This is a much stronger copied-label comparator than target selective source absent result because it matches the source-absent deletion at the whole-target-group level while preserving the same deleted BPE mass and almost identical support/position/exposure.

## Training run

Script: `experiments/archive/representation_and_objectives/scripts/target_selective_wholeword_trainer.py`.

The whole-word copied-content control is in progress.

Output run root: `experiments/archive/representation_and_objectives/training/runs/target_selective_drop_copied_content_wholeword_20M`.

It is a single extra 20M arm from the same frozen crossview v2 20M result init SHA `ddc8532a6eb81d2eed0e7671fdd06d68dfeeac08f96f4d7d8e057391b6ed5e68`, same data, same identity-attached WWM masks, same source+rewrite visible input, same optimizer/LR schedule. It changes only the loss labels for the selected whole copied-content groups.

Scientific meaning before result:

- If `drop_abs_content` remains worse than `drop_copied_content_wholeword` specifically on source-absent compact-side fixed events, the target selective source absent result source-absent target-type channel survives a much stronger word-level control.
- If the effect collapses, target selective source absent result was likely exaggerated by partial subword deletion and arbitrary copied-token selection; the internal effect should then be downgraded and not advanced into historical geometry.
- Either outcome is about the local 20M masked-denoising mechanism. The pending cheap7 readout is only early external coupling, not a route decision, because the historical compact advantage matured late.

## Prepared readout

Script: `experiments/archive/representation_and_objectives/scripts/wholeword_control_readout.py`.

It compares `full`, target selective source absent result `drop_abs`, target selective source absent result token-piece `drop_copied_tok`, and wholeword copied control plan `drop_copied_word` at `chck_10M` and `chck_20M` on the fixed 12,288 earlier analysis compact-side probe events, with pair-cluster bootstrap. The central value is `drop_abs_minus_drop_copied_word` on `source_absent_content`.

A follow-up should also adapt the target selective source absent result exposure split to this new arm so we can see whether any surviving source-absent effect persists on probe words never WWM-selected during the two training epochs.
