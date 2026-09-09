# dualview panel readiness and budget — dual-view panel readiness, control semantics, and charged-budget tradeoff

## Completed training arms

All three required 20M-prefix training arms now exist under the corrected trainer / charged-exposure definition:

- aligned: `training/runs/dualview_aligned_20M_seed43022`
  - 481 updates; 19,021,584 main words + 966,720 auxiliary words = 19,988,304 charged words
  - final loss 4.379467; mean aux loss 7.470588
  - checkpoint under `hf_model/final` (no `chck_20M`, because next batch would exceed cap)
- shuffled: `training/runs/dualview_shuffled_20M_seed43022`
  - same update/exposure/mask/target/accounting as aligned
  - final loss 4.439597; mean aux loss 7.532271
  - checkpoint under `hf_model/final`
- mlm_only exact reference: `training/runs/dualview_mlm_only_20M_seed43022`
  - 506 updates; 20,000,000 main words + 0 auxiliary words = 20,000,000 charged words
  - final loss 3.870622
  - checkpoint under `hf_model/chck_20M` and `hf_model/final`

The full training-accounting audit `data/dualview_training_audit/aligned_shuffled_mlm_only_completed.json` confirms aligned vs shuffled have identical logs for batch words, aux words, cumulative exposures, masked tokens, auxiliary targets, auxiliary unit counts, view counts, LR sequence, and final accounting. This makes aligned-vs-shuffled a clean source-correspondence contrast.

## Shuffled-control semantics

A possible confound was that the trainer's per-batch source derangement might not match the document-disjoint length-matched decoy that source free transfer synthesis validated. Two audits now resolve this:

- `data/dualview_shuffle_control_audit/shuffle_control_audit.json`
- `data/dualview_control_semantics/control_semantics_summary.json`

Main result: in the all-pairs upper-bound replay over 24,310 potential assignments, same-document false sources are only 0.016% (4 assignments), while 99.984% are different-document. Same packed row is 6.664%, but almost all same-row false sources are still different-document because compact pairs from the same source document are spread across different packed rows. Structurally, 88.76% of pairs belong to multi-pair docs, but only 4 of 3,005 packed rows contain >=2 pairs from the same document. Uniform same-doc opportunity inside batches is only 0.0223%.

Conclusion: the running `shuffled` arm is a valid different-document correspondence control. The static precomputed decoy map is stricter on per-pair source length (100% source-word match, same-doc 0.033%) but is not required before reading this panel. The current batch derangement preserves exact source-word multiset and total auxiliary charge per batch; its remaining imperfection is per-target length mismatch (mean abs 8.70 words, p95 23), not source-correspondence contamination.

## Controlled-budget substitution

The exact no-aux reference is not the adapter matched horizon plan approximation; it has now been trained under the same corrected trainer. The budget audit `data/dualview_budget_substitution/budget_substitution_audit.json` shows:

- The first 481 main updates are exactly shared between dual-view and mlm_only in batch/masked-token geometry.
- Dual-view stops at 19.021584M main words because 0.966720M auxiliary words are charged.
- mlm_only continues through updates 482–506, seeing an extra 6,344 stream rows / 978,416 words.
- The displaced main words are mostly CHILDES (241,440), Gutenberg (189,440), OpenSubtitles (176,960), qwen_pair_packed (169,980), SimpleWiki (101,440), BNC Spoken (54,880), compact-view reinvest rows (41,556), and Switchboard (2,720).
- The extra mlm_only segment contains 295 pair rows / 1,173 constituent pairs.

Therefore aligned-vs-mlm_only is a real legal-budget comparison: the auxiliary mechanism must overcome the main-stream experience it displaces. Aligned-vs-shuffled isolates true correspondence at fixed auxiliary budget.

## Mechanism evidence before official scoring

`data/dualview_training_trajectory/aligned_vs_shuffled_training_trajectory.json` shows identical accounting and lower aligned losses:

- final main loss aligned - shuffled = -0.06013
- mean main loss aligned - shuffled = -0.03166
- mean auxiliary loss aligned - shuffled = -0.06168

This is only mechanism evidence. It says true source correspondence is actually used by the optimized private pathway under matched masks/targets/exposure. It does not by itself imply broad BabyLM competence; cheap7 and fragile-family sentinels decide route value.

## Pending evaluation work

Queued or running evaluations:

- Aligned cheap7: pending, output roots `data/dualview_aligned_20m_*`.
- Shuffled cheap7: pending, output roots `data/dualview_shuffled_20m_*`.
- MLM-only cheap7: pending, output roots `data/dualview_mlm_only_20m_*`.

After all three payloads exist, run:

```bash
python -B experiments/archive/frontier_consolidation/scripts/dualview_sentinel_compare.py \
  --base experiments/archive/frontier_consolidation/data/legal20m_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_20M.json \
  --base-label reference_20M \
  --candidate aligned=experiments/archive/frontier_consolidation/data/dualview_aligned_20m_eval/per_target/dualview_aligned_20M.json \
  --candidate shuffled=experiments/archive/frontier_consolidation/data/dualview_shuffled_20m_eval/per_target/dualview_shuffled_20M.json \
  --candidate mlm_only=experiments/archive/frontier_consolidation/data/dualview_mlm_only_20m_eval/per_target/dualview_mlm_only_20M.json \
  --pairwise-candidates \
  --label dualview_panel_complete
```

Continue beyond 20M only if aligned beats both shuffled and exact mlm_only while avoiding the fragile-family damage pattern seen in scale1.75/U256 (EWoK material/spatial/quantitative, Reading, SuperGLUE-related Supplement groups). If aligned only beats shuffled but remains below mlm_only, the source correspondence is real but not worth the current legal-budget substitution; rebuild the private-pathway mechanism rather than extending unchanged.
