# dualview panel readiness and budget dual-view control semantics summary

## Batch derangement currently used by trainer
Assignments (all-pairs upper bound): `24310`
same_doc: `0.0165%`; same_row: `6.6639%`; different_doc: `99.9835%`
same-doc opportunity if source assignment were uniform inside batches: `0.0223%`
source-word mismatch mean abs `8.696092143150967`; p95 `23`; exact source-word match `3.7186%`

## Static precomputed decoy map (not used by current trainer)
assignments: `12155`; same_doc `0.0329%`; exact source-word match `100.0000%`; mean abs length delta `0`

## Document layout
pairs in multi-pair documents: `88.76%`, but packed rows with >=2 pairs from same document: `4` / `3005`.

## Decision
same-doc contamination in the actual batch-derangement control is structurally negligible (0.016% in all-pairs upper-bound; same-doc opportunity if uniform source assignment is also tiny), so the already-running shuffled arm is a valid different-document correspondence control. Length mismatch is real, but the batch derangement preserves the exact source-word multiset and total auxiliary charge per batch; it is a secondary fairness issue for small positive effects, not the contamination failure mode.

JSON: `experiments/archive/frontier_consolidation/data/dualview_control_semantics/control_semantics_summary.json`
