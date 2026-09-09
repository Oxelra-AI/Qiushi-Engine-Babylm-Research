# earlier analysis state-margin scorer result

Created UTC: 2026-09-06T21:23:33Z

## Instrument

Each packet is scored under T=source+true update, U=source+swapped compatible update, and N=source only. Candidate phrases are source_state, original_new_state, and swapped_new_state. Phrase scores use one-token-at-a-time pseudo-log-likelihood normalized by candidate token count; content-token margins are written separately. The primary `template` slot is an explicit current-state query; optional `use` slot scoring keeps the generated use-sentence frame when the state span is identifiable. Higher margins mean the first named state phrase is more probable in the scored slot.

This instrument intentionally preserves raw T/U/N values for base and intervention; a movement under N alone is surface/generator-format movement, not evidence of update use. The small training dose means tenths of a nat are scientifically meaningful; packet-level paired deltas and SEs are the relevant quantities.

## Main paired deltas

For UPDATED_USE, `delta_T_update_use` is intervention-base movement of original_new_state over source_state under the true update; `delta_U_retention` is source_state over swapped_new_state when the update belongs to another entity. For DISTRACTOR, T and U retention rows compare source_state against original or swapped distractor states.

| seed | ck | set | slot | type | n | T update | se | U retention | se | update+Uret avg | se | T retention | se | retention avg | se |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 43122 | chck_86M | extended_nontrain | template | UNCHANGED_DISTRACTOR_USE | 200 | NA | NA | +0.0254 | +0.0708 | NA | NA | +0.1170 | +0.0626 | +0.0712 | +0.0537 |
| 43122 | chck_86M | extended_nontrain | template | UPDATED_USE | 1125 | -0.1147 | +0.0336 | +0.0207 | +0.0323 | -0.0470 | +0.0172 | NA | NA | NA | NA |
| 43122 | chck_86M | balanced_heldout | template | UNCHANGED_DISTRACTOR_USE | 200 | NA | NA | +0.0254 | +0.0708 | NA | NA | +0.1170 | +0.0626 | +0.0712 | +0.0537 |
| 43122 | chck_86M | balanced_heldout | template | UPDATED_USE | 200 | -0.1621 | +0.0810 | +0.0997 | +0.0820 | -0.0312 | +0.0415 | NA | NA | NA | NA |
| 43122 | final | extended_nontrain | template | UNCHANGED_DISTRACTOR_USE | 200 | NA | NA | +0.0300 | +0.0702 | NA | NA | +0.1307 | +0.0612 | +0.0803 | +0.0535 |
| 43122 | final | extended_nontrain | template | UPDATED_USE | 1125 | -0.0863 | +0.0333 | +0.0021 | +0.0320 | -0.0421 | +0.0170 | NA | NA | NA | NA |
| 43122 | final | balanced_heldout | template | UNCHANGED_DISTRACTOR_USE | 200 | NA | NA | +0.0300 | +0.0702 | NA | NA | +0.1307 | +0.0612 | +0.0803 | +0.0535 |
| 43122 | final | balanced_heldout | template | UPDATED_USE | 200 | -0.1270 | +0.0805 | +0.0910 | +0.0780 | -0.0180 | +0.0395 | NA | NA | NA | NA |

## Output files

- plan: `experiments/archive/relation_learning/data/state_margin_scores_seed43122/score_plan.json`
- encoding_metadata: `experiments/archive/relation_learning/data/state_margin_scores_seed43122/encoding_metadata.json`
- swap_assignments: `experiments/archive/relation_learning/data/state_margin_scores_seed43122/swap_assignments.csv`
- candidate_phrase_score_rows: `experiments/archive/relation_learning/data/state_margin_scores_seed43122/candidate_phrase_score_rows.csv`
- raw_TUN_margin_rows: `experiments/archive/relation_learning/data/state_margin_scores_seed43122/raw_TUN_margin_rows.csv`
- raw_TUN_margin_summary: `experiments/archive/relation_learning/data/state_margin_scores_seed43122/raw_TUN_margin_summary.csv`
- paired_arm_delta_rows: `experiments/archive/relation_learning/data/state_margin_scores_seed43122/paired_arm_delta_rows.csv`
- paired_arm_delta_summary: `experiments/archive/relation_learning/data/state_margin_scores_seed43122/paired_arm_delta_summary.csv`
- relation_composite_delta_rows: `experiments/archive/relation_learning/data/state_margin_scores_seed43122/relation_composite_delta_rows.csv`
- relation_composite_delta_summary: `experiments/archive/relation_learning/data/state_margin_scores_seed43122/relation_composite_delta_summary.csv`
- note: `research/documents/relation_learning/data/state_margin_scores_seed43122/state_margin_score_note.md`

## Notes for interpretation

Use the extended set for UPDATED power and the balanced set when comparing UPDATED and DISTRACTOR with equal denominators. Report state-type and conflict-stratified rows from `relation_composite_delta_summary.csv` before mapping to official Entity, because the probe distribution is dominated by other relation states rather than Entity-like containment/location.
