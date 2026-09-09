# real stream training readiness real-stream training-side evidence and next comparison readiness

## What finished

The corrected real stream policy appended structural-policy real-stream training has completed for both objectives under `data/real_stream_policy_comparison/`:

- `inherited_wwm`: ordinary row-keyed 15% whole-word masking on the same structural compact+topup stream.
- `correspondence_focus`: ordinary WWM on non-Qwen rows, with source-visible second-view content-word targets on `qwen_pair_packed` rows.

Both trained for 80 macro-updates from the trusted coherent86 parent using the same 3.1628M-word prefix of `compact_structural_reinvest_reference_tail_wordpaced_segments.jsonl`. The prefix contains 20,545 rows, 3,848 Qwen packed rows, 11,827 source/view pair segments, 2,057 compact-modified Qwen rows, and 2,681 compact-modified pair occurrences. The early prefix has zero top-up rows because this is the appended-topup stream; it primarily tests compact substitution plus objective translation, not reinvested support.

The produced checkpoints exist:

- `data/real_stream_policy_comparison/inherited_wwm/checkpoints/update_0080/`
- `data/real_stream_policy_comparison/correspondence_focus/checkpoints/update_0080/`

Training synthesis is saved at `data/real_stream_training_synthesis/training_synthesis.json`.

## Credit-mass facts that matter for interpretation

A completed target-credit audit for the appended structural prefix is saved at `data/real_stream_credit_audit/available_credit_audit_summary.json` and `structural_appended_per_update_credit.csv`.

Across 80 updates:

- inherited WWM target tokens: 697,958
- inherited WWM Qwen target tokens: 103,775, i.e. 14.87% of WWM target mass
- correspondence-focus total target tokens: 622,769
- correspondence-focus Qwen target tokens: 28,586, i.e. 4.59% of focus target mass
- focus total targets / WWM total targets: 0.8923
- focus Qwen targets / WWM Qwen targets: 0.2755

Thus `correspondence_focus` does not simply intensify Qwen correspondence; it protects the source from masking and targets second-view content words, but also reduces total Qwen loss mass by about 72.5%. If it preserves broad behavior but fails to move common-target source-following, insufficient Qwen pressure remains a concrete explanation. If it improves preservation relative to WWM, some of that may come from less Qwen pressure rather than better correspondence use.

Only about 19.94% of selected focus groups in the structural stream are `unverified_structural_shortening_candidate`; about 80.05% are inherited/not-in-policy/keep-current/exact-source-return or other non-admitted segments. This means the current appended-tail result tests a mixed policy: focused use of existing inherited paired experience plus some generated structural shortenings. It is not a pure compaction experiment.

## Clean unchanged-text control prepared

To separate objective value from generated compaction, I built an unchanged inherited source/current-rewrite tail with Qwen segment metadata only:

- script: `scripts/annotate_unchanged_qwen_tail.py`
- tail: `data/unchanged_pair_segments_tail/reference_tail_unchanged_qwen_segments.jsonl`
- summary: `data/unchanged_pair_segments_tail/tail_build_summary.json`

It preserves exactly the reference tail text and word count: 90,609 rows, 13,994,705 tail words, legal total 100,000,000 words from coherent86. All 17,107 Qwen rows matched metadata, with 52,572 pair segments; there are zero reconstruction mismatches and zero bad segment offsets. A dry run of `real_stream_train.py` on this unchanged tail succeeded under `data/unchanged_focus_control_dryrun/`: the first macro has 33 Qwen rows, 0 compact-modified rows, and correspondence-focus selects 254 Qwen target tokens from 194 unchanged-inherited-current groups.

This control should be the next comparison if the pending evaluation suggests that objective allocation over existing paired experience may matter more than the generated structural shortening policy. It can test whether useful correspondence learning can be improved without uncertain generated compaction or reinvestment.

## Pending learner evidence

The planned evaluation compares the two corrected real stream policy checkpoints with the earlier analysis common-target probe and Cheap7. Its results were still pending and cannot be inferred from training losses. The next major investment should depend on the joint outcome:

- common-target source-following movement, including source-original/source-altered/no-source decomposition if available;
- fast Cheap7 broad preservation relative to coherent86 and the ordinary-tail reference;
- whether correspondence-focus gains, if any, are plausibly due to source-visible correspondence targets or merely lower Qwen pressure.

If evaluation shows no broad damage and meaningful common-target movement, longer/interleaved real-stream training becomes worth testing. If evaluation shows broad preservation but little source-following movement, calibrate a higher Qwen target-mass focused objective before concluding source-visible real-stream learning failed. If evaluation shows damage, inspect which Cheap7 components move and compare with WWM to decide whether the structural generated policy, the objective, or the late-continuation schedule is responsible.
