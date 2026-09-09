# Realized State-Use Construction

Scientific status: historical construction and control record. Successful construction is not evidence of downstream improvement.

The corrected training interface was checked for 100 optimizer updates against its reference. The record reported zero mismatches: first loss 9.837543487548828, update100 loss 6.302092552185059, and update100 cumulative exposure 3,952,098 words. These are preserved verification values, not newly measured losses.

Qwen3.5-9B generation produced 11,348 UPDATED_USE outputs and 11,349 DISTRACTOR_USE outputs. Validation accepted 2,791 and 1,866 respectively. Balancing retained 3,732 packets, 1,866 per type; 3,332 became training packets and 400 held-out packets, with equal type counts in each split.

The retained balanced set had no source/use LCS length at least 6, no content Jaccard at least 0.8, and no accepted use-sentence cues. Content Jaccard mean/median were 0.2033/0.1765; source/use LCS mean/median/max were 2.32/2/5. Cue filtering had rejected 1,913 raw outputs. These tests reduce particular shortcuts, but do not establish that the desired state-update computation was learned.

## Materialized Stream

- 3,332 packets addressed 3,299 unique pair IDs; 33 pairs exceeding 256 tokens reverted to their original material per pass.
- There were zero metadata misses and zero text-reconstruction mismatches.
- Per pass, 2,996 rows and 3,299 pairs were replaced. Across ten passes the packet counts were 16,490 UPDATED_USE and 16,500 DISTRACTOR_USE.
- Mean word change was -3.0 per pair; total change was -90,080 words, or -0.09%.
- Mean/max token length were 181.5/256; 370 cases exceeded 240 tokens and 90 exceeded 250.
- The final stream contained 647,400 rows and 99,909,920 words.

The recorded recipe used seed43022, 99,909,920 maximum word exposure, a 2,529-update learning-rate schedule, adapter bottleneck 128 and scale 1.75, and an 8-by-480 DeBERTa-v2 model of about 35.5M parameters. At this record's boundary, the full manual sample review and held-out state-type labeling were still incomplete. Later evaluations and the negative route synthesis remain separate.

Evidence: [row materializer](../../../experiments/archive/relation_learning/scripts/qwen_row_materializer.py), [subsequent state-update route synthesis](state_update_route_synthesis.md). The training-identity measurements above are historical checks reported by the construction record, not a new verification of the training interface.

Related methodology: [corrected macro-update bridge trainer](../../../experiments/archive/functional_learning/scripts/corrected_bridge_trainer.py). That implementation documents a separate relation-overlay objective and deterministic ordinary corruption; it is not a substitute for the historical training-identity check above.
