# contrastive entity binding design contrastive entity-binding intervention design
This note records a CPU audit and prototype for the next natural-language intervention after the state-update route weakened. It is not a training result and it does not use the pending pa87 official evaluations.
## Why the old state-update packet cannot be bundled with parent anchoring
The earlier synthesis shows that the replacement packet family strengthened source-state retention while neutral-adjusted target-update use was negative in both seeds. It also displaced the inherited ALN qwen-pair ingredient and was slightly under 100M words. Parent anchoring could preserve a parent function, but it cannot make a packet require entity-specific updating when a source-retention or recency shortcut solves much of the exposure.
## What this audit measured
The script read the validated earlier analysis plain-use packets and looked for same-pair UPDATED_USE and UNCHANGED_DISTRACTOR_USE examples with identical source sentence and target entity. It then built masked contexts by masking only the target state phrase in the final use sentence, using the actual compliant16k tokenizer when available.

Packet file counts:

| file | rows | UPDATED | RETAIN | unique pair IDs | same-pair both types |
|---|---:|---:|---:|---:|---:|
| `experiments/archive/relation_learning/data/state_use_generation/validated_state_use_packets_plainuse_full_all.jsonl` | 4657 | 2791 | 1866 | 4657 | 0 |
| `experiments/archive/relation_learning/data/state_use_generation/validated_state_use_packets_plainuse_full_balanced.jsonl` | 3732 | 1866 | 1866 | 3732 | 0 |
| `experiments/archive/relation_learning/data/state_use_generation/validated_state_use_packets_plainuse_full_train.jsonl` | 3332 | 1666 | 1666 | 3332 | 0 |
| `experiments/archive/relation_learning/data/state_use_generation/validated_state_use_packets_plainuse_full_heldout.jsonl` | 400 | 200 | 200 | 400 | 0 |

Strict same-source/same-target contrast candidates found: **0**. Rejection counts among same-pair opportunities: `{}`.

Single-packet masked-input audit:

| type | n | answer in source | answer in update | foil in source | foil in update | missing mask | cue hits | mean answer pieces | mean use-source LCS | mean use-update LCS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `UPDATED_USE` | 2791 | 0.0029 | 0.6428 | 0.4901 | 0.0666 | 0.5801 | 0.0000 | 4.5278 | 1.7248 | 2.7073 |
| `UNCHANGED_DISTRACTOR_USE` | 1866 | 0.2551 | 0.0021 | 0.0102 | 0.5557 | 0.9925 | 0.0011 | 6.9861 | 2.8215 | 0.9432 |

This table makes the scientific risk concrete. In the intended UPDATED row the correct state is normally present in the update sentence, so a recency/copy rule can help. In the intended RETAIN row the correct state is present in the original source while an irrelevant entity's new state is present in the update sentence, so a source-retention rule can help. A useful intervention must improve both rows in held-out pair combinations; one-sided gains are not evidence of operation binding.
## ALN preservation and budget direction
The inherited ALN pool exists at `experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl` with 64381 rows and 10000000 words; the audit counted 0 `qwen_aligned` rows and 64381 filler rows. Its SHA256 is `e0a3cdc20e39f2715fbdb0cfbc6c4aff61d51924c480a982049878272c5690b3`. The next intervention should add contrastive packets from filler rows, not remove this ALN structure.
## Required packet structure for the next GPU-worthy arm
For each source and queried entity, create a paired contrast with the same source sentence and same target entity:

- UPDATE row: source state S(e) is followed by an update to the target entity e, and the masked use sentence should prefer the new state U(e).
- RETAIN row: the same source state S(e) is followed by an update to a different entity d, and the masked use sentence should prefer S(e), not U(d).
- The use sentence must contain no temporal/persistence cues such as now/still/remain/current/new, and the masked state span must be the only occurrence of the answer in the use sentence.
- The two rows should be held out by source/entity combination, not merely by sentence text, so that held-out scoring tests recombination of entity selector and update operation.

The first result should be a vector, not a scalar: neutral-adjusted UPDATE margin and neutral-adjusted RETAIN margin on held-out same-source pairs. Both must move in the useful direction; an average that trades update for retention repeats the earlier analysis failure.
## Files
- summary_json: `experiments/archive/functional_learning/data/contrastive_entity_binding/contrastive_binding_audit_summary.json`
- candidate_sample_jsonl: `experiments/archive/functional_learning/data/contrastive_entity_binding/same_pair_contrast_candidates_sample.jsonl`
- note: `research/notes/functional_learning/contrastive_entity_binding_design.md`
