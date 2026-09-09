# source attested fluent bridge prototype readout source-attested fluent bridge prototype

This is a constructibility probe, not a BabyLM training result. It tests whether Qwen3.5 can produce fluent compact-like views whose content lemmas are all attested in the source.

Prompts: 96; outputs: 96. Prototype accepted 58/96 (0.604); zero unsupported content lemmas 87/96 (0.906).

| regime | n | accepted | accept rate | zero unsupported lemma rate | gen/natural words mean | token/natural mean | relation retained |
|---|---:|---:|---:|---:|---:|---:|---:|
| anchor_gapfill | 48 | 30 | 0.625 | 0.875 | 1.198 | 1.128 | 0.938 |
| lexclosed_compress | 48 | 28 | 0.583 | 0.938 | 1.199 | 1.113 | 0.875 |

Automatic lexical closure is not semantic truth. Review sample: `source_attested_fluent_bridge_review_sample.jsonl`.

Summary JSON: `experiments/archive/frontier_consolidation/data/source_attested_fluent_bridge_prototype/source_attested_fluent_bridge_generation_summary.json`
