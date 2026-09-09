# fineweb high anchor slice FineWeb source-by-rewrite asset repair

No generation, training, or evaluation was launched. The purpose is to make the next source-by-rewrite experiment sharper if the active SimpleWiki semantic-view contrast is weak.

## Why this repair was needed
Peer earlier analysis produced a useful factual-source asset, but direct inspection in frontier_consolidation still found citation metadata rows (`Volume/Issue/Page/doi`), literary quoted fragments, and weak publication-notice rows. A generation slice on those examples would spend H100 time on source noise rather than on the scientific question: whether broad factual sentences become more learnable when paired with faithful simplifications.

## Repaired tiers
| tier | rows | words | unique docs | mean words | p95 words | use |
|---|---:|---:|---:|---:|---:|---|
| high_precision_from_strict_factual | 9,517 | 214,771 | 3,913 | 22.57 | 36 | recommended first Qwen faithfulness slice |
| medium_repaired_from_strict_plus_balanced | 21,465 | 469,887 | 5,018 | 21.89 | 38 | fallback if high precision is too small |

High-precision full source budget: source words 214,771; paired source+rewrite mass about 408,065 words at rewrite/source ratio 0.90 (range 375,849--440,281 for ratios 0.75--1.05).

## Output files
- high_precision_sources: `experiments/archive/frontier_consolidation/data/fineweb_source_by_rewrite_repair/fineweb_factual_high_precision_sources.jsonl`
- medium_repaired_sources: `experiments/archive/frontier_consolidation/data/fineweb_source_by_rewrite_repair/fineweb_factual_medium_repaired_sources.jsonl`
- full_high_precision_prompts: `experiments/archive/frontier_consolidation/data/fineweb_source_by_rewrite_repair/fineweb_factual_high_precision_simplification_prompts_all.jsonl`
- pilot2048_prompts: `experiments/archive/frontier_consolidation/data/fineweb_source_by_rewrite_repair/fineweb_factual_high_precision_simplification_prompts_pilot2048.jsonl`
- pilot4096_prompts: `experiments/archive/frontier_consolidation/data/fineweb_source_by_rewrite_repair/fineweb_factual_high_precision_simplification_prompts_pilot4096.jsonl`
- metadata: `experiments/archive/frontier_consolidation/data/fineweb_source_by_rewrite_repair/fineweb_source_by_rewrite_repair_metadata.json`
- samples: `experiments/archive/frontier_consolidation/data/fineweb_source_by_rewrite_repair/fineweb_source_by_rewrite_repair_samples.json`
- note: `research/notes/frontier_consolidation/fineweb_source_by_rewrite_repair.md`

## Experimental meaning
If the pending SimpleWiki semantic-view treatment beats packet-local repetition, this repaired FineWeb asset is secondary. If the SimpleWiki effect is weak, the next discriminating H100 action should be a small generation-and-audit slice from `pilot2048_prompts`, followed only then by a matched materialization: same FineWeb source sentences plus accepted simplifications versus the same source sentences with length-matched same-source repetition and identical official filler. This preserves the source-breadth question while isolating the effect of a faithful second view.
