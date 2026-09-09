# Wikipedia simplification natural-restatement probe

The primary instrument contains **1,200** WikiLarge-clean source→simplification pairs (low=400, medium=400, high=400) and **10,800** T/U/N records. Each pair supplies one source-token-ID-present and two source-token-ID-absent single-token targets, so denominator differences do not drive bin comparisons.

## Scientific role

T presents the true complex Wikipedia source, U an exact-token-length unrelated WikiLarge source, and N an exact-token-length ordinary BabyLM source-slot segment. The same simplified target and mask position are used in all three conditions. T−U isolates related-source use within Wikipedia register; T−N anchors that effect against ordinary source-slot text; U−N measures register or unrelated-neighbor effects.

## Quality and holdout

Pairs pass lexical, introduced-name, numeric-consistency, encoding-noise, length, semantic-similarity (cosine ≥0.65), and directional entailment (source→target probability ≥0.8) filters. Exact tokenizer-ID screening found no selected target sentence or full T body in CLEAN, VIEW, or REPEAT. Source-only exposure is reported rather than excluded. The dataset has no article/title/id field, so only surface holdout—not article-level holdout—is possible.

## Target denominators

| bin | pairs | overlap targets per condition | non-overlap targets per condition |
|---|---:|---:|---:|
| low | 400 | 400 | 800 |
| medium | 400 | 400 | 800 |
| high | 400 | 400 | 800 |

The WikiLarge probe has 1,200 overlap and 2,400 non-overlap targets per condition, versus 689 overlap and 1422 non-overlap targets in the primary CHILDES probe. More importantly, WikiLarge’s source-token-absent targets occur inside an aligned simplification relation rather than merely following an adjacent utterance.

## Files and scoring

Use `wikipedia_simplification_pairs.jsonl`, `wikipedia_simplification_probe_records.jsonl`, `probe_stats.json`, and `validation_results.json` as the primary bundle. Inspect `manual_sample.md` before scoring. `dataset_provenance.json` records the pinned public revision and license.

Run `score_wikipedia_simplification_probe.py --plan-only` first, then invoke it on the desired DeBERTa arms. The full command is recorded in `Research_Report.md`.
