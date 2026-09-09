# babylm2026 surface refresh — INITIAL_MODEL_STUDIES core-competence materialization inspection

Source directory: `experiments/archive/initial_model_studies/data/core_competence_screen`.

## File inventory

| file | rows | words | sources | score fields | content group |
|---|---:|---:|---|---|---|
| `composition_selected_curriculum.jsonl` | 195 | 99840 | bnc_spoken.train.txt:188, childes.train.txt:7 | comp, curr | G1 |
| `composition_selected_shuffled.jsonl` | 195 | 99840 | bnc_spoken.train.txt:188, childes.train.txt:7 | comp, curr | G1 |
| `official_curriculum.jsonl` | 25000 | 4000000 | childes.train.txt:7090, gutenberg.train.txt:6418, open_subtitles.train.txt:5693 | curr | G3 |
| `official_flat_curriculum.jsonl` | 195 | 99840 | bnc_spoken.train.txt:120, childes.train.txt:75 | comp, curr | G2 |
| `official_flat_shuffled.jsonl` | 195 | 99840 | bnc_spoken.train.txt:120, childes.train.txt:75 | comp, curr | G2 |
| `official_random_order_a.jsonl` | 25000 | 4000000 | childes.train.txt:7090, gutenberg.train.txt:6418, open_subtitles.train.txt:5693 | curr | G3 |
| `official_random_order_b.jsonl` | 25000 | 4000000 | childes.train.txt:7090, gutenberg.train.txt:6418, open_subtitles.train.txt:5693 | curr | G3 |
| `official_shuffled.jsonl` | 25000 | 4000000 | childes.train.txt:7090, gutenberg.train.txt:6418, open_subtitles.train.txt:5693 | curr | G3 |

## Research reading

The three files `official_random_order_a.jsonl`, `official_random_order_b.jsonl`, and `official_curriculum.jsonl` are the true earlier analysis same-content order arms: each has 25,000 rows and 4,000,000 words, and they share the same text multiset. Their only intended difference is row order.

The four files named `official_flat_*` and `composition_selected_*` are much smaller: each has 195 rows and 99,840 words, mostly 512-word rows. They are not described by `metadata.json` and should be treated as earlier small materialization leftovers until their construction code and source/length matching are reconstructed. They cannot directly answer the earlier analysis 4M compositional-selection question.

Therefore earlier analysis only closed the simple same-content ordering version. The content-selection mechanism remains unresolved because the intended 4M source/length-matched compositional-selection arms were not actually materialized in the inspected metadata set.

Full JSON: `experiments/archive/compact_experience/data/initial_model_core_screen_inspection.json`.
