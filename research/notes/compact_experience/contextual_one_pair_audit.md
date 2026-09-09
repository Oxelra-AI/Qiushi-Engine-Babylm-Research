# Contextual one-pair feasibility audit

Uses only official training pool and initial and clean-materialization Qwen source metadata; no official AoA/CDI words, child curves, AoA predictions, AoA scores, or downstream eval outputs are read.

Located pairs: 37594 / 37594 (1.0000)
Methods: {'sentence_split': 37594}

## By cohort
- base: 19374 / 19374 (1.0000)
- extra_shard0: 9045 / 9045 (1.0000)
- extra_shard1: 9175 / 9175 (1.0000)

## Cap summaries
- cap 80: row_words_mean=80.38937064425174 context_words_mean=36.31850827259669 total_context_words=1365358
- cap 120: row_words_mean=120.0 context_words_mean=75.92913762834495 total_context_words=2854480
- cap 160: row_words_mean=160.0 context_words_mean=115.92913762834495 total_context_words=4358240

Output JSON: `experiments/archive/compact_experience/data/contextual_one_pair_audit/contextual_one_pair_feasibility.json`
