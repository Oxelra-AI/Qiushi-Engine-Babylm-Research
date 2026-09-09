# cached fineweb broad source candidate cached FineWeb-Edu broad-source candidate

This is a prepared data pair, not trained evidence. It should be used only if the running semantic-view contrast does not provide a strong enough same-source transformation signal or if a separate broad factual-source test becomes scientifically preferable.

## Construction

- Shared clean-Qwen pair block: 12,236 rows, 1,656,800 words (16.57%).
- Treatment-only cached FineWeb-Edu random-quality block: 18,750 rows, 3,000,000 words (30.00%).
- Control block: official BabyLM filler chunked to the exact FineWeb row-length sequence, 18,750 rows and 3,000,000 words.
- Shared official tail after the replacement block: 33,395 rows, 5,343,200 words (53.43%).
- Treatment and control have identical clean-Qwen rows, identical tail text, exact 10M words, and the same row-length sequence.

## Interpretation if later trained

This tests broad public factual-source replacement, not generated paraphrastic variation. It inherits INITIAL_MODEL_STUDIES evidence that simple FineWeb relation filtering was unstable at small scale, so a later run should be interpreted against that history and only continued if official-compatible downstream scores improve the hard EWoK/Entity/COMPS/GlobalPIQA region without destroying the clean-Qwen strengths.

Verification JSON: `experiments/archive/representation_and_objectives/training/data/cached_fineweb_broad_source_candidate/materialization_verification.json`

Metadata JSON: `experiments/archive/representation_and_objectives/training/data/cached_fineweb_broad_source_candidate/materialization_metadata.json`

Sample rows JSON: `experiments/archive/representation_and_objectives/training/data/cached_fineweb_broad_source_candidate/sample_rows.json`
