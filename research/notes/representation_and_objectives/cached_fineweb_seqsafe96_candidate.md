# cached fineweb quality audit seq256-safe cached FineWeb-Edu single-document candidate

Quality-filtered single-doc cached FineWeb text re-chunked to 96-word rows so baseline16k seq256 truncation is near-zero and matches the official control block. Not trained evidence.

- Clean-Qwen pair block: 12,236 rows, 1,656,800 words (16.57%).
- Seq-safe FineWeb block: 21,916 rows, 1,753,280 words (17.53%); unique docs 5,566; dropped multi-doc 7,718, dropped quality 74.
- Official length-matched control block: 21,916 rows, 1,753,280 words.
- Shared official tail: 41,187 rows, 6,589,920 words (65.90%).
- Treatment/control preserve identical clean-Qwen rows, identical tail, exact 10M words, identical row-length sequence.

Verification JSON: `experiments/archive/representation_and_objectives/training/data/cached_fineweb_seqsafe96_candidate/materialization_verification.json`

Metadata JSON: `experiments/archive/representation_and_objectives/training/data/cached_fineweb_seqsafe96_candidate/materialization_metadata.json`
