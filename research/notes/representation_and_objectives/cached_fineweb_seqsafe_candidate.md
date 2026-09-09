# cached fineweb quality audit seq256-safe cached FineWeb-Edu single-document candidate

Quality-filtered single-doc cached FineWeb text re-chunked to 112-word rows so baseline16k seq256 truncation is near-zero and matches the official control block. Not trained evidence.

- Clean-Qwen pair block: 12,236 rows, 1,656,800 words (16.57%).
- Seq-safe FineWeb block: 10,958 rows, 1,227,296 words (12.27%); unique docs 5,566; dropped multi-doc 7,718, dropped quality 74.
- Official length-matched control block: 10,958 rows, 1,227,296 words.
- Shared official tail: 1 rows, 7,115,904 words (71.16%).
- Treatment/control preserve identical clean-Qwen rows, identical tail, exact 10M words, identical row-length sequence.

Verification JSON: `experiments/archive/representation_and_objectives/training/data/cached_fineweb_seqsafe_candidate/materialization_verification.json`

Metadata JSON: `experiments/archive/representation_and_objectives/training/data/cached_fineweb_seqsafe_candidate/materialization_metadata.json`
