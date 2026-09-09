# cached fineweb broad source candidate cached FineWeb-Edu single-document candidate

This is a cleaner broad-factual-source candidate, not trained evidence. It drops cached FineWeb rows whose 160-word chunk crosses document IDs, reducing the packing confound found in the raw cached 3M file.

- Clean-Qwen pair block: 12,236 rows, 1,656,800 words (16.57%).
- Cached FineWeb single-doc block: 11,032 rows, 1,765,120 words (17.65%); unique docs 5,587.
- Official length-matched control block: 11,032 rows, 1,765,120 words.
- Shared official tail: 41,113 rows, 6,578,080 words (65.78%).
- Treatment/control preserve identical clean-Qwen rows, identical tail, exact 10M words, and identical row-length sequence.

Interpretation if later trained: this is a lower-confound public FineWeb-Edu source-distribution test. It is smaller than the raw cached 3M arm (1.765M words instead of 3M) and still comes from INITIAL_MODEL_STUDIES cached material, because live HF streaming is currently unavailable.

Verification JSON: `experiments/archive/representation_and_objectives/training/data/cached_fineweb_single_doc_candidate/materialization_verification.json`

Metadata JSON: `experiments/archive/representation_and_objectives/training/data/cached_fineweb_single_doc_candidate/materialization_metadata.json`

Sample rows: `experiments/archive/representation_and_objectives/training/data/cached_fineweb_single_doc_candidate/sample_rows.json`
