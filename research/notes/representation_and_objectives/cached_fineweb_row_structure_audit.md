# cached fineweb broad source candidate cached FineWeb row-structure audit

Input: `experiments/archive/initial_model_studies/data/fineweb_relation_matched_3M/fineweb_random_quality_3000000w.jsonl`

Rows/words: 18,750 / 3,000,000.

Document-ID count distribution: `{'1': 11032, '2': 7473, '3': 245}`. Multi-doc rows: 7,718 (41.16%).

Single-document rows: 11,032 (1,765,120 words, 58.84% of cached words).

Startish rows by heuristic: 2,924 (15.59%); single-doc startish words: 265,440.

Interpretation: use the full 3M cached arm only as a broad but noisy fallback. A cleaner candidate should use the single-document subset (~1.77M words) plus matched official filler, because same-source semantic-view work already showed that row-boundary/coherence confounds can matter.

JSON: `experiments/archive/representation_and_objectives/training/data/cached_fineweb_broad_source_candidate/fineweb_row_structure_audit.json`
