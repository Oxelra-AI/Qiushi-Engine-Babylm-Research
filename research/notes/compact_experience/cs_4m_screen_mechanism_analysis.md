# cs 4m residualized screen C/S 4M screen mechanism analysis

This note analyzes the materialized JSONL arms in `data/cs_4m_screen/` before any H100 training. The purpose is to determine whether the selected arms separate compositional-constraint density from recoverable-signal density rather than mostly selecting cleaner, easier-to-tokenize, or internally different text.

## Exact file-level results

- Union of selected rows: 59657 of 62,500 official 160-word rows.
- All five arms contain 25,000 rows and 4,000,000 words: True.
- C-high/S-high overlap: 17408 shared rows = 0.696 of C-high.
- C-high/C-control overlap: 0 rows.
- R-a/R-b overlap: 9930 rows = 0.397 of R-a, near the expected 0.4 for independent 40% samples.

## Score movement

- Random-reference mean C-score: 0.446205; C-high mean C-score: 0.560148; C-control mean C-score: 0.329593; S-high mean C-score: 0.531932.
- Random-reference mean S-score: 0.417830; S-high mean S-score: 0.476476; C-high mean S-score: 0.458887; C-control mean S-score: 0.374980.
- Row-level C/S correlation on the selected-row union: 0.835. By source: {"bnc_spoken.train.txt": 0.715863, "childes.train.txt": 0.830044, "gutenberg.train.txt": 0.516482, "open_subtitles.train.txt": 0.763084, "simple_wiki.train.txt": 0.761606, "switchboard.train.txt": 0.729954}.

## Surface, style, and tokenizer-compression coupling

- A source+surface+tokenizer-compression linear model explains R²=0.760 of C-score and R²=0.922 of S-score on the selected-row union.
- After removing source, row position, style/quality, and compression features, C-high residual C-score mean is 0.056136 vs random mean 0.000157; S-high residual S-score mean is 0.016091 vs random mean -0.000373.
- After additionally removing the other intended score, S-high residual S-score mean is 0.008867; C-high residual C-score mean after removing S-score is 0.028111.

## Interpretation for the next run

The JSONL files are structurally usable by the trainer, but this arm set is not yet a clean mechanism experiment. C-high and S-high share more than two thirds of their rows, S-high still has a high C-score, and both scores are substantially predictable from source-internal position, style/quality, and tokenizer-compression proxies. A positive training result from these exact arms would not identify whether the useful factor was compositional structure, recoverable lexical signal, cleaner text, tokenization ease, or an internal genre slice.

Do not send these arms to H100 as the first decisive screen. Treat them as a profile of the official pool. The next construction should use the saved row-level union features here and, preferably, a full-pool feature table to build residualized or nearest-neighbor matched arms: C-high with a confound-matched lower-C reference, S-high selected on non-structural recoverable signal after matching C-score and tokenizer compression, and a C/S-separated arm with low overlap by construction.


Detailed JSON: `experiments/archive/compact_experience/data/cs_4m_screen/mechanism_analysis.json`

Row-level feature CSV: `experiments/archive/compact_experience/data/cs_4m_screen/row_feature_union.csv`
