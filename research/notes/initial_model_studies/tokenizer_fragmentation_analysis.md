# bert8x512 b256 available coordinate — Tokenizer fragmentation analysis

Evidence JSON: `experiments/archive/initial_model_studies/data/tokenizer_fragmentation_analysis.json`

| tokenizer | probe mean subtokens | tokens/word (20k lines) | frac lines >256 | est words covered @256 |
|---|---:|---:|---:|---:|
| baseline16k | 1.150 | 1.407 | 0.000 | 16.0 |
| official40k | 1.053 | 1.208 | 0.000 | 16.0 |

Group stats and per-word fragmentation are in the JSON.
