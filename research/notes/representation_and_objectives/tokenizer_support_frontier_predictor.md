# family specific tokenizer predictor — Legal tokenizer support/segmentation frontier predictor

CPU-only first-order predictor of the full nine-column official vector for candidate legal tokenizers, calibrated on the two measured anchors (legal16k, legal40k) and the tokenizer support spectrum pool-support drivers. Trains and evaluates no model.

- Self-check reproduces both anchors exactly: **True**
- Anchors: legal16k Overall 40.8640 (seg 1.4669, mass<50 0.0020); legal40k Overall 40.7804 (seg 1.3943, mass<50 0.0413)
- Column attribution: seg-driven ['BLiMP', 'EWoK', 'Supplement']; support-driven ['Entity', 'GlobalPIQA']; flat-held ['AoA', 'COMPS', 'Reading', 'SuperGLUE']

## Predicted Overall by candidate

| tokenizer | vocab | tok/word | mass<50 | pred Overall | vs 41.8 | vs 40k mean |
|---|---:|---:|---:|---:|---:|---:|
| legal_a01_16k | 16384 | 1.4669 | 0.0020 | 40.8640 | -0.9360 | +0.0836 |
| legal_byte_bpe_40k | 40000 | 1.3943 | 0.0413 | 40.7804 | -1.0196 | +0.0000 |
| legal_byte_bpe_40k_minfreq25 | 29529 | 1.4139 | 0.0286 | 40.8359 | -0.9641 | +0.0555 |
| legal_byte_bpe_40k_minfreq50 | 19609 | 1.4484 | 0.0031 | 40.9854 | -0.8146 | +0.2051 |
| legal_byte_bpe_24k | 24576 | 1.4281 | 0.0183 | 40.8939 | -0.9061 | +0.1135 |
| legal_byte_bpe_32k | 32768 | 1.4066 | 0.0336 | 40.8109 | -0.9891 | +0.0306 |

- **minfreq25 predicted Overall = 40.8359** (band ±0.3642); margin vs 41.8 = -0.9641
- minfreq50 predicted Overall = 40.9854
- best predicted candidate: legal_byte_bpe_40k_minfreq50 at 40.9854
- crosses 41.8 within ±0.3642 band: **False**

## How to read this for the next H100 route

If the best support-floored candidate + uncertainty band stays below 41.8, a pure tokenizer support-floor run is unlikely to cross the frontier by itself, and the next expensive route must ADD a distinct factor (depth 12x384 — already running seed43022 — masking curriculum WWM->token, or a sharper objective) rather than another point on the tokenizer axis. If it crosses within band, the support-floor tokenizer becomes the strongest single-factor compliant repair and justifies one H100 pair.

JSON: `experiments/archive/representation_and_objectives/data/tokenizer_support_frontier_predictor/tokenizer_support_frontier_predictor.json`
CSV: `experiments/archive/representation_and_objectives/data/tokenizer_support_frontier_predictor/predicted_tokenizer_vectors.csv`
