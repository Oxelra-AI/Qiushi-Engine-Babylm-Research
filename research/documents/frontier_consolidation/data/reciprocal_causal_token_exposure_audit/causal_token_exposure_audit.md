# earlier analysis causal GPT token/effective-exposure audit

Tokenizer SHA: `e6723383f9642945134a09a9641d52e595624d3c5a99622e2a47e4d912378102`

Both arms are legal-word matched: 10,000,000 words per epoch, 10 epochs = 100,000,000 charged words. The quantities below describe the tokenizer/chunk stream actually seen by the GPT trainer.

| quantity | compact | repeat | compact-repeat | rel. |
|---|---:|---:|---:|---:|
| raw tokens incl EOS / epoch | 14,600,013 | 14,535,701 | 64,312 | 0.4424% |
| active 256-token positions / epoch | 14,599,936 | 14,535,680 | 64,256 | 0.4421% |
| chunks / epoch | 57,031 | 56,780 | 251 | 0.4421% |
| token/word | 1.460001 | 1.453570 | 0.006431 | 0.4424% |
| total updates at batch 128 | 4,460 | 4,440 | 20 | 0.4505% |
| total updates at batch 256 | 2,230 | 2,220 | 10 | 0.4505% |

## Interpretation
The causal replication should be interpreted as a word-budget-matched experiment with a small tokenizer/chunk exposure difference intrinsic to the different texts. A compact advantage must be read across checkpoints and families, not reduced to this token-count delta or to one column spike.

JSON: `experiments/archive/frontier_consolidation/data/reciprocal_causal_token_exposure_audit/causal_token_exposure_audit.json`
