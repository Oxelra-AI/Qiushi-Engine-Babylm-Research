# earlier analysis causal GPT token/effective-exposure audit

Tokenizer SHA: `e6723383f9642945134a09a9641d52e595624d3c5a99622e2a47e4d912378102`

Both arms are legal-word matched: 10,000,000 words per epoch, 10 epochs = 100,000,000 charged words. The quantities below describe the tokenizer/chunk stream actually seen by the GPT trainer.

| quantity | compact | repeat | compact-repeat | rel. |
|---|---:|---:|---:|---:|
| raw tokens incl EOS / epoch | 14,587,760 | 14,556,070 | 31,690 | 0.2177% |
| active 256-token positions / epoch | 14,587,648 | 14,555,904 | 31,744 | 0.2181% |
| chunks / epoch | 56,983 | 56,859 | 124 | 0.2181% |
| token/word | 1.458776 | 1.455607 | 0.003169 | 0.2177% |
| total updates at batch 128 | 4,460 | 4,450 | 10 | 0.2247% |
| total updates at batch 256 | 2,230 | 2,230 | 0 | 0.0000% |

## Interpretation
The causal replication should be interpreted as a word-budget-matched experiment with a small tokenizer/chunk exposure difference intrinsic to the different texts. A compact advantage must be read across checkpoints and families, not reduced to this token-count delta or to one column spike.

JSON: `experiments/archive/frontier_consolidation/data/reciprocal_causal_token_exposure_audit_dosematched_shuffle/causal_token_exposure_audit.json`
