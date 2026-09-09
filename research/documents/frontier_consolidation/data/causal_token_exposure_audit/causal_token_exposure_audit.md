# earlier analysis causal GPT token/effective-exposure audit

Tokenizer SHA: `e6723383f9642945134a09a9641d52e595624d3c5a99622e2a47e4d912378102`

Both arms are legal-word matched: 10,000,000 words per epoch, 10 epochs = 100,000,000 charged words. The quantities below describe the tokenizer/chunk stream actually seen by the GPT trainer.

| quantity | compact | repeat | compact-repeat | rel. |
|---|---:|---:|---:|---:|
| raw tokens incl EOS / epoch | 14,587,969 | 14,555,825 | 32,144 | 0.2208% |
| active 256-token positions / epoch | 14,587,904 | 14,555,648 | 32,256 | 0.2216% |
| chunks / epoch | 56,984 | 56,858 | 126 | 0.2216% |
| token/word | 1.458797 | 1.455583 | 0.003214 | 0.2208% |
| total updates at batch 128 | 4,460 | 4,450 | 10 | 0.2247% |
| total updates at batch 256 | 2,230 | 2,230 | 0 | 0.0000% |

## Interpretation
The causal replication should be interpreted as a word-budget-matched experiment with a small tokenizer/chunk exposure difference intrinsic to the different texts. A compact advantage must be read across checkpoints and families, not reduced to this token-count delta or to one column spike.

JSON: `experiments/archive/frontier_consolidation/data/causal_token_exposure_audit/causal_token_exposure_audit.json`
