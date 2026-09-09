# earlier analysis causal GPT training accounting audit

## compact

Status: `CAUSAL_GPT_TRAINING_COMPLETE`; params: 30156480; final loss: 3.292451.

Pool SHA: `fa2216ce8f376767f900447e1e44e1932177583229119223d0554a78414af1ed`; tokenizer SHA: `e6723383f9642945134a09a9641d52e595624d3c5a99622e2a47e4d912378102`.

Legal charged words: 100000000; steps: 2230; epochs: 10; active tokens/epoch: 14587904; dropped tail tokens/epoch: 65.

| endpoint | words | step | loss | model exists |
|---|---:|---:|---:|---|
| chck_20M | 20000000 | 446 | 4.507562 | True |
| chck_50M | 50000000 | 1115 | 3.607763 | True |
| chck_70M | 70000000 | 1561 | 3.410655 | True |
| chck_82M | 82021620 | 1829 | 3.312122 | True |
| chck_90M | 90000000 | 2007 | 3.311631 | True |
| chck_100M | 100000000 | 2230 | 3.292451 | True |

## repeat

Status: `CAUSAL_GPT_TRAINING_COMPLETE`; params: 30156480; final loss: 3.28013.

Pool SHA: `e40b3a8978a00897020e1075557b658196c1aeb2a0136b4cba871ccdc334a61d`; tokenizer SHA: `e6723383f9642945134a09a9641d52e595624d3c5a99622e2a47e4d912378102`.

Legal charged words: 100000000; steps: 2230; epochs: 10; active tokens/epoch: 14555648; dropped tail tokens/epoch: 177.

| endpoint | words | step | loss | model exists |
|---|---:|---:|---:|---|
| chck_20M | 20000000 | 446 | 4.504033 | True |
| chck_50M | 50000000 | 1115 | 3.589342 | True |
| chck_70M | 70000000 | 1561 | 3.395927 | True |
| chck_82M | 82026100 | 1829 | 3.303995 | True |
| chck_90M | 90000000 | 2007 | 3.299829 | True |
| chck_100M | 100000000 | 2230 | 3.28013 | True |

## compact - repeat accounting deltas

- final_loss: 0.012320999999999582
- raw_tokens_per_epoch_with_eos: 32144
- active_tokens_per_epoch: 32256
- active_token_relative_to_repeat: 0.002216046994266418
- dropped_tail_tokens_per_epoch: -112
- charged_words_per_active_token: -0.0015190990000000237

Selected checkpoint deltas (loss remains non-decisive):

| endpoint | Δwords | Δstep | Δloss |
|---|---:|---:|---:|
| chck_20M | 0 | 0 | 0.0035290000000003374 |
| chck_50M | 0 | 0 | 0.01842100000000002 |
| chck_70M | 0 | 0 | 0.014728000000000296 |
| chck_82M | -4480 | 0 | 0.008126999999999995 |
| chck_90M | 0 | 0 | 0.011802000000000312 |
| chck_100M | 0 | 0 | 0.012320999999999582 |

Official-compatible selected cheap7 results are required for the architecture-transfer conclusion.

JSON: `experiments/archive/frontier_consolidation/data/causal_training_accounting_audit/causal_training_accounting_audit.json`
