# clean natural nearduplicate adjacency RoBERTa REPEAT_SPLIT integration

This note integrates the delivered RoBERTa `REPEAT_SPLIT` seed43022 run with the prior RoBERTa CLEAN/REPEAT/VIEW roberta probe result probe rows. It is an architecture/learner-coordinate extent test, not a new BabyLM leaderboard run.

## Main numbers

Late means average checkpoints 60M,70M,80M,90M,100M to match the prior RoBERTa readout. Rewrite gain is `unrelated-source NLL - true-source NLL`, so a positive arm contrast means stronger true-source use relative to its unrelated-source denominator.

| Readout | R−C | RS−C | R−RS | Interpretation |
|---|---:|---:|---:|---|
| Natural-copy gain | -0.0487 | -0.1261 | +0.0774 | RoBERTa had no positive original exact-copy advantage, and splitting does not reveal a DeBERTa-like copy routine. |
| Compact rewrite nonoverlap gain | -0.3983 | -0.1014 | -0.2968 | REPEAT remains below CLEAN, but RS is also below CLEAN; unlike DeBERTa, row splitting does not collapse the gain all the way to CLEAN. |
| Nonoverlap true-source NLL (lower better; arm contrast A−B) | +0.4924 | +0.0603 | +0.4321 | The RoBERTa true-source deficit is not uniquely local-repeat-specific at this checkpoint; split exposure still carries a deficit. |
| Nonoverlap unrelated-source NLL (lower better; arm contrast A−B) | +0.0942 | -0.0411 | +0.1353 | Decomposition does not reproduce DeBERTa's clean true-worse/unrelated-better locality collapse. |
| Entity rel≥3 full margin | +0.2138 | +0.2319 | -0.0181 | RoBERTa Entity remains weak and should not be used as the behavioral face of the DeBERTa mechanism. |

## Scientific reading

The delivered RoBERTa split arm constrains architecture generality rather than strengthening it. In DeBERTa, exact local recurrence produced a large compact nonoverlap cost relative to CLEAN and the matched REPEAT_SPLIT arm strongly attenuated that source-specific residual. In this single RoBERTa seed, the original R−C nonoverlap gain was already weaker, and RS−C remains negative rather than landing near CLEAN. RoBERTa also lacks the DeBERTa natural-copy advantage. Therefore the safest conclusion is that relation type is a training variable whose conversion depends on learner coordinate/objective phase; the DeBERTa MLM locality result should not be stated as an architecture-free law.

The useful positive evidence from RoBERTa remains narrower: RoBERTa separates the VIEW/restatement arm from REPEAT on compact rewrite source use, but the exact-recurrence locality mechanism is not reproduced in the same form. This is not a failure of the DeBERTa result; it is a boundary measurement that protects the general principle from overextension.

## Files

- Raw RS rows: `experiments/archive/relation_learning/data/roberta_repeat_split_probe_dynamic`
- Integrated late contrasts: `experiments/archive/relation_learning/data/roberta_repeat_split_integration/roberta_repeat_split_late_contrasts.csv`
- Role summaries: `experiments/archive/relation_learning/data/roberta_repeat_split_integration/roberta_roles_summary.csv`
- Result JSON: `experiments/archive/relation_learning/data/roberta_repeat_split_integration/roberta_repeat_split_integration_result.json`
