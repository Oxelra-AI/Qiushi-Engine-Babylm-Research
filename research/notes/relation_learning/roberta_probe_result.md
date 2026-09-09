# roberta probe result RoBERTa relation-structure probe result

Source probe outputs: `experiments/archive/relation_learning/data/roberta_probe_dynamic`. Summary JSON: `experiments/archive/relation_learning/data/roberta_probe_summary/roberta_probe_summary.json`.

## Scientific reading

The same checkpoint probes used for DeBERTa seeds43022/43122 were run on the earlier analysis RoBERTa seed43022 VIEW/REPEAT/CLEAN arms at 60M, 70M, 80M, 90M, and 100M. This is an architecture/learner-coordinate test, not a new leaderboard run.

RoBERTa does **not** reproduce the DeBERTa held-out natural copy ordering. Across 60M-100M, copy-gain contrasts on 2,000 held-out natural source-repeat pairs are RminusC -0.0487, RminusV -0.0569, and VminusC +0.0082. Thus the RoBERTa learner has no visible exact-recurrence advantage on this probe; if anything, REPEAT is slightly below the other arms. This means the exact-copy tendency found in DeBERTa is not architecture-independent at this checkpoint/loss regime.

RoBERTa **does** reproduce the content-conditioning ordering on compact held-out rewrites. On all rewrite content tokens, VminusC is +0.1058 and VminusR is +0.4272; on tokenizer-nonoverlap tokens, VminusC is +0.0666 and VminusR is +0.4649. CminusR is also positive, especially nonoverlap +0.3983. For nonoverlap tokens, REPEAT has worse true-source NLL than CLEAN by +0.4924 nats/token and worse unrelated-source NLL by +0.0942; the lower gain is not created by an unusually easy unrelated-source denominator. VIEW has better true-source NLL than CLEAN by -0.5116 while also having better unrelated-source NLL by -0.4450, so the small V-C gain is the residual benefit after a broad rewrite-register fit advantage.

The Entity stale/update cue-ablation signals are small in RoBERTa and should not be read as the same behavioral mechanism as DeBERTa. On the stale-foil update subset, full gold-over-stale margin VminusR is -0.1183 while VminusC is +0.0478. The no-last-update effect does not reproduce the DeBERTa sign: VminusR is +0.0335 rather than negative. This weakens any claim that RoBERTa already expresses the Entity state-update behavior, even though the content-conditioning probe separates the training arms.

## Consequence for the data-efficient learning principle

The result improves the principle by forcing the learner coordinate back into the account. Fixed-budget relation structure is not converted into identical computations by every architecture/training phase. Nonidentical restatement produces a content-conditioning tendency in both DeBERTa and RoBERTa, but exact recurrence becoming a transferable natural-copy advantage appears stronger in DeBERTa than in this RoBERTa run. The principle should therefore be stated as an interaction between experience relation and learner coordinate/training phase, not as an architecture-free law. The next behavioral evidence must decide whether the DeBERTa seed43222 Entity crossover survives and whether a natural re-mention probe can show the copy/content split outside compact rewrites and boxes.
