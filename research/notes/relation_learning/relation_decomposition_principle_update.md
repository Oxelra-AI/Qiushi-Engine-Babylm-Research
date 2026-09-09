# pair level relation robustness relation-probe term decomposition

This CPU-only analysis reuses the already scored probe results interpretation DeBERTa and roberta probe result RoBERTa probe rows. It separates each source-conditioning gain into the true-source NLL term and the unrelated-source control term, and it rescales natural-copy gain by each arm's own unrepeated-control NLL level.

## Main non-overlap rewrite result

For rewrite tokens whose tokenizer IDs do not occur in the source, gain = NLL(unrelated source) - NLL(true source). In the table below, true/unrelated deltas are A minus B; positive true-source delta means arm A is worse with the actual source present. `excess_true_cost` = true-source delta - unrelated-source delta; positive values mean worse true-source use beyond the broad control level, and it equals the negative of the gain delta.

| arch | seed | ckpts | contrast | gain delta | true-source delta | unrelated delta | excess true-source cost | sd(gain) over ckpts |
|---|---:|---|---|---:|---:|---:|---:|---:|
| D | 43022 | D_late80_100 | RminusC | -0.7517 | +0.4480 | -0.3037 | +0.7517 | +0.1479 |
| D | 43022 | D_late80_100 | VminusC | +0.6837 | -1.1788 | -0.4951 | -0.6837 | +0.0503 |
| D | 43022 | D_late80_100 | VminusR | +1.4354 | -1.6268 | -0.1914 | -1.4354 | +0.0976 |
| D | 43122 | D_late80_100 | RminusC | -1.0433 | +0.6931 | -0.3502 | +1.0433 | +0.0507 |
| D | 43122 | D_late80_100 | VminusC | +0.8938 | -1.3544 | -0.4606 | -0.8938 | +0.0229 |
| D | 43122 | D_late80_100 | VminusR | +1.9371 | -2.0475 | -0.1104 | -1.9371 | +0.0733 |
| RBT | 43022 | RBT_all60_100 | RminusC | -0.3983 | +0.4924 | +0.0942 | +0.3983 | +0.1358 |
| RBT | 43022 | RBT_all60_100 | VminusC | +0.0666 | -0.5116 | -0.4450 | -0.0666 | +0.1265 |
| RBT | 43022 | RBT_all60_100 | VminusR | +0.4649 | -1.0040 | -0.5391 | -0.4649 | +0.2503 |

Scientific reading: the cross-architecture invariant is the REPEAT cost. In late DeBERTa and in the RoBERTa 60M-100M average, REPEAT is worse than CLEAN on non-overlap true-source NLL, and the source-conditioning loss remains after subtracting the unrelated-source control. The RoBERTa 60M true-source term alone is an early exception, but the conditioning-gain cost is present at every measured RoBERTa checkpoint. Thus exact in-window natural recurrence does not merely fail to buy nonidentical content use; it installs a competing tendency that makes the learner worse than the no-companion baseline on held-out source-conditioned nonidentical tokens.

The VIEW-side benefit differs by architecture. In DeBERTa, V-C non-overlap gain is large in both seeds and its true-source advantage exceeds the unrelated-source advantage, so DeBERTa provides direct evidence for content-conditioned source use beyond compact-register fit. In RoBERTa, V-C true-source and unrelated-source improvements are both large and close, leaving only a small residual conditioning gain. Therefore the positive VIEW half is strongly supported for DeBERTa but only weakly separated from broad compact-register fit in this RoBERTa run.

## Natural-copy gain after control normalization

Copy gain = NLL(unrepeated control) - NLL(repeated source). `norm gain` divides each arm's mean gain by that same arm's mean unrepeated-control NLL within the checkpoint and group, so arm contrasts are not dominated by different loss levels.

| arch | seed | ckpts | contrast | raw gain delta | normalized gain delta | control NLL delta | repeated NLL delta | sd(raw) over ckpts | raw checkpoint values | norm checkpoint values |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| D | 43022 | D_late80_100 | RminusC | +0.4996 | +0.0497 | +0.3115 | -0.1881 | +0.0364 | chck_100M:0.467219;chck_80M:0.539061;chck_90M:0.492457 | chck_100M:0.0473442;chck_80M:0.0526207;chck_90M:0.04905 |
| D | 43022 | D_late80_100 | VminusC | +0.3271 | +0.0091 | +0.3178 | -0.0093 | +0.0332 | chck_100M:0.319913;chck_80M:0.36332;chck_90M:0.298087 | chck_100M:0.00780991;chck_80M:0.0123156;chck_90M:0.00705786 |
| D | 43022 | D_late80_100 | RminusV | +0.1725 | +0.0406 | -0.0064 | -0.1788 | +0.0237 | chck_100M:0.147306;chck_80M:0.175741;chck_90M:0.19437 | chck_100M:0.0395343;chck_80M:0.0403051;chck_90M:0.0419921 |
| D | 43122 | D_late80_100 | RminusC | +0.6678 | +0.0531 | +0.4780 | -0.1898 | +0.0292 | chck_100M:0.655754;chck_80M:0.701177;chck_90M:0.646567 | chck_100M:0.0521379;chck_80M:0.0541631;chck_90M:0.0530136 |
| D | 43122 | D_late80_100 | VminusC | +0.3103 | +0.0119 | +0.2892 | -0.0211 | +0.0233 | chck_100M:0.292563;chck_80M:0.336767;chck_90M:0.301686 | chck_100M:0.0116945;chck_80M:0.0109361;chck_90M:0.0131609 |
| D | 43122 | D_late80_100 | RminusV | +0.3575 | +0.0412 | +0.1888 | -0.1687 | +0.0109 | chck_100M:0.363191;chck_80M:0.36441;chck_90M:0.34488 | chck_100M:0.0404434;chck_80M:0.043227;chck_90M:0.0398528 |
| RBT | 43022 | RBT_all60_100 | RminusC | -0.0487 | -0.0272 | +1.1754 | +1.2241 | +0.1682 | chck_100M:-0.197993;chck_60M:0.173657;chck_70M:0.0835122;chck_80M:-0.113276;chck_90M:-0.189454 | chck_100M:-0.0611678;chck_60M:0.0249479;chck_70M:0.00196133;chck_80M:-0.0422803;chck_90M:-0.0592577 |
| RBT | 43022 | RBT_all60_100 | VminusC | +0.0082 | -0.0036 | +0.2970 | +0.2888 | +0.0502 | chck_100M:-0.011807;chck_60M:0.0961227;chck_70M:-0.000133392;chck_80M:-0.028494;chck_90M:-0.0146558 | chck_100M:-0.00613133;chck_60M:0.0126004;chck_70M:-0.00673856;chck_80M:-0.0106756;chck_90M:-0.00695427 |
| RBT | 43022 | RBT_all60_100 | RminusV | -0.0569 | -0.0236 | +0.8784 | +0.9353 | +0.1315 | chck_100M:-0.186186;chck_60M:0.0775343;chck_70M:0.0836456;chck_80M:-0.084782;chck_90M:-0.174798 | chck_100M:-0.0550364;chck_60M:0.0123475;chck_70M:0.00869989;chck_80M:-0.0316047;chck_90M:-0.0523034 |

Scientific reading: DeBERTa's natural-copy signal remains large after normalization. For REPEAT versus CLEAN, the raw gain advantage is about +0.50 and +0.67 nats across the two seeds, and normalized gain remains higher by about +0.050 and +0.053 of the arm control NLL. RoBERTa's raw R-C mean is close to zero and changes sign across checkpoints; after normalization it is still near zero. This is not evidence of an opposite RoBERTa copy tendency. It says the probe barely resolves a copy-practice difference in RoBERTa at this loss level, while it resolves one clearly in DeBERTa.

## Principle update

The most stable statement is now negative-and-constructive: under a fixed finite budget, exact in-window recurrence can train a cross-span identity tendency that competes with nonidentical source-conditioned content use; this active cost appears in both DeBERTa and RoBERTa on held-out compact non-overlap rewrite tokens. The complementary positive result is learner-dependent: DeBERTa turns nonidentical restatement into strong content-conditioned source use and a multi-update Entity advantage, whereas RoBERTa shows the repetition cost and only a small V-C residual after compact-register fit is subtracted.

This changes the center of the research route. The general data-efficient learning principle should not rest only on VIEW helping Entity. It should rest on the structure of finite experience installing particular cross-span relations: repeated identity can be useful for unchanged-state retrieval and local copying, but the same practiced identity relation can actively degrade nonidentical content use; varied restatement supplies evidence for a different relation, which in DeBERTa converts into state-update discrimination. The pending third DeBERTa seed and the natural re-mention instrument now test whether the positive conversion and behavioral crossover survive outside the two-seed compact-rewrite setting.

## Files

- rewrite_arm_terms: `experiments/archive/relation_learning/data/relation_decomposition/rewrite_arm_terms.csv`
- rewrite_contrast_terms_by_checkpoint: `experiments/archive/relation_learning/data/relation_decomposition/rewrite_contrast_terms_by_checkpoint.csv`
- rewrite_contrast_late_summary: `experiments/archive/relation_learning/data/relation_decomposition/rewrite_contrast_late_summary.csv`
- copy_arm_terms: `experiments/archive/relation_learning/data/relation_decomposition/copy_arm_terms.csv`
- copy_contrast_terms_by_checkpoint: `experiments/archive/relation_learning/data/relation_decomposition/copy_contrast_terms_by_checkpoint.csv`
- copy_contrast_late_summary: `experiments/archive/relation_learning/data/relation_decomposition/copy_contrast_late_summary.csv`
- summary_json: `experiments/archive/relation_learning/data/relation_decomposition/relation_decomposition_summary.json`
- note: `research/notes/relation_learning/relation_decomposition_principle_update.md`
