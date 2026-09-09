# ordinary heldout price probe two-seed split-control replication

This integration compares the original local arms with split-row controls at seeds 43022 and 43122 using the same held-out copy and compact-rewrite probes. The split rows retain the same selected source and companion texts and the same 100M-word budget, but remove source/companion co-occurrence within the training window.

## Compact rewrite token-nonoverlap summary

| seed | R−C gain | RS−C gain | RS−R gain | V−C gain | VS−C gain | VS−V gain | reading |
|---:|---:|---:|---:|---:|---:|---:|---|
| 43022 | -0.7517 | -0.0313 | +0.7204 | +0.6837 | +0.0298 | -0.6539 | RS near_clean; VS near_clean |
| 43122 | -1.0433 | -0.1898 | +0.8535 | +0.8938 | -0.0355 | -0.9293 | RS near_clean; VS near_clean |

Across the two split seeds, RS−C token-nonoverlap gain averages -0.1105. Seed43022 is essentially CLEAN-like (-0.0313); seed43122 is still far attenuated from original R−C -1.0433 but lands at -0.1898, just inside the ±0.25 near-CLEAN band rather than exactly at zero. The same-window exact-recurrence cost is therefore strongly reduced in both seeds, with a small remaining negative residual in seed43122.
Across the two split seeds, VS−C token-nonoverlap gain averages -0.0028. Seed43022 is near zero (+0.0298); seed43122 is also near zero but slightly negative (-0.0355) while original V−C at the same seed is +0.8938. This repeats the collapse of VIEW's source-specific compact-rewrite benefit under row splitting.

## T/U term reading

| seed | R−C ΔT | RS−C ΔT | R−C ΔU | RS−C ΔU | V−C ΔT | VS−C ΔT | V−C ΔU | VS−C ΔU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 43022 | +0.4480 | -0.5469 | -0.3037 | -0.5782 | -1.1788 | -0.8518 | -0.4951 | -0.8220 |
| 43122 | +0.6931 | -0.3806 | -0.3502 | -0.5704 | -1.3544 | -0.8164 | -0.4606 | -0.8519 |

For seed43122, RS−C no longer has the original true-source-worse / unrelated-source-better sign reversal: both T and U improve relative to CLEAN (ΔT −0.3804, ΔU −0.5702). The remaining negative RS−C gain comes from the true source helping less than the unrelated source after both broad terms improve. VS−C similarly improves both T and U, with U slightly more improved, making the gain slightly negative despite the local VIEW arm being strongly positive.

## Natural-copy summary

| seed | R−C copy | RS−C copy | RS−R copy | V−C copy | VS−C copy | VS−V copy |
|---:|---:|---:|---:|---:|---:|---:|
| 43022 | +0.4996 | -0.2005 | -0.7001 | +0.3271 | +0.0997 | -0.2274 |
| 43122 | +0.6678 | +0.2693 | -0.3986 | +0.3103 | +0.2038 | -0.1066 |

The natural-copy side is not a symmetric collapse. RS−C copy gain averages +0.0344: seed43022 was below CLEAN in raw gain after normalization complications, but seed43122 keeps a positive copy gain of +0.2693, smaller than original R−C +0.6678. VS−C copy gain remains positive in both seeds and averages +0.1517. Thus row splitting removes the source-specific compact-rewrite residuals more cleanly than it removes all source-present exact-copy behavior.

## Pair bootstrap on nonoverlap gain

| seed | contrast | mean | 95% pair interval | n pairs |
|---:|---|---:|---:|---:|
| 43022 | RminusC | -0.7852 | [-0.8733, -0.6913] | 1465 |
| 43022 | RSminusC | -0.0470 | [-0.1237, +0.0304] | 1465 |
| 43022 | VminusC | +0.6711 | [+0.5746, +0.7652] | 1465 |
| 43022 | VSminusC | +0.0040 | [-0.0725, +0.0859] | 1465 |
| 43122 | RminusC | -1.0550 | [-1.1574, -0.9500] | 1465 |
| 43122 | RSminusC | -0.1913 | [-0.2671, -0.1077] | 1465 |
| 43122 | VminusC | +0.8891 | [+0.8022, +0.9831] | 1465 |
| 43122 | VSminusC | -0.0531 | [-0.1325, +0.0284] | 1465 |

## Scientific update

The second split seed repeats the central locality result for compact nonidentical source use. Same-content row splitting collapses both sides toward CLEAN: exact recurrence no longer creates the large original source-conditioned cost, and restatement no longer creates the large original source-conditioned benefit. The updated result is slightly less symmetric than the seed43022 2×2: seed43122 REPEAT_SPLIT retains a small residual cost and a reduced but positive copy gain, while VIEW_SPLIT is slightly below CLEAN on compact nonoverlap gain. This strengthens the core statement that within-window relation practice is necessary for the large source-specific residuals, while warning that split exposure can still change ordinary fit and exact-copy behavior.

Data outputs: `experiments/archive/relation_learning/data/split_seed_replication`.
