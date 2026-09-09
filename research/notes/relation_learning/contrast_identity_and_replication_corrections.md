# Contrast Identities, Audit Counts, and Replication Limits

Scientific status: corrections to the interpretation of existing measurements. No additional training, evaluation, resampling or model inference was performed for this record.

## Natural Transfer: Preserve the Comparison Arm

The Wikipedia-to-Simple-English probe contains 1,200 pairs. Its source-recurring target class is called `overlap`: target content recurs in its conditioning source. This is not a label for overlap between training and evaluation datasets. The source-absent substitution class is `nonoverlap`.

For true-source advantage `A_T = NLL(N) - NLL(T)`, the existing pair-averaged, target-class-specific table gives the following across three training seeds. Errors are seed standard deviations, not confidence intervals.

| Target Class | Contrast | Mean Delta A_T | Seed SD |
|---|---|---:|---:|
| Source-recurring | VIEW - CLEAN | +0.2376 | 0.0706 |
| Source-recurring | REPEAT - CLEAN | +0.1090 | 0.1216 |
| Source-recurring | VIEW - REPEAT | +0.1286 | 0.0533 |
| Source-absent | VIEW - CLEAN | +0.0316 | 0.0398 |
| Source-absent | REPEAT - CLEAN | -0.2171 | 0.0285 |
| Source-absent | VIEW - REPEAT | +0.2487 | 0.0646 |

Thus the historical rounded value **+0.24 +/- 0.07 on source-recurring targets belongs to VIEW minus CLEAN, not VIEW minus REPEAT**. The latter is +0.1286 +/- 0.0533. The distinction changes the stated effect magnitude without changing any measurement. It also prevents conflating source-recurring and source-absent contrasts or substituting a class-balanced aggregate for a target-weighted aggregate. The [existing numerical table](../../../experiments/archive/relation_learning/data/numerical_repair/wikipedia_transfer_summary_recomputed_classbalanced_and_tokenweighted.csv) retains the full precision and aggregation labels.

## Exposure Audits: Hits Are Not Unique Training Rows

Two historical screens examined different selected pair populations against the same 225,109 diagnostic reference texts. Their counts are not alternative versions of one result.

| Screened Population | Selected Pairs | Source/Rewrite Texts | Blocking Hits | Affected Pair IDs |
|---|---:|---:|---:|---:|
| Initial additional-dose selection | 17,804 | 35,608 | 35,341 | 2,330 |
| Inherited aligned-restatement block | 37,594 | 75,188 | 33,431 | 3,395 |

The initial additional-dose selection comprised 9,385 dose21 pairs and 8,419 top-up pairs. Its affected IDs split into 1,209 dose21 and 1,121 top-up pairs. Its 35,341 hits comprised 25,744 matches to the 6,992-row reference set and 9,597 to the 2,647-row set. These are the initial selection counts, not the later screened and reconstructed dose-stream counts.

The inherited block instead produced 24,071, 9,359 and one blocking hit against the two row sets and the Wikipedia probe, respectively. Multiple reference matches can correspond to one source text or pair. Consequently, neither 35,341 nor 33,431 is a count of unique leaked training rows, and the former must not be attributed to the inherited block. These screens address their stated diagnostic reference collections; they do not establish that every official benchmark is contaminated or clean. See the [additional-dose audit](../../../experiments/archive/relation_learning/data/dose_leakage_audit/audit_summary.json) and [inherited-block audit](../../../experiments/archive/relation_learning/data/inherited_aln_leakage_audit/audit_summary.json).

## Replication: A Realized Difference Is Not a Noise Threshold

The matched ordinary-continuation endpoints have Entity scores 28.16 and 28.09 and SuperGLUE scores 68.98845 and 69.27328. Their absolute differences, approximately 0.07 and 0.285 component points, describe the two observed runs. They do not establish a universal +/-0.07 noise band, a 0.3-point significance cutoff, or a sampling distribution for other policy contrasts. In particular, the ordinary-continuation Entity changes from the parent, -0.16 and -0.23, are not numerically inside a +/-0.07 band.

The retained evidence is narrower: both continuation seeds started from the same frozen parent; each observed Overall rung was positive; and the component changes and complete-policy gains are available in the [existing endpoint table](../../../results/training_strategy_comparison.csv). They are not independent from-scratch replications. Component-score differences and Overall differences also have different units: Overall averages nine components.

The separate, already-completed paired-item analysis reported clean64 minus exact `(M,S)` as +0.043874 Overall, with a 2.5--97.5% resampling interval of [-0.034541, +0.129073] and a positive fraction of 0.8650. This is conditional sensitivity to resampling the fixed validation items, not training-seed uncertainty or a probability of general superiority. The [paired-item analysis](../../documents/functional_learning/data/direct_increment_paired_uncertainty_reworded/direct_increment_paired_uncertainty.md) states those limits. Its full-precision increment rounds to +0.044 at three decimals; subtracting the displayed endpoints 42.246 and 42.203 instead yields +0.043, a display-rounding difference rather than another experimental result.
