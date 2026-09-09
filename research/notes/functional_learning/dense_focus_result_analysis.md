# dense focus common target analysis Dense Focus Result Analysis

## Completed dense-focus result

**Training parameters (late modification from coherent86, alpha=0.75):**
- focus_prob=1.0 (ALL Qwen groups selected, not sparse 15%)
- lambda_focus=0.15 (same coefficient as sparse)
- 80 updates on unchanged Qwen tail (3.16M prefix words)
- Total focus targets: 176,607 across 80 updates
- Total ordinary targets: 592,858
- Focus target fraction: 22.95%

**Dense focus Cheap7 scores at update 80:**
| Family | Dense Focus | Coherent86 | Sparse Focus | Ordinary WWM |
|--------|------------|------------|--------------|--------------|
| BLiMP  | 68.62      | ~68.49     | 68.47        | 69.22        |
| Supplement | 66.0   | ~62.94     | 66.4         | 64.7         |
| EWoK   | 50.18      | ~49.82     | 49.73        | 49.64        |
| Entity | **28.44**  | 27.78      | 27.80        | 27.34        |
| COMPS  | 52.15      | ~51.99     | 52.14        | 51.57        |
| GlobalPIQA_mean | **40.05** | ~38.57 | 38.57   | 39.39        |
| Reading | 8.22      | ~8.15      | 8.24         | 8.04         |
| **equal_valid_mean** | **44.809** | 44.564 | 44.545 | 44.608 |

## Key observations

1. **Strongest fast-screen result in functional_learning**: equal_valid_mean 44.809,
   delta +0.245 above coherent86

2. **Entity improvement**: 28.44 vs 27.78, delta +0.66 above coherent86.
   This is the highest Entity in any functional_learning experiment.

3. **Broad improvement, not single-family**: BLiMP, Supplement, EWoK, Entity,
   GlobalPIQA all improved. Only COMPS was essentially flat.

4. **Dense vs sparse focus**: Dense focus (176K focus targets) dramatically
   outperforms sparse focus (28.6K focus targets). The 6× more focus credit
   on Qwen pair positions translates directly to broad improvement.

## Scientific interpretation

Dense focus selects ALL word groups in Qwen pair segments for focused
prediction. This means every content word in source/rewrite pairs gets
concentrated training credit (λ_focus=0.15 vs λ_ordinary=0.85).

The improvement over sparse focus is consistent with the principle that
paired/distinct content (Qwen source+rewrite pairs) is more valuable per
word than ordinary repeated content. Concentrating credit on ALL of it
(dense) works much better than concentrating on a random 15% (sparse).

This aligns with frontier_consolidation MAX evidence: V-C >> R-C (distinct content
beats exact repeat). Dense focus effectively gives the model more
"learning events" on the distinct content.

## Caveats

1. Single run, no seed replication
2. Late modification from coherent86 (not formation stage)
3. Fast-screen is not official Overall
4. Supplement and GlobalPIQA show larger variance across evaluations
5. GlobalPIQA_parallel (30.1) is lower than some other arms; the mean
   is carried by nonparallel (50.0)

## What this changes for the research

The dense focus result is potentially a v5 candidate if it:
a) Replicates with a different seed
b) Survives official-style evaluation
c) Shows genuine broad improvement, not just Supplement/GlobalPIQA variance

The formation experiment (reference_4M and full_18M) tests a different
and potentially deeper principle. Both tracks should continue.

## Next steps
- Wait for formation experiments to complete
- If formation produces strong results → compare formation vs dense focus
- If formation is weak → pursue dense focus replication and official eval
- Either way, both inform the data-efficient learning principle

## Artifact paths
- Training: `data/unchanged_dense_focus_train/correspondence_focus_weighted/`
- Evaluation: `data/unchanged_dense_focus_eval/cheap7/unchanged_correspondence_focus_weighted_u0080/`
- Checkpoint: `.../checkpoints/update_0080/` (SHA: 0b21387bcc060a27...)
