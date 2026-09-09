# earlier analysis dose overlap preregistration

Created: 2026-09-07T03:14:15Z

The repaired aligned-restatement dose should not be read as merely more of the inherited relation. The added pairs differ in source mix and in lexical overlap. This note is written before trusted dose-face scores are consumed.

## Same-definition overlap summaries

| pool | pairs | pair words | mean Jaccard | median Jaccard | p25-p75 Jaccard | mean min-overlap | mean LCS | frac LCS>=6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| inherited_compact_experience_aln_selected | 37594 | 1656800 | 0.4679 | 0.4545 | 0.2500-0.6667 | 0.6344 | 6.47 | 0.4337 |
| added_dose21_selected | 9299 | 443200 | 0.3009 | 0.2727 | 0.1818-0.4000 | 0.4812 | 3.47 | 0.0000 |
| added_dose25_extra | 8340 | 400000 | 0.2998 | 0.2727 | 0.1765-0.4000 | 0.4779 | 3.47 | 0.0000 |
| added_dose25_superset | 17639 | 843200 | 0.3004 | 0.2727 | 0.1786-0.4000 | 0.4797 | 3.47 | 0.0000 |

## Calibrated expectations for reading faces

COMPACT_EXPERIENCE ALN--OFF Wikipedia overlap reference uses seed mean 3.8986 from +3.5975/+4.1998 at 1.6568M inherited pair words per 10M. Total aligned-word scaling gives dose21 1.043 and dose25 1.984. SimpleWiki-word scaling gives dose21 0.551 and dose25 1.041.
Compact overlap uses the approximate ALN--OFF reference 1.670; total aligned-word scaling gives dose21 0.447 and dose25 0.850.

For source-absent/nonoverlap targets, preregister three branches: flat = asymmetric reach; degrading = restatement-side fixed-budget liability; rising = practiced substitution-like relation transfer because lower-overlap added pairs contain more rewrite tokens absent from the source. A nonoverlap rise is not generic broad generalization by itself.

Detailed metrics: `experiments/archive/relation_learning/data/dose_overlap_preregistration`.
