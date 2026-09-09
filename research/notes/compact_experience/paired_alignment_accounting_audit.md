# paired alignment accounting audit — Paired-alignment accounting audit

JSON: `experiments/archive/compact_experience/data/paired_alignment_accounting_audit.json`

The accounting audit identifies why the existing script does not support a valid Wave 2 comparison.

## Arm accounting

| arm | base words | missing to 10M | exposure words | effective passes | extra beyond 10 passes | status |
|---|---:|---:|---:|---:|---:|---|
| aligned | 9999840 | 160 | 100000000 | 10.000160 | 1600 | exceeds_10_passes |
| mismatched | 9999840 | 160 | 100000000 | 10.000160 | 1600 | exceeds_10_passes |
| single_repeat | 9999840 | 160 | 100000000 | 10.000160 | 1600 | exceeds_10_passes |
| single_orig | 9551200 | 448800 | 100000000 | 10.469889 | 4488000 | exceeds_10_passes |

## Consequences

- The Wave-1 ALIGNED/MISMATCHED comparison is still a matched mechanism screen: both use 9,999,840-word pools and 100,000,000-word exposure, so both are 10 full passes plus 1,600 words.
- The same runs should not be treated as official-candidate compliant under a literal ≤10 epoch interpretation until rebuilt with exactly 10,000,000-word base pools or rerun at exactly 99,998,400 exposure with explicit final-checkpoint handling.
- Existing Wave 2 must not be launched: SINGLE_ORIG has only 9,551,200 base words, so 100,000,000 exposure would be about 10.47 passes and would confound the control as well as violate the literal epoch limit.
- The ALIGNED–MISMATCHED contrast tests meaning-related/coherent adjacency versus scrambled local adjacency. It does not by itself separate paraphrase/rewrite correspondence from topical coherence; a same-topic non-synonymous adjacency control is needed next.
