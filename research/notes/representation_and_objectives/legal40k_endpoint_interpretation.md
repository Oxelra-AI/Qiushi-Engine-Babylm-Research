# legal40k endpoint interpretation legal-40k endpoint interpretation

Status: `LEGAL40K_ENDPOINT_INTERPRETATION_READY`

Visible leader Overall: `41.8`

## Overall
- seed 43022: legal40k `41.140578`; legal16k `40.703956`; delta `+0.436621`; margin vs 41.8 `-0.659422`
- seed 43122: legal40k `40.420132`; legal16k `41.023994`; delta `-0.603862`; margin vs 41.8 `-1.379868`
- legal40k mean `40.780355`; delta vs legal16k mean `-0.083620`

## Restore/preserve pattern vs legal16k
- Supplement: `+2.612487`
- EWoK: `+1.106247`
- GlobalPIQA: `-3.941748`
- Entity: `-1.414207`
- COMPS: `+0.127639`

## Largest mean movements vs legal16k
- GlobalPIQA: `-3.941748`
- Supplement: `+2.612487`
- Entity: `-1.414207`
- BLiMP: `+1.387726`
- EWoK: `+1.106247`
- SuperGLUE: `-0.481321`
- Reading: `-0.149407`
- COMPS: `+0.127639`
- Overall: `-0.083620`
- AoA: `+0.000000`

## Integrity
- Hardened collation integrity checks passed for both seeds.

## Scientific reading
- Neither legal-40k seed clears the visible leader; vocabulary-breadth/embedding-capacity alone did not solve the compliant SOTA problem.
- Mean Supplement and EWoK both recovered relative to legal16k; interpret 40k as partially repairing the legal representation coordinate even if Overall remains below frontier.
- At least one compact-view-preserve column fell materially relative to legal16k; inspect full vector before choosing relation masking versus depth.

## Files
- json: `experiments/archive/representation_and_objectives/data/legal40k_endpoint_interpretation/legal40k_endpoint_interpretation.json`
- column_csv: `experiments/archive/representation_and_objectives/data/legal40k_endpoint_interpretation/legal40k_column_comparison.csv`
- detail_csv: `experiments/archive/representation_and_objectives/data/legal40k_endpoint_interpretation/legal40k_detail_comparison.csv`
- note: `research/notes/representation_and_objectives/legal40k_endpoint_interpretation.md`
- top_level_comparison_script: `experiments/archive/representation_and_objectives/scripts/compare_legal40k_two_seed_results.py`
