# legal40k accum training completion legal-40k two-seed comparison

Visible leader baseline: 41.800

## Overall
- legal40k seed43022: 41.140578 (margin -0.659422)
- legal40k seed43122: 40.420132 (margin -1.379868)
- legal40k mean: 40.780355
- legal16k mean: 40.863975
- mean delta legal40k - legal16k: -0.083620

## Restore/preserve pattern vs legal16k
- Supplement: +2.612487
- EWoK: +1.106247
- GlobalPIQA: -3.941748
- Entity: -1.414207
- COMPS: +0.127639

## Scientific reading
- Neither legal-40k seed clears the visible leader; do not spend another full pair on intermediate vocabulary sizes. Move to a different curriculum or architecture route.
- The load-bearing pattern is whether Supplement/EWoK recover from legal-16k while GlobalPIQA/Entity/COMPS do not surrender the compact-view gains.

JSON: `experiments/archive/representation_and_objectives/data/legal40k_two_seed_comparison/legal40k_two_seed_comparison.json`
