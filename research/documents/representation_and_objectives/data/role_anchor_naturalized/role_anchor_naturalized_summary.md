# role coordinate anchor and state probe naturalized role-anchor transfer test

Seen predicates: ['dax', 'feg', 'mip', 'lor']
Held predicates: ['zup', 'niv', 'kem', 'rox']
No world/domain/case markers; identical relation syntax; pair combinations held out but names reused.

| arm | runs | fit runs | train | seen comp | held hyp | held ctx | held both |
|---|---:|---:|---:|---:|---:|---:|---:|
| noheld_filler | 3 | 0 | 0.584 | nan | nan | nan | nan |
| exposure_only | 3 | 0 | 0.507 | nan | nan | nan | nan |
| true_anchor | 3 | 0 | 0.501 | nan | nan | nan | nan |
| shuffled_anchor | 3 | 0 | 0.500 | nan | nan | nan | nan |
| coverage_only | 3 | 0 | 0.557 | nan | nan | nan | nan |

Numbers in the four right columns are train-fit-only selected and fit-run means. Raw seed values and train-kind accuracies are in the JSON.

Summary JSON: `experiments/archive/representation_and_objectives/data/role_anchor_naturalized/role_anchor_naturalized_summary.json`
