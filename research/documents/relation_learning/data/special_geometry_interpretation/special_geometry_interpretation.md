# context geometry and v5 design special-token geometry interpretation

This summarizes KL mass from `special_geometry_kl_bins` on the same 32 matched examples used by format attribution controls.  Mass share is the fraction of total private/slow KL in that form, not a count of items.

| endpoint | form | KL mean | special mass | content dist=1 mass | content dist<=15 mass | far-content mass | short<=31 mass |
|---|---|---:|---:|---:|---:|---:|---:|
| coherent86_alpha075 | coherent_add_special | 0.00082465 | 6.9% | 2.4% | 13.6% | 79.6% | 0.0% |
| coherent86_alpha075 | short_add_special | 0.00183627 | 30.1% | 10.2% | 47.7% | 22.3% | 48.7% |
| coherent_special_98097_alpha075 | coherent_add_special | 0.00193546 | 11.4% | 38.0% | 45.8% | 42.8% | 0.0% |
| coherent_special_98097_alpha075 | short_add_special | 0.00974691 | 40.9% | 41.8% | 52.4% | 6.7% | 62.7% |
| coherent_special_98098_alpha075 | coherent_add_special | 0.00240540 | 12.1% | 43.5% | 50.0% | 37.9% | 0.0% |
| coherent_special_98098_alpha075 | short_add_special | 0.01199229 | 47.9% | 35.2% | 45.4% | 6.7% | 67.4% |

Main reading: compare coherent-special endpoints to coherent86.  If the extra KL mass concentrates at special tokens and adjacent content, then the branch is not merely off-leash globally; it is using a geometry channel created by adding special tokens to inputs whose relative-position geometry differs sharply between 256-token coherent rows and short evaluation rows.

CSV: `experiments/archive/relation_learning/data/special_geometry_interpretation/special_geometry_mass_shares.csv`
JSON: `experiments/archive/relation_learning/data/special_geometry_interpretation/special_geometry_interpretation.json`
