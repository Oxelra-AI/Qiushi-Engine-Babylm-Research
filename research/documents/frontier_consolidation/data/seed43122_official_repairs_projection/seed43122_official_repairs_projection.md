# earlier analysis — seed43122 projection after official EWoK/AoA repairs

This does not treat the full seed43122 official vector as delivered. It updates only the pieces that are now official-coordinate: EWoK and AoA.

## Delivered official-coordinate repairs
- EWoK: 51.89 over 7618 items
- AoA: 0.00, row_count_values=[8005]

## Absolute 41.8 projection with fast non-EWoK columns held fixed
- Seven-column sum (BLiMP/Supp/EWoK/Entity/COMPS/GPIQA/Reading): 302.670
- SuperGLUE required for Overall 41.8 if AoA=0: 73.530
  - clean_seed43122_superglue: SG=68.745, Overall=41.2684, margin=-0.5316, TE_vs_clean43122=+0.6183
  - clean_seed43022_superglue: SG=70.309, Overall=41.4421, margin=-0.3579, TE_vs_clean43122=+0.7920
  - reinvest_seed43022_superglue: SG=71.036, Overall=41.5229, margin=-0.2771, TE_vs_clean43122=+0.8728
  - compact_repeat_core_superglue: SG=70.624, Overall=41.4772, margin=-0.3228, TE_vs_clean43122=+0.8271
  - compact_view_core_superglue: SG=68.901, Overall=41.2857, margin=-0.5143, TE_vs_clean43122=+0.6356
  - visible_leader_superglue: SG=69.790, Overall=41.3844, margin=-0.4156, TE_vs_clean43122=+0.7344
  - required_for_41p8: SG=73.530, Overall=41.8000, margin=+0.0000, TE_vs_clean43122=+1.1499

## EWoK treatment-effect update
- TE43022 EWoK full official: +3.347
- TE43122 EWoK with official repair: +1.460
- DiD EWoK with official repair: -1.887
- Old fast-based EWoK DiD: -3.970

The EWoK official repair narrows the apparent seed43122 treatment-specific EWoK loss, but it remains the main non-replicating column in the preliminary treatment-effect picture.

Machine-readable output: `experiments/archive/frontier_consolidation/data/seed43122_official_repairs_projection/seed43122_official_repairs_projection.json`
