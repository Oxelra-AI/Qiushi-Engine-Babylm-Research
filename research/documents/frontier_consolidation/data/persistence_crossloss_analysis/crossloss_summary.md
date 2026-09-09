# earlier analysis persistence cross-loss analysis

CPU-only cross evaluation of admitted/displaced text sets under clean/view/repeat/breadth models.
Positive `clean_minus_owner` means the intervention model predicts its admitted text better than the clean model; near-zero means the clean model already had the text distribution covered.

## Checkpoint summary

### chck_40M
- view_changed: clean_loss=4.303809, owner_loss=3.393431, clean_minus_owner=0.910378
- repeat_changed: clean_loss=3.851267, owner_loss=2.490689, clean_minus_owner=1.360578
- breadth_changed: clean_loss=5.697204, owner_loss=5.298017, clean_minus_owner=0.399187
- clean_displaced: clean_loss=3.44143, owner_loss=3.44143, clean_minus_owner=0.0

### chck_80M
- view_changed: clean_loss=3.330201, owner_loss=2.678407, clean_minus_owner=0.651794
- repeat_changed: clean_loss=2.983797, owner_loss=2.069881, clean_minus_owner=0.913916
- breadth_changed: clean_loss=5.309294, owner_loss=4.821729, clean_minus_owner=0.487565
- clean_displaced: clean_loss=3.02667, owner_loss=3.02667, clean_minus_owner=0.0

### chck_100M
- view_changed: clean_loss=3.26586, owner_loss=2.61754, clean_minus_owner=0.64832
- repeat_changed: clean_loss=2.949448, owner_loss=2.039907, clean_minus_owner=0.909541
- breadth_changed: clean_loss=5.266488, owner_loss=4.781175, clean_minus_owner=0.485313
- clean_displaced: clean_loss=2.988545, owner_loss=2.988545, clean_minus_owner=0.0

## Files
- JSON: `experiments/archive/frontier_consolidation/data/persistence_crossloss_analysis/crossloss_results.json`
- CSV: `experiments/archive/frontier_consolidation/data/persistence_crossloss_analysis/crossloss_results.csv`
