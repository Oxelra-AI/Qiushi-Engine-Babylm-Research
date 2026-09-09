# cue audit answer slot rows: Seed43022 Base Cheap7 and Dose Status

## Base43022 Cheap7 (chck_100M, adapter-scaled, trusted loader)

| Column | Score |
|---|---|
| BLiMP | 68.62 |
| Supplement | 62.90 |
| EWoK | 49.08 |
| Entity | **27.46** |
| COMPS | 52.30 |
| GlobalPIQA | 36.11 |
| Reading | 8.32 |
| **cheap7** | **43.54** |

Entity at 27.46 confirms substantial headroom. This is the column most directly 
targeted by the binding training.

## Dose Status

### Seed43022 (complete)
- base0: complete, cheap7=43.54
- dose21: training complete, cheap7 IN PROGRESS (BLiMP=67.61, Supplement=62.37, rest pending)
- dose25: training complete, cheap7 IN PROGRESS (not started)

### Seed43122 (partially complete)
- dose25: COMPLETED 2026-09-07T04:15:54Z, 100M words, 100 checkpoints, loss=2.608
- dose21: AT 94M, still training (est. 10-15 min remaining)

### Pending measurements
- Seed43022 cheap7: base complete; dose21 in progress.
- Trusted ordinary-fit axes: planned; scoring not started.
- OFF-ALN strict complement: planned; scoring not started.
- Seed43122 training: dose25 done; dose21 near completion.

## Files
- Base cheap7: `data/seed43022_cheap7/base43022/base43022_cheap7_summary.json`
- Dose25 seed43122 run: `training/runs/probe_clean_restatement_dose25_seed43122/`
