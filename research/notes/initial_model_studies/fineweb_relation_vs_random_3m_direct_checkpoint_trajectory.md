# fineweb relation vs random 3m direct checkpoint trajectory repaired direct-checkpoint FineWeb relation 3M trajectory

Evidence JSON: `experiments/archive/initial_model_studies/data/fineweb_relation_vs_random_3m_direct_checkpoint_trajectory.json`

Evaluation repair: each local checkpoint directory is passed directly as `model_path_or_name`; no `revision_name` is used. This corrects the fineweb random quality 3m profile local-parent/revision loading artifact.

| checkpoint | ΔBLiMP | ΔSupplement | ΔEWoK | ΔEntity | ΔCOMPS | ΔReading | random hash | relation hash |
|---|---:|---:|---:|---:|---:|---:|---|---|
| chck_1M | +0.6300 | +1.2000 | +3.6300 | +0.1200 | -0.0300 | -0.4900 | b1278fb5a535f294 | 2b58cb5ff612dc4c |
| chck_2M | -0.0100 | -2.8000 | -2.0900 | -0.0500 | -0.2300 | +0.0900 | 739e7f7b1699fe12 | fd85e7ae50528e10 |
| chck_3M | -0.9000 | +1.6000 | +0.5400 | +0.3000 | +0.3100 | +0.0850 | 9c5879d4755ec58b | e3714f87f9c76170 |

## Absolute scores

### chck_1M

| arm | BLiMP | Supplement | EWoK | Entity | COMPS | Reading |
|---|---:|---:|---:|---:|---:|---:|
| random_quality | 53.6600 | 44.0000 | 46.7300 | 17.6500 | 49.9700 | 6.7800 |
| relation_explicit | 54.2900 | 45.2000 | 50.3600 | 17.7700 | 49.9400 | 6.2900 |

### chck_2M

| arm | BLiMP | Supplement | EWoK | Entity | COMPS | Reading |
|---|---:|---:|---:|---:|---:|---:|
| random_quality | 53.4400 | 46.0000 | 50.9100 | 17.5300 | 50.0000 | 6.6600 |
| relation_explicit | 53.4300 | 43.2000 | 48.8200 | 17.4800 | 49.7700 | 6.7500 |

### chck_3M

| arm | BLiMP | Supplement | EWoK | Entity | COMPS | Reading |
|---|---:|---:|---:|---:|---:|---:|
| random_quality | 54.2200 | 44.8000 | 49.6400 | 17.5700 | 49.9600 | 6.7550 |
| relation_explicit | 53.3200 | 46.4000 | 50.1800 | 17.8700 | 50.2700 | 6.8400 |

