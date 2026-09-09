# chck82 endpoint evidence and reproduction state chck_82M vs refreshed public Strict-Small leader

Status: **PASS**
Public leader: `wwm_curriculum_simplification_40k` (`go76dof/wwm_curriculum_simplification_40k`), Overall 41.800000
Candidate: `chck_82M`, hardened repeated Overall 41.942481, margin +0.142481

| column | candidate | public leader | delta |
|---|---:|---:|---:|
| BLiMP | 68.491284 | 67.200000 | +1.291284 |
| Supplement | 62.937811 | 56.010000 | +6.927811 |
| EWoK | 50.055453 | 56.070000 | -6.014547 |
| Entity | 28.314042 | 28.450000 | -0.135958 |
| COMPS | 52.191175 | 53.570000 | -1.378825 |
| SuperGLUE | 69.766181 | 69.790000 | -0.023819 |
| GlobalPIQA | 37.577670 | 39.670000 | -2.092330 |
| Reading | 8.148714 | 5.420000 | +2.728714 |
| AoA | 0.000000 | 0.000000 | +0.000000 |
| **Overall** | **41.942481** | **41.800000** | **+0.142481** |

Candidate recomputed Overall: `41.942481167386`; cheap7: `43.959449876452`.
Measurement repeatability Overall delta: `0.00022695351662349594`; max column delta: `0.005893446487505116`.

Scientific reading: the endpoint is above the visible public Overall target, but still trails on EWoK/COMPS/GlobalPIQA; endpoint preservation and mechanism interpretation must remain separate.

JSON: `experiments/archive/representation_and_objectives/data/chck82_public_leader_comparison/chck82_public_leader_comparison.json`
