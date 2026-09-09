# muon50 gp and relation pair synthesis — Muon-switch GlobalPIQA 40M→50M synthesis

This file compares the completed 40M and 50M GlobalPIQA hard-rank readouts row-by-row. It is evaluation-only and must not be used to tune an official-example scorer.

| arm | mode | acc40 | acc50 | Δacc | hard52 acc40 | hard52 acc50 | Δhard52 | hard52 margin40 | hard52 margin50 | Δmargin | gained | lost | rank↑ | rank↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adamw | parallel | 26.21 | 28.16 | +1.94 | 1.92 | 3.85 | +1.92 | 1.730 | 1.762 | +0.031 | 7 | 5 | 24 | 20 |
| adamw | nonparallel | 51.00 | 45.00 | -6.00 |  |  |  |  |  |  | 7 | 13 | 7 | 13 |
| continuous_muon | parallel | 22.33 | 27.18 | +4.85 | 1.92 | 5.77 | +3.85 | 1.872 | 1.849 | -0.023 | 10 | 5 | 30 | 18 |
| continuous_muon | nonparallel | 45.00 | 48.00 | +3.00 |  |  |  |  |  |  | 11 | 8 | 11 | 8 |
| muon20toadamw | parallel | 20.39 | 26.21 | +5.83 | 1.92 | 5.77 | +3.85 | 1.841 | 1.699 | -0.142 | 13 | 7 | 29 | 19 |
| muon20toadamw | nonparallel | 56.00 | 51.00 | -5.00 |  |  |  |  |  |  | 8 | 13 | 8 | 13 |
| muon40toadamw | parallel | 22.33 | 23.30 | +0.97 | 1.92 | 7.69 | +5.77 | 1.872 | 1.680 | -0.192 | 7 | 6 | 22 | 19 |
| muon40toadamw | nonparallel | 45.00 | 45.00 | +0.00 |  |  |  |  |  |  | 9 | 9 | 9 | 9 |

Interpretation:
- At 50M, neither abrupt switch arm solves the GlobalPIQA_parallel hard-rank problem. Muon20→AdamW recovers parallel from its 40M trough but remains below AdamW50 on parallel while only retaining nonparallel breadth. Muon40→AdamW after ~10M AdamW recovery remains poor on both parallel and nonparallel.
- Hard52 accuracy remains very low and hard52 margins remain deep, so this evidence supports stopping the exact abrupt empty-moment handoff if broad/EWoK readouts show the same tradeoff.

JSON: `experiments/archive/representation_and_objectives/data/muon_gp_40m_50m_synthesis/muon_gp_40m_50m_synthesis.json`
Mode delta CSV: `experiments/archive/representation_and_objectives/data/muon_gp_40m_50m_synthesis/muon_gp_40m_50m_mode_deltas.csv`
Row delta CSV: `experiments/archive/representation_and_objectives/data/muon_gp_40m_50m_synthesis/muon_gp_40m_50m_row_deltas.csv`
