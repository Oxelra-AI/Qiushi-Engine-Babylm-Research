# muon50 gp and relation pair synthesis — Muon→AdamW 50M artifact harvest

The 40M comparison could not test AdamW recovery for `muon40toadamw`, because 40M is the switch boundary. This table uses the deepest common existing 50M checkpoints without launching new training.

| arm | role | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read | cheap7 | Δcheap7 vs AdamW50 | Δcheap7 vs cont-Muon50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adamw50 | matched legal compact-view AdamW reference at 50M | 63.80 | 59.11 | 48.46 | 26.26 | 51.61 | 36.58 | 7.990 | 41.9729 | +0.0000 | -0.5936 |
| muon50_cont | continuous matched-decay Muon reference at 50M | 64.15 | 59.02 | 52.35 | 26.11 | 51.42 | 37.59 | 7.325 | 42.5664 | +0.5936 | +0.0000 |
| muon20toadamw50 | 20M Muon followed by about 30M AdamW recovery/consolidation | 64.46 | 59.54 | 50.11 | 25.41 | 51.46 | 38.61 | 8.075 | 42.5229 | +0.5500 | -0.0436 |
| muon40toadamw50 | 40M Muon followed by about 10M AdamW recovery after the observed handoff shock | 63.68 | 60.30 | 50.12 | 26.63 | 51.65 | 34.15 | 6.705 | 41.8907 | -0.0821 | -0.6757 |

Interpretive boundary: negative results close only these abrupt switch artifacts with empty AdamW hidden-matrix moments, not smoother moment-preserving or blended consolidation.
Summary JSON: `experiments/archive/representation_and_objectives/data/muon_switch_50m_eval/muon_switch_50m_summary.json`
CSV: `experiments/archive/representation_and_objectives/data/muon_switch_50m_eval/muon_switch_50m_scores.csv`
