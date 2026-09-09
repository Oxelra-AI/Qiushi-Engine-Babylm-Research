# muon switch 40m globalpiqa margin — Muon→AdamW 40M artifact harvest

switch runs currently expose checkpoints only through 40M, so this is a same-exposure 40M comparison rather than the intended 70M/80M mature comparison.

| arm | role | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read | cheap7 | Δcheap7 vs AdamW40 | Δcheap7 vs cont-Muon40 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adamw40 | matched legal compact-view AdamW reference | 63.19 | 57.60 | 48.23 | 26.81 | 52.44 | 38.61 | 8.675 | 42.2214 | +0.0000 | +0.8221 |
| muon40_cont | continuous matched-decay Muon reference at the same 40M exposure | 61.67 | 59.83 | 51.66 | 24.72 | 50.81 | 33.66 | 7.440 | 41.3993 | -0.8221 | +0.0000 |
| muon20toadamw40 | tests whether the strong 20M Muon geometry survives 20M of AdamW consolidation | 62.55 | 57.88 | 51.01 | 23.87 | 51.71 | 38.20 | 7.795 | 41.8586 | -0.3629 | +0.4593 |
| muon40toadamw40 | switch-boundary control; nearly continuous Muon before AdamW has real post-switch exposure | 61.67 | 59.83 | 51.66 | 24.72 | 50.81 | 33.66 | 7.440 | 41.3993 | -0.8221 | +0.0000 |

Summary JSON: `experiments/archive/representation_and_objectives/data/muon_switch_40m_eval/muon_switch_40m_summary.json`
CSV: `experiments/archive/representation_and_objectives/data/muon_switch_40m_eval/muon_switch_40m_scores.csv`
