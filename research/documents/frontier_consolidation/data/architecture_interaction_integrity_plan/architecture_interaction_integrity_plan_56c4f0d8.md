# architecture_interaction_integrity_plan

Created UTC: `2026-09-02T11:34:41Z`

Compare compact-minus-repeat under full DeBERTa with compact-minus-repeat under no_disentangle_abs. Survival without c2p/p2c score tensors shows those score terms are not necessary; attenuation implicates the whole removed score-term package plus parameterization and score-composition changes.

- `full_compact_existing`: existing legal full-DeBERTa compact reference; reused only after exact-coordinate checks at `experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2`
- `full_repeat`: new missing legal full-DeBERTa repeat reference at `experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_repeat_deberta100M_seed43022`
- `nodis_compact`: compact arm with c2p/p2c attention-score terms removed and absolute input positions retained at `experiments/archive/frontier_consolidation/training/runs/no_disentangle_abs_compact_deberta100M_seed43022`
- `nodis_repeat`: repeat arm with c2p/p2c attention-score terms removed and absolute input positions retained at `experiments/archive/frontier_consolidation/training/runs/no_disentangle_abs_repeat_deberta100M_seed43022`
