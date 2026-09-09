# stabilization precheck while grids pending stabilization precheck while DeBERTa grids are pending

## Active research state

The score-bearing endpoint branch is already protected separately:

- Public submitted fallback: scale1.75 `chck_82M`, displayed Overall 41.94.
- New ordinary-training endpoint branch: scale1.75 `chck_84M`, public HF model `leslie721007/babylm-strict-small-scale1p75-chck84` revision `040284de9ac49eee6dc1b30cf65aea97ec17d86e`, projected Overall(AoA0) 42.0189129742181.
- Strongest local endpoint carrier remains coherent86 alpha0.75 projected Overall(AoA0) 42.1210, but its evidence points to amplitude-controlled redistribution rather than a transferable learning principle.

The decisive science dependencies are still running under managed GPU execution:

- resume missing reference/scale1.25 DeBERTa common-grid rows.
- score the scale1.75 seed43122 common grid.

Directional-fork results remained unavailable at the time of this note.

## What stabilization precheck while grids pending added

### 1. Same-trajectory late-weight parameter geometry

Script:

- `experiments/archive/frontier_consolidation/scripts/late_weight_geometry_probe.py`

Output:

- `experiments/archive/frontier_consolidation/data/late_weight_geometry_probe/late_weight_geometry_probe.json`
- `research/documents/frontier_consolidation/data/late_weight_geometry_probe/late_weight_geometry_probe.md`

Main numbers:

- Reference 82M→84M vs 84M→86M update cosine: total `0.078230`, adapter `0.061092`, backbone+heads `0.078720`.
- Mean 2M update L2 and mean successive update cosine are almost the same across all three trained trajectories:
  - reference scale1.75 seed43022: mean L2 `1.837740`, mean cosine `0.101629`, adapter mean cosine `0.088860`.
  - scale1.25 seed43022: mean L2 `1.843914`, mean cosine `0.101662`, adapter mean cosine `0.089062`.
  - scale1.75 seed43122: mean L2 `1.840498`, mean cosine `0.100553`, adapter mean cosine `0.086355`.
- The built chck84 endpoint carrier validation average `reference_scale1p75_seed43022__center_80_82_84_uniform` moves from the 84M checkpoint in a direction with cosine `-0.057732` against 84M→86M and `-0.038974` against 84M→100M. It is therefore geometrically a low-pass point backing away from later-training drift, not an extrapolation further into it.

Interpretation: parameter geometry supports treating same-trajectory late averaging as a coherent low-pass stabilization hypothesis if the seed43122 grid shows late-phase instability. It does not show BabyLM competence.

### 2. CPU legal-corpus masked-LM sanity probe

Script:

- `experiments/archive/frontier_consolidation/scripts/late_average_training_mlm_probe.py`

Output:

- `experiments/archive/frontier_consolidation/data/late_average_training_mlm_probe/late_average_training_mlm_probe.json`
- `research/documents/frontier_consolidation/data/late_average_training_mlm_probe/late_average_training_mlm_probe.md`
- row-level records: `experiments/archive/frontier_consolidation/data/late_average_training_mlm_probe/row_level_model_nll_records.json`

Probe details:

- deterministic whole-word-style masking on a balanced legal-corpus sample;
- `202` rows, `6268` masked tokens, mean sequence length `208.35`;
- trusted-code loading for all checked models.

Mean masked-token NLL on this sample:

| model | mean NLL |
|---|---:|
| `chck_100M` | 2.848895 |
| `chck_86M` | 2.860995 |
| `avg_80_82_84_uniform` | 2.868350 |
| `chck_84M` | 2.873649 |
| `chck_82M` | 2.878524 |
| `chck_80M` | 2.882761 |

The average is not a nonlinear weight-space failure on this ordinary MLM surface: it is slightly better than `chck_84M` on sample NLL (`-0.005299`) and lower than the arithmetic mean of its three endpoint NLLs (`-0.009962`); it has lower row-level NLL than `chck_84M` on 56.93% of sampled rows.

Important scientific meaning: this only checks functional smoothness and collapse risk. It also re-demonstrates why MLM NLL cannot select the BabyLM endpoint: `chck_100M` is best on this probe while known official-compatible competence is worse than the 82M/84M late peak.

### 3. Dry selected-eval path for the built average

Script:

- `experiments/archive/frontier_consolidation/scripts/plan_average_candidate_selected_eval.py`

Output:

- `experiments/archive/frontier_consolidation/data/average_candidate_selected_eval_plan/average_candidate_selected_eval_plan.json`
- `research/documents/frontier_consolidation/data/average_candidate_selected_eval_plan/average_candidate_selected_eval_plan.md`

The dry plan creates a run-dir view at:

- `experiments/archive/frontier_consolidation/data/average_candidate_selected_eval_plan/run_view/hf_model/chck_84M`

pointing to the built average candidate:

- `experiments/archive/frontier_consolidation/data/late_weight_average_scaffold/candidates/reference_scale1p75_seed43022__center_80_82_84_uniform`

A dry run of the existing selected evaluator found no missing endpoint. The future one-endpoint command is recorded in the plan, writing to:

- `experiments/archive/frontier_consolidation/data/avg80_82_84_selected_eval_if_authorized`

No selected BabyLM score was produced for the average.

## How to use this later

The next scientific decision should still come from delivered evidence:

1. Read the resumed DeBERTa and seed43122 common-grid results once complete.
2. Run `scripts/selected_mlm_integrity_check.py` on affected selected-grid output directories.
3. Run `scripts/interpret_deberta_common_grid.py` and `scripts/compare_selected_mlm_trajectories.py` over reference, scale1.25, and seed43122 grids.
4. Interpret seed43122 as joint initialization plus mask-stream robustness; interpret scale1.25 as seed/mask-matched adapter-scale contrast.

If seed43122 reproduces a similar broad late phase, the science should focus on residual-capacity learning dynamics rather than same-trajectory averaging. If seed43122 does not reproduce it, the chck84 endpoint carrier validation/stabilization precheck while grids pending average scaffold is now mechanically and functionally sane enough for one selected cheap-task score as a low-cost stabilization probe, before any new training. Its result must be compared to `chck_82M`, `chck_84M`, `chck_86M`, and `chck_100M` across cheap7, cheap6 without GlobalPIQA, cheap5 without GlobalPIQA/Reading, and EWoK/Entity; a likelihood improvement alone is not meaningful.

No leaderboard submission, model upload, or new GPU work was performed in stabilization precheck while grids pending.
