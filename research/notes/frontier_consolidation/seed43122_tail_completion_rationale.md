# seed43122 tail completion rationale seed43122 tail completion rationale

The seed43122 route decision decision correctly read the recovered 70--90M seed43122 evidence, but the own-peak portion was incomplete because 92--100M had not been fully scored. The current action is a bounded completion of the same already-trained scale1.75 seed43122 grid, not a new seed/scale search and not a stabilization experiment.

## Scientific question

Does the unscored seed43122 tail contain a higher selected cheap-task peak than the observed 88M checkpoint?

Observed complete seed43122 endpoints before seed43122 tail completion rationale:

- 70M 43.384286
- 72M 43.330000
- 74M 43.471429
- 76M 43.849286
- 78M 43.723571
- 80M 43.625714
- 82M 43.584286
- 84M 43.726429
- 86M 43.627857
- 88M 43.909286
- 90M 43.664286
- 92M partial: BLiMP 66.99, Supplement 61.37, EWoK 52.08, Entity 25.89, COMPS 51.59, missing GlobalPIQA and Reading.

The fixed reference window 82M->84M->86M is already complete for seed43122, so the already-observed evidence rejects recurrence of the reference signed transition at that coordinate. The remaining gap is whether seed43122's own late peak is actually 88M or a later 92/94/96/98/100M checkpoint.

## Minimum reliable action launched

The missing seed43122 tail rows are being evaluated using the existing official-compatible selected cheap-task evaluator:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/resume_mlm_common_grid.py \
  --job scale1p75_seed43122_dense,experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_seed43122_dense100M,experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43122_dense_common2M \
  --endpoints chck_92M chck_94M chck_96M chck_98M chck_100M \
  --gpu 0
```

This should skip already-complete columns, finish missing GlobalPIQA/Reading for 92M, score the four missing tail endpoints, and reconstruct `selected_trajectory.json` / summary files under the same grid directory. It does not run SuperGLUE or AoA and does not submit anything.

## Interpretation after delivery

After the seed43122 tail evaluation completes:

1. Read the completed evaluation output plus the reconstructed `selected_trajectory_summary.md`.
2. Run `scripts/selected_mlm_integrity_check.py` on `data/selected_trajectory_eval_scale1p75_seed43122_dense_common2M` and require all 16 cheap rows to be complete before treating the tail as closed.
3. If no 92--100M endpoint beats 88M cheap7=43.909286, the seed43122 own-peak gap closes; keep the seed43122 route decision conclusion: scale1.75 residual-capacity late behavior is structurally present but stochastic at signed subtask/item level, and seed43022 chck84 is an endpoint asset rather than a reproducible learning law.
4. If a later endpoint beats 88M, run the existing signed transition readout ready signed-transition tool and transition background for seed43122 fixed/all-window background tool once on the completed all-three grid, so the true seed43122 own-peak window is used. Do not build new analysis tooling.
5. Only after this can the evidence determine whether the next research object is stochastic broad-competence stabilization. Do not spend GPU on same-trajectory averaging or any stabilization candidate before this tail question is settled.

Protected assets remain unchanged: public chck82 displayed 41.94; ordinary chck84 Overall(AoA0)=42.0189129742181 with HF revision `040284de9ac49eee6dc1b30cf65aea97ec17d86e`; coherent86 alpha0.75 projected Overall(AoA0)=42.1210247099666 but redistributive.
