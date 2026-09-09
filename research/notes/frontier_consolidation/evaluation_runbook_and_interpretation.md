# evaluation runbook and interpretation evaluation runbook and interpretation guardrails

## Current state

- The two causal GPT architecture-transfer training comparisons remain pending:
  - compact semantic-view causal arm, expected run dir `experiments/archive/frontier_consolidation/training/runs/causal_gpt_compact_seed43022_100M`.
  - matched repeat causal arm, expected run dir `experiments/archive/frontier_consolidation/training/runs/causal_gpt_repeat_seed43022_100M`.
- asset freeze and principles DeBERTa dense runs have already finished and provide 50 checkpoints each:
  - `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_seed43122_dense100M`.
  - `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p25_seed43022_dense100M`.
- Protected reference remains `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder`.

## Expensive-work purpose before launching evaluations

The next GPU evaluations decide whether to continue the asset freeze and principles/173 theory branch, alter it, or close it:

1. **DeBERTa common late grid** decides whether the scale1.75 82M peak is robust under joint init+mask-stream change and whether scale1.25 changes peak timing/width on the seed/mask-matched stream. If the peak is unstable or any gain is one-column redistribution, this branch should not be promoted as a general learning principle. If a lower scale broadens or delays high competence while preserving family balance, it motivates a focused follow-up on schedule/amplitude rather than new data construction.
2. **Causal GPT compact-vs-repeat selected grid** decides whether the compact semantic same-window view mechanism transfers into a decoder-only coordinate. A compact advantage must be broad across checkpoints and families and materially larger than the earlier analysis +0.2216% effective active-token asymmetry. A null/negative result bounds this architecture coordinate only; it does not erase the DeBERTa compact-view evidence.

No leaderboard submission is part of these evaluations.

## DeBERTa selected MLM common grid

Use the same endpoints for all three trajectories:

`chck_70M chck_72M chck_74M chck_76M chck_78M chck_80M chck_82M chck_84M chck_86M chck_88M chck_90M chck_92M chck_94M chck_96M chck_98M chck_100M`

Suggested parallel launch when GPUs are available:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/selected_mlm_checkpoint_eval.py \
  --run-dir experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder \
  --gpu 0 \
  --out-dir experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M \
  --label scale1p75_seed43022_reference \
  --endpoints chck_70M chck_72M chck_74M chck_76M chck_78M chck_80M chck_82M chck_84M chck_86M chck_88M chck_90M chck_92M chck_94M chck_96M chck_98M chck_100M
```

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/selected_mlm_checkpoint_eval.py \
  --run-dir experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_seed43122_dense100M \
  --gpu 0 \
  --out-dir experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43122_dense_common2M \
  --label scale1p75_seed43122_dense \
  --endpoints chck_70M chck_72M chck_74M chck_76M chck_78M chck_80M chck_82M chck_84M chck_86M chck_88M chck_90M chck_92M chck_94M chck_96M chck_98M chck_100M
```

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/selected_mlm_checkpoint_eval.py \
  --run-dir experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p25_seed43022_dense100M \
  --gpu 1 \
  --out-dir experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p25_seed43022_dense_common2M \
  --label scale1p25_seed43022_dense \
  --endpoints chck_70M chck_72M chck_74M chck_76M chck_78M chck_80M chck_82M chck_84M chck_86M chck_88M chck_90M chck_92M chck_94M chck_96M chck_98M chck_100M
```

After all three outputs exist, run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/compare_selected_mlm_trajectories.py \
  --trajectory scale1p75_seed43022_reference=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M/selected_trajectory.json \
  --trajectory scale1p75_seed43122_dense=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43122_dense_common2M/selected_trajectory.json \
  --trajectory scale1p25_seed43022_dense=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p25_seed43022_dense_common2M/selected_trajectory.json \
  --reference-label scale1p75_seed43022_reference \
  --out-dir experiments/archive/frontier_consolidation/data/selected_mlm_trajectory_comparison
```

Interpretation:

- Compare `scale1p25_seed43022_dense` primarily to the protected reference, because they share seed/mask stream and differ only in adapter scale.
- Compare `scale1p75_seed43122_dense` to the reference only as joint init+mask-stream robustness; it cannot separate initialization from WWM mask randomness.
- Use the comparator's cheap7, cheap6(no GlobalPIQA), cheap5(no GlobalPIQA/Reading), EWoK/Entity, and syntax/COMPS summaries. Do not let a GlobalPIQA or Reading spike carry a mechanism conclusion.

## Causal GPT selected grid after training delivery

Endpoints:

`chck_20M chck_50M chck_70M chck_82M chck_90M chck_100M`

Suggested parallel launch when the causal training tasks have delivered:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/selected_causal_checkpoint_eval.py \
  --run-dir experiments/archive/frontier_consolidation/training/runs/causal_gpt_compact_seed43022_100M \
  --label causal_compact --gpu 0 \
  --out-dir experiments/archive/frontier_consolidation/data/selected_causal_eval_compact \
  --endpoints chck_20M chck_50M chck_70M chck_82M chck_90M chck_100M
```

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/selected_causal_checkpoint_eval.py \
  --run-dir experiments/archive/frontier_consolidation/training/runs/causal_gpt_repeat_seed43022_100M \
  --label causal_repeat --gpu 1 \
  --out-dir experiments/archive/frontier_consolidation/data/selected_causal_eval_repeat \
  --endpoints chck_20M chck_50M chck_70M chck_82M chck_90M chck_100M
```

Then compare:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/compare_causal_selected_trajectories.py \
  --compact experiments/archive/frontier_consolidation/data/selected_causal_eval_compact/selected_causal_trajectory.json \
  --repeat experiments/archive/frontier_consolidation/data/selected_causal_eval_repeat/selected_causal_trajectory.json \
  --out-dir experiments/archive/frontier_consolidation/data/causal_compact_repeat_contrast
```

Interpretation:

- A real positive result is sustained compact-over-repeat movement over several checkpoints with at least four of seven cheap columns positive and no dependence on one small/volatile column.
- A tiny compact edge comparable to the +0.2216% active-token asymmetry is weak evidence; a one-checkpoint spike is not a transferable learning principle.
- A null/negative result says compact same-window views may need bidirectional MLM-style learning or DeBERTa-like inductive bias in this implementation; it does not erase the earlier legal DeBERTa compact/reinvestment causal result.

## Checked scripts

- `scripts/selected_mlm_checkpoint_eval.py` compiled in dense and causal evaluation refined plan.
- `scripts/selected_causal_checkpoint_eval.py` had a evaluation runbook and interpretation robustness repair creating per-endpoint dirs before writing wrapper logs; compiled in evaluation runbook and interpretation.
- `scripts/compare_causal_selected_trajectories.py` compiled in evaluation runbook and interpretation.
- `scripts/compare_selected_mlm_trajectories.py` compiled and smoked on synthetic two-point trajectories in evaluation runbook and interpretation.

## earlier analysis update: reference cache seeding and causal evaluation launches

- Both causal selected evaluations are now running:
  - compact selected causal cheap7 to `data/selected_causal_eval_compact` (GPU0).
  - repeat selected causal cheap7 to `data/selected_causal_eval_repeat` (GPU1).
  - After both deliver, compare with `scripts/compare_causal_selected_trajectories.py` (stronger comparator; family/breadth aware) using `--accounting-json data/causal_training_accounting_audit/causal_training_accounting_audit.json`. The older `compare_causal_selected_trajectories.py` is a thinner fallback.

- Causal training accounting (CPU): compact and repeat both charged exactly 100,000,000 legal words, 2230 steps, same neutral tokenizer SHA `e6723383...`. Compact active tokens/epoch 14,587,904 vs repeat 14,555,648 (+0.2216% compact). Repeat has slightly lower training loss at every selected checkpoint (final 3.28013 vs 3.292451). Loss is corpus-fit only; official selected cheap7 decides transfer. See `data/causal_training_accounting_audit/`.

- Reference DeBERTa common-grid cache status (`data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M/eval/per_target/`):
  - `chck_100M`: cache-seeded from exact-hash earlier analysis staged full eval (evaluation runbook and interpretation).
  - `chck_80M`: cache-seeded from exact-hash earlier analysis split payloads via `scripts/seed_reference_cache_from_split.py` (cheap7 43.81214285714286; all seven columns hash-matched to final ladder).
  - `chck_70M`: NOT cache-seeded. Old 70M split payloads lacked GlobalPIQA_nonparallel, so a blocking GPU eval started and was killed by a 300s timeout, leaving a partial empty-tasks cache; that partial file was removed. 70M must be scored fresh in the batched reference-grid GPU run.
  - Remaining reference points needing GPU: `chck_70M chck_72M chck_74M chck_76M chck_78M chck_82M chck_84M chck_86M chck_88M chck_90M chck_92M chck_94M chck_96M chck_98M` (14 points; 80M and 100M cached).

- Adapter-norm context (CPU, `data/adapter_norm_trajectory/`): scale1.25 dense grows larger raw adapter up-weights (up_l2 ~15.0 vs ~13.8) but effective scaled up-norm is ~0.78x the scale1.75 reference (effective up/stock ratio ~0.0815 vs ~0.1044). So scale1.25 does NOT fully compensate via weight growth; its lower effective residual energy is real and its selected trajectory can be read against a genuine residual-energy difference. This is mechanism context only; official selected scores decide the route.

- Each selected DeBERTa common-grid comparison covers 14 points. The selected evaluator reuses cached 80M/100M results.

## causal transfer result synthesis update: causal selected result and DeBERTa common-grid launches

- Both selected causal GPT evaluations completed and passed the causal transfer result synthesis integrity check (`data/causal_selected_integrity_check/`).
- Causal compact-vs-repeat contrast is saved at `data/causal_compact_repeat_selected_contrast/causal_compact_repeat_selected_contrast.{json,md}` and synthesized in `notes/causal_transfer_result_synthesis.md`.
- Scientific result: compact semantic same-window views do **not** show broad architecture-general transfer in the GPT2 causal coordinate. Mean Δcheap7 is only +0.077738 and is not sustained/broad; mean Δcheap6(no GlobalPIQA) is -0.144583, mean Δcheap5(no GlobalPIQA/Reading) is -0.151333, and mean Δrelation_state(EWoK+Entity) is -0.319167. Positive compact endpoints (82M/90M/100M) are carried by GlobalPIQA/Reading rather than broad family improvement. Compact also has +0.2216% active-token asymmetry, so the result is weaker than a true broad transfer signal.
- This bounds the DeBERTa compact-view/reinvestment mechanism to the MLM/bidirectional coordinate in the tested implementation; it does not erase the legal DeBERTa causal evidence for compact views in that coordinate.

- DeBERTa common-grid scoring was launched:
  - protected reference scale1.75 seed43022 on GPU0, output `data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M`. The wrapper should skip cached 80M and 100M; remaining 14 endpoints need fresh scoring.
  - scale1.25 seed43022 dense run on GPU1, output `data/selected_trajectory_eval_scale1p25_seed43022_dense_common2M`, all 16 endpoints.
- Still pending: scale1.75 seed43122 dense common-grid scoring to `data/selected_trajectory_eval_scale1p75_seed43122_dense_common2M`.
- After all three DeBERTa selected trajectories exist, run `scripts/compare_selected_mlm_trajectories.py` with the command above. Interpret scale1.25 only as seed/mask-matched adapter-amplitude contrast; interpret scale1.75 seed43122 only as joint init+mask-stream robustness.


## topology 2x2 scaffold and deberta pending state update: reciprocal/topology route constrained and 2×2 scaffold prepared

The reciprocal-semantic explanation is weakened because DeBERTa non-copy lift is tiny or negative while copied-token and causal-repeat lift dominate. The reciprocal scaffold does not justify new 100M runs. If the DeBERTa triangle scores revive this direction, first isolate the interaction with a low-cost matched 2x2: compact versus repeat crossed with one-way duplicated exposure versus reciprocal both-way exposure, with identical pairs, recurrent dose, filler, exposure and seed.

CPU/file-only artifacts prepared in topology 2x2 scaffold and deberta pending state:

- Scaffold builder: `scripts/build_causal_topology_2x2.py`
- Scaffold: `data/causal_topology_2x2_scaffold/manifest.{json,md}`
- Comparator: `scripts/compare_causal_topology_2x2.py`
- Comparator smoke: `data/topology_2x2_comparator_smoke/out/`
- Synthesis note: `notes/topology_2x2_scaffold_and_deberta_pending_state.md`

Final four arm SHAs after correcting the base-order and filler-label bugs:

- `compact_oneway`: `13213897f9c54fe651ca407db0be4ed03d8df6a5c6f71f390ad76bd813cd0ac5`
- `repeat_oneway`: `81ed5960c80cbc9ef0636a118a4d43017ccbf42c3eff03b22e3b902f5d158666`
- `compact_reciprocal`: `94946452b42c0b3f100612436f7675bd00e3bba154d29c25a32dd9504b979f10`
- `repeat_reciprocal`: `cfc708eb4d352a9892932c2ca441015fcc19956db110a63f1016e8940283de26`

All four use 6,071 selected pairs, 12,142 pair units, 423,512 pair words, 9,576,488 shared filler words, exactly 10M legal words, matching row positions and matching row-word sequences. Topology changes active-token counts by 0 within each semantic arm; compact retains +31,744 active positions/epoch (+124 chunks) vs repeat in both topology cells. Batch size 256 equalizes update count at 223 steps/epoch for all four cells. Any future screen should start at 20M with `--batch-size 256` and endpoints `chck_10M chck_20M`, then use `compare_causal_topology_2x2.py`; extend only if the semantic-by-topology interaction is broad and survives removal of GlobalPIQA/Reading.

The reference and scale1.25 DeBERTa common grids remain pending. The scale1.75 seed43122 dry plan at `data/seed43122_common_grid_launch_plan/selected_eval_plan.json` has no missing endpoints, but its scoring has not been launched.
