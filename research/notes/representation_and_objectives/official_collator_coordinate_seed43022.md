# official collator coordinate seed43022 — Single unmodified-collator official coordinate for compact_view_reinvest seed43022

## Findings
The 42.0868 compact reinvest full eval summary scalar has now been reproduced through **one unmodified upstream
BabyLM collator coordinate**, with the two known non-pristine components corrected.
The corrected, official-path Overall is **42.0331347900748**, still **+0.2331** over the
visible leader (41.8).

- Collator: fresh checkout at commit `6f825c291e2c4c78ad33b1935fd64d45f52642dc`
  (`experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict`),
  no edits to `evaluation_pipeline` or `scripts`.
- Staged results tree + collated JSON:
  `experiments/archive/representation_and_objectives/data/pristine_collate_seed43022/results/hf_model/all_full_preds_and_fast_scores_mlm.json`
- Summary + per-column detail:
  `experiments/archive/representation_and_objectives/data/pristine_collate_seed43022/pristine_collate_seed43022_summary.json`

The unmodified collator accepted every column: **no null keys**, EWoK total **7618**,
AoA **8005 rows per checkpoint** across 19 checkpoints. This clears the two earlier analysis/035
blockers: the 6560-row min_context=20 AoA and the 6666-row stale-local EWoK are both gone.

## Official-path nine columns (seed43022 chck_100M)
| Column | Official-path | compact reinvest full eval summary record | delta | cause |
|---|---|---|---|---|
| BLiMP | 66.8723 | 66.87 | +0.0023 | rounding |
| Supplement | 63.2758 | 63.28 | -0.0042 | rounding |
| EWoK | 53.5366 | 53.67 | **-0.1334** | pristine 7618-row official EWoK vs stale-local 6666 |
| Entity | 27.7457 | 27.75 | -0.0043 | rounding |
| COMPS | 51.9688 | 51.97 | -0.0012 | rounding |
| SuperGLUE | 71.0360 | 71.3811 | **-0.3450** | leaderboard primary metric (f1 for MRPC/QQP) vs all-accuracy mean |
| GlobalPIQA | 35.6214 | 35.62 | +0.0014 | rounding |
| Reading | 8.2416 | 8.24 | +0.0016 | rounding |
| AoA | 0.0 | 0.0 | 0.0 | official min_context=0, 504 words, 8005 contexts, curve fitness 0.0 |
| **Overall** | **42.0331** | 42.0868 | **-0.0537** | EWoK -0.0148 + SuperGLUE -0.0384 |

The two corrections are arithmetically consistent: predicted 42.0336 ≈ scored 42.0331.

## SuperGLUE aggregation convention (resolved)
The pristine `strict/scripts/print_results_table.py` FINETUNE_METRIC is explicitly
"matching the leaderboard definition in about.py": **f1 for MRPC and QQP, accuracy for
boolq/mnli/multirc/rte/wsc**. Under this authoritative convention SuperGLUE = 71.0360.
The compact reinvest full eval summary record used mean-of-accuracy = 71.3811. Both conventions leave Overall
above 41.8:
- primary-metric (authoritative): Overall 42.0331, margin +0.2331
- all-accuracy (compact reinvest full eval summary): Overall 42.0723 with pristine EWoK, margin +0.2723

## AoA remains 0.0 on the official path
Official min_context=0 rerun (`s35_t26_tool1`) gave raw curve fitness 0.0, n_words=241,
8005 contexts/checkpoint. AoA=0.0 is therefore an authentic official-path measurement,
not an artifact of the earlier min_context=20 subset. Because the margin is carried
partly by non-AoA columns and AoA is already at the floor, AoA cannot fall further.

## GlobalPIQA data provenance note
The official Strict-Evals HF snapshot does not ship `global_piqa_*/eng_latn.jsonl`; the
official `global_piqa/dl.py` generates them from `mrlbenchmarks/global-piqa-*`. This
step scored GlobalPIQA against the already-generated INITIAL_MODEL_STUDIES files (parallel 103 rows sha
`cb9513ed...`, nonparallel 100 rows sha `aeec831d...`), which match the sizes the
collator's `_check_size_global_piqa` expects (103 / 100). Recorded in the summary.

## Robustness: seed43122 is materially weaker on the fast surface
`s25_t33_tool1` trained seed43122 to 100M (rc=0). Fast no-AoA equal7 = 42.9329 vs
seed43022 44.2886 (EWoK -3.73, Supplement -3.20, Entity -1.39, BLiMP -0.88, COMPS -0.43,
GlobalPIQA -0.485, Reading +0.625). The SOTA margin is carried by a favorable seed.
the fast surface has mispredicted full behavior before, so the
lowest-cost decisive next experiment is the **full official-compatible evaluation of the
already-trained seed43122**, not packaging seed43022 alone or cancelling replication.

## State
- Corrected official coordinate: seed43022 Overall **42.0331** (primary-metric SuperGLUE),
  reproduced end-to-end through one unmodified collator. This is the submission coordinate.
- Remaining: full seed43122 evaluation (robustness); teacher/tool/tokenizer + license
  ledger; source ancestry of the changed-block score-bearing overlap rows; portable
  path-independent model package.
