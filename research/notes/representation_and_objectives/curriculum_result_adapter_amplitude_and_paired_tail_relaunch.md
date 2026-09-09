# curriculum result adapter amplitude and paired tail relaunch — corrected curriculum result, adapter amplitude readout, paired-tail relaunch

## Corrected AdamW word-boundary sequence-length curriculum: completed and read

The corrected curriculum run delivered cleanly (exit 0). Verified geometry/config from run files, not stdout alone:

- `curriculum_geometry_manifest.json` / `scientific_metrics.json`: 647,400 rows, 100,000,000 words, total 4,782 steps, phase geometry 499,280/1,951 (64), 400,830/1,566 (128), 323,700/1,265 (256), all `tail_drops=0`.
- warmup 286 from `warmup_fraction=0.06`; `train_rng_seed=43023`; `min_chunk_tokens=1`; seed 43; init 43022; legal40k tokenizer.
- `hf_model/chck_100M/config.json`: `pad_token_id=3`, `bos_token_id=1`, `eos_token_id=2`, `pos_att_type=["p2c","c2p"]`, `max_relative_positions=256`, 8×480, 45,826,720 params. This is the clean matched DeBERTa-v2 coordinate (unlike the contaminated lamb curriculum route decision LAMB package).
- `loss_last` 2.6460; landmark losses: 20M earlier analysis 3.765 (baseline) vs curriculum phase-0 similar; 50M loss ~2.93 at seq128; 100M 2.646.

Cheap7 (official-compatible evaluator, `experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py`, out `experiments/archive/representation_and_objectives/data/curriculum_cheap7_eval/per_target/curriculum_adamw_8x480_legal40k_seed43022_warmup006.json`):

| column | curriculum | fixed-256 baseline (legal40k accum training completion) | delta |
|---|---:|---:|---:|
| BLiMP | 67.84 | 66.87 | +0.97 |
| Supplement | 61.33 | 58.86-59? (58.86 FW / 63.28 inherited) — matched fixed256 anchor uses legal40k full eval | — |
| EWoK | 50.03 | 51.47 (pristine) | -1.44 |
| Entity | 22.23 | ~ | — |
| COMPS | 51.76 | ~ | — |
| GlobalPIQA | 35.225 (par 18.45 / nonpar 52.0) | ~ | — |
| Reading | 8.055 | ~ | — |
| **cheap7** | **42.3529** | **43.1079** | **-0.755** |

Result: the corrected, fully matched word-boundary 64→128→256 AdamW curriculum scores cheap7 **42.3529**, **-0.755 below** the matched fixed-256 legal40k anchor **43.1079**, and far below the visible leader region ~43.77. GlobalPIQA_parallel stays at 18.45 (the same deep-rank failure), EWoK falls to 50.03 vs pristine 51.47.

**Conclusion:** sequence-length curriculum alone, under the trusted matched AdamW/legal40k coordinate, does not help and mildly hurts. This closes curriculum-alone as a crossing route. It does not by itself say anything about LAMB (still confounded) or about curriculum combined with a different optimizer.

Because curriculum missed the anchor, the held paired tail factor separation design paired-tail continuations are now the next bounded probe (see below), not another fresh 100M endpoint.

## Adapter residual amplitude readout (no training)

live adapter128 20M gained BLiMP/Supplement/COMPS, lost EWoK/GlobalPIQA/Reading (cheap7 39.389). paired tail smoke and adapter ablation decomposed: GlobalPIQA damage is direct residual output; EWoK damage is stock-backbone drift. curriculum result adapter amplitude and paired tail relaunch added an inference-only amplitude sweep on the same trained checkpoint (`experiments/archive/representation_and_objectives/data/adapter_scale_key_columns`), scoring the harmed/benefited key columns:

| scale | key4 (Supp/EWoK/GP/Read) | Supp | EWoK | GPpar | GP | Reading |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 (disabled inf) | 36.911 | 56.12 | 48.98 | 22.33 | 34.165 | 8.38 |
| 0.25 | 36.909 | 55.89 | 49.21 | 22.33 | 34.165 | 8.37 |
| 0.50 | 36.710 | 55.90 | 49.39 | 20.39 | 33.195 | 8.36 |
| 0.75 | 36.659 | 56.27 | 49.32 | 19.42 | 32.71 | 8.34 |
| 1.00 (live) | 36.564 | 56.23 | 49.49 | 18.45 | 32.225 | 8.31 |
| 1.25 | 36.889 | 56.89 | 49.66 | 18.45 | 32.725 | 8.28 |
| 1.50 | 37.005 | 56.89 | 50.17 | 19.42 | 32.71 | 8.25 |

aoa mincontext discrepancy audit/disabled key4 anchor = 37.261; live scale=1 key4 = 36.564.

Readings:
- No inference scale reaches the aoa mincontext discrepancy audit key4 anchor (37.261). Best is scale 1.5 at 37.005, still -0.256.
- GlobalPIQA is maximized at/near zero adapter output (22.33 par at scale ≤0.25), confirming the direct-residual GlobalPIQA damage: amplifying the branch does not recover parallel GP.
- Higher scale (1.25-1.5) recovers EWoK/Supplement but cannot repair GP. So the branch output trades broad linguistic/relational surfaces against GlobalPIQA; no post-hoc amplitude gives a free win.
- Full cheap7 at scale 1.5 (adding BLiMP 60.55, Entity 18.43, COMPS 50.50): still below aoa mincontext discrepancy audit 20M cheap7 anchor 39.6636.

**Conclusion:** post-hoc amplitude scaling of an ordinary joint adapter does not rescue it. This is a mechanistic readout, not a submission tuning rule. It reinforces the paired tail smoke and adapter ablation route judgment: the useful adapter direction is coupling control (stop-gradient input / frozen-reader) that protects the stock trajectory and regulates residual output, not width or amplitude alone.

## Paired-tail relaunch after path-fix

The paired tail factor separation design paired tails were first launched with two wrong paths (nonexistent `data/legal40k_tokenizer` and a nonexistent copied `legal40k_accum_train_examples_seed43022.jsonl`). Both failed before training (HF repo-id validation error / FileNotFound) — operational path errors, not evidence. Corrected paths recovered from the legal40k accum training completion manifest:

- `--example_jsonl experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl`
- `--tokenizer_path experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k`

Dry runs with corrected paths reproduce the exact tail geometry (129,256 examples / 19,965,632 words / 505 steps / source earlier analysis) and LR conventions (residual first update 1.0720866e-4, reheat first update 0.0). Relaunched comparisons:

- residual_low_lr on GPU0 → `training/runs/tail_residual_low_lr_freshmom_from_chck80M`
- reheat_cosine on GPU1 → `training/runs/tail_reheat_cosine_freshmom_from_chck80M`

Both start from the exact same fixed-256 legal40k `chck_80M` (80,034,368 words / earlier analysis), consume identical tail rows with `train_rng_seed=53023` replay-matched masks/dropout and fresh AdamW moments; the only intended difference is LR trajectory.

**Stop rule:** compare both cheap7 with the uninterrupted fixed-256 anchor 43.1079 and the 80M reference. If neither improves the full cheap7 surface, close the optimizer-state reset / LR reheat family; do not extend either tail. If either improves broad columns AND the hard GlobalPIQA/EWoK surfaces, that becomes the next line to develop.
