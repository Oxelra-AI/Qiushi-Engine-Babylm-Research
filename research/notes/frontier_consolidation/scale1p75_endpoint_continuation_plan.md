# residual capacity independent_review synthesis — scale1.75 endpoint continuation plan

## Why this exists

The fixed train-time `adapter_scale=1.75` residual route has an exact reproducible 20M prefix and a nonmonotone 20/30/40/50M cheap-column trajectory. The continuation criterion is that the sign at 80M alone should not decide the route. If 70M/80M evidence is mixed or positive rather than coherently worsening, the scientific endpoint is the official-compatible 100M result.

This note preserves the exact way to continue if warranted, while no new 100M GPU run has been launched yet.

## Key execution fact

The base `masking_curriculum_trainer.py` saves HuggingFace model checkpoints only (`save_pretrained`), not AdamW, scheduler, CUDA RNG, or dataloader state. Therefore an exact continuation from the current 80M model-only checkpoint to the same trajectory's 100M endpoint is not available.

If endpoint continuation is warranted, use a deterministic full run from initialization with the same seeds, stream, model, and 100M LR horizon. It is a rerun of the same trajectory to the official endpoint, not a resumed tail. The prior exact-prefix test shows deterministic reproducibility under the same command family; changing checkpoint cadence should not change model updates because checkpoint saving and dynamics-trace flushing are logging/serialization operations only, but the endpoint run should still verify hashes/metrics at 20M/50M/80M if possible.

## 100M training command, if late evidence warrants

Use a fresh output directory:

`experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder`

Command:

```bash
CUDA_VISIBLE_DEVICES=0 python -B experiments/archive/frontier_consolidation/scripts/adapter_scaled_trainer.py \
  --adapter_bottleneck 128 \
  --adapter_enabled 1 \
  --adapter_scale 1.75 \
  --gpu 0 \
  --example_jsonl experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl \
  --example_jsonl_label adapter128_scale1p75_matched_100Mhorizon_100M_official_ladder \
  --example_jsonl_meta experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json \
  --output_dir experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder \
  --tokenizer_path experiments/archive/frontier_consolidation/data/compliant_tokenizer \
  --tokenizer_label compliant16k_reinvest10M \
  --hidden_size 480 \
  --n_layer 8 \
  --n_head 8 \
  --ffn_mult 4 \
  --seed 43 \
  --extra_init_seed 43022 \
  --train_rng_seed 43023 \
  --batch_size 256 \
  --seq_length 256 \
  --max_seq_length 256 \
  --learning_rate 0.001 \
  --warmup_fraction 0.06 \
  --weight_decay 0.01 \
  --masking_curriculum wwm_fixed \
  --mask_prob_start 0.15 \
  --mask_prob_end 0.15 \
  --checkpoint_words 1000000 \
  --max_word_exposure 100000000 \
  --lr_total_steps 2529 \
  --num_workers 0 \
  --log_every 100 \
  --dynamics_trace_every 500
```

Use `checkpoint_words=1000000` for the official AoA ladder (`chck_1M`..`chck_9M` and `chck_10M`..`chck_100M`). Earlier 20/50/80M diagnostic runs used 10M cadence to save IO and cannot support official AoA by themselves.

## Lowest-cost endpoint evaluation sequence after 100M training

1. Confirm `scientific_metrics.json` exists, `word_exposure=100000000`, `actual_training_steps=2529`, legal tokenizer path/label, same stream hash/label, and checkpoints include the required AoA ladder.
2. First run cheap official-compatible columns at `chck_100M` to compare with spatial repair route status 100M cheap7 43.0057. This is the fastest score-bearing endpoint readout.
3. If cheap7 is not close to the needed endpoint scale, do not spend on full SuperGLUE/AoA unless domain behavior is scientifically valuable enough to inspect. spatial repair route status complete Overall is 41.2577708964 and, if SuperGLUE/AoA matched spatial repair route status, the cheap7 needed for Overall 41.8 is about 43.7029 (+0.6972 over spatial repair route status 100M cheap7).
4. If cheap7 is promising or domain behavior is broad, run full official-compatible evaluation including SuperGLUE and AoA through `evaluate_compliant_endpoint.py`.

## Full official-compatible evaluator command, after cheap endpoint readout justifies it

```bash
python -B experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py \
  --arm reinvest \
  --run-dir experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder \
  --target adapter128_scale1p75_h100M100M_seed43022_official_ladder \
  --endpoint chck_100M \
  --out-root experiments/archive/frontier_consolidation/data/scale1p75_100M_full_eval/eval \
  --collate-root experiments/archive/frontier_consolidation/data/scale1p75_100M_full_eval/collate \
  --gpu 0
```

## How to interpret the 70/80M trigger

Do not launch the 100M run merely because one late point is positive. Launch it if the 70M/80M readout is mixed or positive in the sense that fixed scale1.75 is not showing a widening broad deficit: high-operation Entity/BLiMP/COMPS gains persist or rotate into recoverable domains, and EWoK/GlobalPIQA/Reading are not jointly worsening. Close fixed-scale maturation if late points jointly show negative cheap7 and a growing pressure-axis deficit (EWoK + GlobalPIQA + Reading) unlikely to recover in the final 20M.
