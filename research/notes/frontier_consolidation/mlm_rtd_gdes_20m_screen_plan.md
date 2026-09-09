# mlm rtd gdes 20m screen plan — MLM+RTD-GDES 20M screen plan

## Scientific purpose

The active goal remains a fully legal BabyLM Strict-Small Overall SOTA. The current best legal endpoint, spatial repair route status compact-view-reinvest, is 41.257770896404615 Overall and remains below the visible 41.8 target. The protected corpus mechanism is compact semantic second views plus reinvested source diversity; the missing mechanism is no longer simple data dose, tokenizer seam repair, checkpoint recombination, optimizer tail reset, or benchmark-shaped masking.

earlier analysis produced the first positive pre-training mechanism evidence for a genuinely different learning-signal route:

- trained balanced RTD head on frozen 80M encoder: hard held-out balanced accuracy 0.6844, AUROC 0.7584;
- random corruptions easier by +0.0698 balanced accuracy and +0.1911 AUROC, so model-sampled corruptions are less shortcut-detectable and require more contextual judgment;
- all 8 trunk layers had positive MLM-vs-RTD gradient cosine, mean +0.0429;
- RTD/MLM trunk gradient norm ratio mean 0.241 at lambda 1.0;
- word embeddings and relative embeddings should be protected by GDES, especially rel_embeddings cosine -0.1876.

The 20M run tests whether dense all-token plausibility supervision can improve early generalization in a way that is broader than the closed objective routes. It is a bounded route admission screen, not an endpoint.

## Expensive-work admission

This GPU work decides whether the MLM+RTD-GDES route should:

1. continue toward a 100M legal endpoint if it gives broad cheap-column improvement over the matched spatial repair route status 20M baseline, especially without damaging Supplement/EWoK/Reading;
2. stop if it rotates competence in the familiar pattern or is clearly worse;
3. be scientifically reconsidered if it is neutral despite preserving the MLM path.

The cost is minimized by training only 20M counted words, because spatial repair route status legal chck_20M already has cheap-column scores. A full 100M run is not justified before a broad 20M signal. Full official SuperGLUE/AoA evaluation is not part of this screen.

## Run launched



Command, run from `cwd=experiments/archive/frontier_consolidation`:

```bash
CUDA_VISIBLE_DEVICES=0 python -B scripts/mlm_rtd_gdes_trainer.py \
  --output_dir training/runs/mlm_rtd_lambda1_seed43022_20M \
  --example_jsonl data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl \
  --tokenizer_path data/compliant_tokenizer \
  --max_word_exposure 20000000 \
  --lr_total_steps 2529 \
  --rtd_lambda 1.0 \
  --rtd_temperature 1.0 \
  --checkpoint_words 5000000 \
  --seed 43 \
  --extra_init_seed 43022 \
  --train_rng_seed 43023 \
  --log_every 25
```

The original spatial repair route status legal trainer used the same JSONL stream, tokenizer, architecture, seed trio, AdamW betas, LR, 100M schedule horizon, fixed WWM p=0.15, batch256, and seq256. The scientific change is the RTD auxiliary with GDES-blocked `word_embeddings` and `rel_embeddings`.

Small comparison detail: spatial repair route status's saved `chck_20M` in the existing 100M run is a checkpoint after threshold crossing at 20,008,711 words, while the new bounded trainer consumes an exact 20,000,000-word row boundary. The difference is only 8,711 words (0.044% of 20M) and should not decide a broad route signal, but it must be remembered when reading tiny score differences.

## Evaluation plan after training

Run cheap official-compatible evaluation only:

```bash
python -B experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py \
  --arm reinvest \
  --target mlm_rtd_lambda1_seed43022_20M \
  --run-dir experiments/archive/frontier_consolidation/training/runs/mlm_rtd_lambda1_seed43022_20M \
  --endpoint chck_20M \
  --out-root experiments/archive/frontier_consolidation/data/mlm_rtd_20M_eval \
  --collate-root experiments/archive/frontier_consolidation/data/mlm_rtd_20M_collate \
  --gpu 1 \
  --columns BLiMP Supplement EWoK Entity COMPS GlobalPIQA_parallel GlobalPIQA_nonparallel Reading
```

Then summarize:

```bash
python -B experiments/archive/frontier_consolidation/scripts/summarize_mlm_rtd_20m.py
```

Primary comparator: `data/legal20m_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_20M.json`.

## Decision interpretation

- Strong continuation evidence: cheap7 rises by at least about +0.35 at 20M and the historically fragile Supplement/EWoK/Reading trio does not show material damage. This would not prove SOTA, but it would justify a 100M endpoint because compact-view gains emerge late and dense supervision may alter early representation formation.
- Stop/rebuild evidence: cheap7 is negative, or Supplement/EWoK/Reading lose about a point or more. Prior screens show ordinary maturation cannot rescue a route already moving those columns backward under a comparable coordinate.
- Mixed small movement: requires analysis of mechanism and score anatomy before any longer training; no automatic 100M run.

The FW-family results further narrow that route: compact/interleaved/row-block 100M cheap7 remain below the needed surface and do not justify full official evaluation. This supports using GPU now on the distinct MLM+RTD-GDES learning-signal route rather than more FW allocation variants.
