# Training and Evaluation Plan

## Training Status at the Time of This Note

### Scale1.75 100M Official-Ladder Endpoint
- **Run dir:** `training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder`
- **What:** Exact deterministic full 100M training of DeBERTa-v2 8×480 with adapter128 scale1.75, legal16k tokenizer, compact-view reinvest corpus, checkpoint_words=1000000 for AoA ladder
- **Seeds:** 43/43022/43023, AdamW lr=0.001, warmup 0.06, weight_decay 0.01, batch256, fixed WWM 0.15
- **Expected:** 2529 steps, 100M words, checkpoints chck_1M through chck_100M
- **Why:** At 80M this model scored cheap7 43.8121, exceeding the ~43.7029 needed for Overall 41.8 if SuperGLUE and AoA stay flat. The only reliable decision requires the exact 100M endpoint.
- **Status:** Training was in progress when this note was written.

### U256 Faithful Experience-Utilization 20M Screen
- **Run dir:** `training/runs/eu_U256_legal16k_seed43022_20M`
- **What:** Stream-order experience-utilization trainer: same corpus/tokenizer/architecture/recipe/seeds as spatial repair route status, changed only training example object so all charged word tokens are visible (14.66M active tokens vs spatial repair route status's ~14.30M, +2.59%)
- **Seeds:** Same as spatial repair route status (43/43022/43023)
- **Why:** Tests whether the legal-coordinate performance loss is partly from row-tail token visibility rather than residual capacity. This is a model/data-structure axis distinct from the adapter route.
- **Status:** Training was in progress when this note was written.

### Planned U256 Evaluation
- **What:** After U256 20M training completion, evaluate cheap7 and compare with spatial repair route status 20M (cheap7 39.6636)
- **Eval script:** `scripts/eval_eu_U256_20M.py`
- **Comparison ref:** spatial repair route status 20M = BLiMP 59.69, Supplement 55.45, EWoK 50.73, Entity 18.65, COMPS 50.26, GlobalPIQA 34.195, Reading 8.67
- **Route signal:** positive if delta > +0.3 cheap7, marginal if within ±0.3, negative if below -0.3
- **Status:** Evaluation remained pending training completion.

## Prepared Evaluation Script

### Full Nine-Column Scale1.75 100M Evaluator
- **Script:** `scripts/full_eval_scale1p75_100M.py`
- **What:** After 100M training, verify the AoA ladder (19 checkpoints), evaluate cheap columns and SuperGLUE/AoA, then use pristine collation for Overall
- **Exact spatial repair route status 100M reference:**
  - BLiMP: 65.871, Supplement: 61.166, EWoK: 50.393, Entity: 27.401, COMPS: 52.008
  - SuperGLUE: 70.280, GlobalPIQA: 36.063, Reading: 8.138, AoA: 0.0
  - **Overall: 41.257770896404615**
- **Target:** Overall ≥ 41.8

## Files Created in active tasks and eval pipeline
- `scripts/stream_order_eu_trainer.py` (EU trainer)
- `scripts/experience_utilization_trainer_ref.py` (chunk library)
- `scripts/eval_eu_U256_20M.py` (U256 evaluation)
- `scripts/full_eval_scale1p75_100M.py` (scale1.75 100M full eval)
- `data/eu_U256_dryrun/` (dry-run metrics confirming +2.59% visibility)

## Scientific Decision Dependencies
1. **U256 20M screen → decide whether to extend to 80M/100M**
   - If positive: extend to full U256 100M with checkpoint_words=1000000
   - If marginal: inspect column-level patterns before committing more GPU
   - If negative: close U256
2. **Scale1.75 100M endpoint → decide whether to run full evaluation**
   - Verify training completion and checkpoint ladder
   - Run full nine-column evaluation
   - If Overall ≥ 41.8: SOTA candidate, prepare for submission
   - If Overall < 41.8: analyze which columns fell short, plan next route
