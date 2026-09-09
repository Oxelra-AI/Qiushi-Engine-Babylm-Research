# coherence margin signal isolation coherence-margin isolation readiness

Status: **COMPLETE**

| endpoint | role | model exists | payload exists | cheap7 complete | cheap7 |
|---|---|---:|---:|---:|---:|
| cohmargin4M_lambda0p10 | lambda>0 pilot under evaluation | True | True | True | 36.692142857142855 |
| scale1p75_chck4M_ref | ordinary charged-word reference, not sufficient isolate | True | False | False | NA |
| cohmargin4M_lambda0 | same-charge same-row no-margin-gradient isolate | False | False | False | NA |

## Commands not run

### train lambda-zero isolate if pilot score warrants it

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/coherence_margin_trainer.py --output_dir experiments/archive/frontier_consolidation/training/runs/cohmargin4M_lambda0_scale1p75_seed43022 --max_word_exposure 4000000 --view_charge_multiplier 2.0 --checkpoint_words 1000000 --micro_batch_size 128 --grad_accum_steps 2 --adapter_scale 1.75 --margin_lambda 0.0 --margin 0.20 --disrupt_span_tokens 8 --lr_total_steps 1265 --seed 43 --train_rng_seed 43022 --gpu <free_gpu>
```

### evaluate cheap7 for scale1p75_chck4M_ref

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py --arm reinvest --run-dir experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder --target scale1p75_standard_chck4M_ref --endpoint chck_4M --out-root experiments/archive/frontier_consolidation/data/scale1p75_chck4M_ref_eval --collate-root experiments/archive/frontier_consolidation/data/scale1p75_chck4M_ref_collate --gpu <free_gpu> --columns BLiMP Supplement EWoK Entity COMPS GlobalPIQA_parallel GlobalPIQA_nonparallel Reading
```

### evaluate/complete cheap7 for scale1p75_chck4M_ref

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py --arm reinvest --run-dir experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder --target scale1p75_standard_chck4M_ref --endpoint chck_4M --out-root experiments/archive/frontier_consolidation/data/scale1p75_chck4M_ref_eval --collate-root experiments/archive/frontier_consolidation/data/scale1p75_chck4M_ref_collate --gpu <free_gpu> --columns BLiMP Supplement EWoK Entity COMPS GlobalPIQA_parallel GlobalPIQA_nonparallel Reading
```


An encouraging coherence-margin pilot score cannot be escalated to 20M until the same-charge same-row lambda-zero arm shows the official movement and NLL separation are caused by the margin gradient rather than exposure/RNG/disruption geometry.

JSON: `experiments/archive/frontier_consolidation/data/cohmargin_isolation_readiness/cohmargin_isolation_readiness.json`
