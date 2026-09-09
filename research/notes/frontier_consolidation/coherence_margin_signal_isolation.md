# coherence margin signal isolation coherence-margin signal isolation status

The reopen structured context margin route coherence-margin pilot is still under official cheap7 evaluation, but coherence margin signal isolation CPU mechanism probes already constrain its interpretation.

## What was checked

I built `scripts/cohmargin_nll_gap_probe.py`, which loads a trusted `AdapterDebertaV2ForMaskedLM`, applies the same WWM masking and `make_disrupted_context` block-shuffle operator as `coherence_margin_trainer.py`, and measures masked-target `NLL(disrupted context) - NLL(coherent context)` on fixed legal stream rows. Positive values mean the model assigns lower NLL to the coherent context for the same masked targets. This is a mechanism probe, not official evaluation.

Probe outputs:

- `data/cohmargin_nll_gap_pilot_train64/cohmargin_nll_gap_probe.{json,md}`
- `data/cohmargin_nll_gap_pilot_holdout64/cohmargin_nll_gap_probe.{json,md}`
- `data/cohmargin_nll_gap_ref2m_train64/cohmargin_nll_gap_probe.{json,md}`
- `data/cohmargin_nll_gap_ref2m_holdout64/cohmargin_nll_gap_probe.{json,md}`
- `data/cohmargin_nll_gap_ref4m_train64/cohmargin_nll_gap_probe.{json,md}`
- `data/cohmargin_nll_gap_ref4m_holdout64/cohmargin_nll_gap_probe.{json,md}`
- Summary: `data/cohmargin_nll_gap_comparison/cohmargin_nll_gap_comparison.{json,md}`

## Main numbers

| probe | bad-minus-coherent NLL gap | SE | z | positive fraction |
|---|---:|---:|---:|---:|
| reopen structured context margin route margin pilot, first 64 train rows | +0.000975 | 0.000342 | +2.85 | 0.5070 |
| ordinary scale1.75 chck2M, same rows | -0.000453 | 0.000218 | -2.08 | 0.4945 |
| ordinary scale1.75 chck4M, same rows | +0.025010 | 0.004073 | +6.14 | 0.5547 |
| reopen structured context margin route margin pilot, 64 rows just after 1,999,862 coherent training words | +0.000326 | 0.000365 | +0.89 | 0.5016 |
| ordinary scale1.75 chck2M, same holdout rows | +0.000080 | 0.000157 | +0.51 | 0.5128 |
| ordinary scale1.75 chck4M, same holdout rows | +0.017922 | 0.003463 | +5.18 | 0.5272 |

The margin pilot is close to the coherent-word matched ordinary 2M reference and far below the charged-word matched ordinary 4M reference. This means the 4M-charged pilot has not yet opened a large coherent-over-disrupted target-likelihood separation. Any encouraging official score would need to be attributed only after comparison to a same-charge same-row lambda-zero arm, not to ordinary chck4M.

## Exact lambda-zero isolation arm

The next isolation arm, if the pilot official cheap7 is promising enough to justify it, should be:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/coherence_margin_trainer.py \
  --output_dir experiments/archive/frontier_consolidation/training/runs/cohmargin4M_lambda0_scale1p75_seed43022 \
  --max_word_exposure 4000000 --view_charge_multiplier 2.0 --checkpoint_words 1000000 \
  --micro_batch_size 128 --grad_accum_steps 2 --adapter_scale 1.75 \
  --margin_lambda 0.0 --margin 0.20 --disrupt_span_tokens 8 \
  --lr_total_steps 1265 --seed 43 --train_rng_seed 43022 --gpu <free_gpu>
```

Do **not** remove disruption construction from the lambda-zero arm. The trainer uses one torch generator for masking and block shuffling; constructing the disrupted view consumes generator draws. Running the same trainer with `--margin_lambda 0.0` preserves the same row order, view charge, optimizer schedule, mask RNG trajectory, disruption RNG consumption, and logging geometry, while eliminating only the margin gradient. Skipping the disrupted view would change later masks and would no longer be the clean isolate.

The follow-up probe after lambda-zero training should use exactly:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/cohmargin_nll_gap_probe.py \
  --label cohmargin4M_lambda0_train64 \
  --model-path experiments/archive/frontier_consolidation/training/runs/cohmargin4M_lambda0_scale1p75_seed43022/hf_model/final \
  --out-dir experiments/archive/frontier_consolidation/data/cohmargin_nll_gap_lambda0_train64 \
  --num-rows 64 --batch-size 8 --device cpu --probe-seed 15900
```

and similarly with `--skip-coherent-words 1999862` for the immediate post-prefix holdout.

## Decision implication

Do not extend the coherence-margin route to 20M from the pilot score alone. If the pilot official cheap7 is poor or broadly destructive, close the route. If the pilot cheap7 looks encouraging, first run the minimal 4M charged lambda-zero same-row arm and compare:

1. official cheap7 and family movement for margin pilot versus lambda-zero, not just versus ordinary chck4M;
2. coherent-vs-disrupted NLL gap for margin pilot versus lambda-zero on the fixed train64/holdout64 probes;
3. whether any score movement is relation/state-relevant rather than COMPS/BLiMP churn or a few GlobalPIQA flips.

Only a positive result on both official movement and mechanism separation would justify a 20M charged paired screen.

## Training-trace and comparator assets added

`data/cohmargin_training_trace_summary/cohmargin_training_trace_summary.{json,md}` summarizes the actual reopen structured context margin route pilot trace. Across 51 optimizer steps and 3,999,724 charged words / 1,999,862 coherent words, the margin loss stayed near `softplus(0.20) ≈ 0.798` (mean 0.799840, last10 0.798956), and the logged `nll_bad_minus_coh_last_micro` averaged -0.000627 with last10 -0.000315. The internal training signal therefore did not visibly separate coherent from disrupted contexts during the 4M-charged pilot.

I also built `scripts/generic_item_family_compare.py` and validated it on the already-complete frozen anchor fastpath disruption design coherent86-vs-chck82 contrast at `data/generic_compare_validation_coherent_vs_chck82/`. It reproduces cheap7 delta +0.1469786949768448 and exposes the known item movement (BLiMP +30 net, Supplement -14 despite score +0.712, EWoK -2, Entity +7, COMPS -137, GlobalPIQA +1), confirming it is suitable for future pilot-vs-lambda0 or pilot-vs-reference payloads once complete.
