# static decoy contingency validation — static-decoy contingency validation

Purpose: repair and minimally validate the static-decoy version of the frozen-82M private-tail trainer, to keep a cleaner source-correspondence control available if the current batch-deranged 4M tail aligned/shuffled score difference is too close to interpret. This is a CPU-only safety check, not a new GPU route.

Script repaired: `experiments/archive/frontier_consolidation/scripts/frozen82_private_tail_trainer_static_decoy.py`.

Validation commands:

- `CUDA_VISIBLE_DEVICES='' python -B .../frozen82_private_tail_trainer_static_decoy.py --output_dir experiments/archive/frontier_consolidation/data/static_decoy_smoke/shuffled_pairactive --mode shuffled --max_updates 2 --max_tail_charged_words 120000 --batch_size 256 --log_every 1`
- `CUDA_VISIBLE_DEVICES='' python -B .../frozen82_private_tail_trainer_static_decoy.py --output_dir experiments/archive/frontier_consolidation/data/static_decoy_smoke/aligned_pairactive --mode aligned --max_updates 2 --max_tail_charged_words 120000 --batch_size 256 --log_every 1`

AST check: passed after repairing line 246 indentation.

Paired two-update smoke result:

| mode | updates | main words | aux words | total consumed | aux targets | aux units/views | mean aux NLL | mean neutral loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aligned | 2 | 78,873 | 796 | 82,092,164 | 128 | 18 conditioned + 18 free | 4.492479991912842 | 1.1446193298736418e-10 |
| static-decoy shuffled | 2 | 78,873 | 796 | 82,092,164 | 128 | 18 conditioned + 18 free | 6.196540768941244 | 1.1446193298736418e-10 |

Interpretation: for the same WWM masks and same pair-active batches, the static-decoy aligned/shuffled arms charge exactly the same main words and auxiliary words and expose the same source-free/conditioned view counts and target counts. The only intended difference is the conditioned source IDs. This makes it a cleaner fallback control than the already-running batch-deranged tail if the measured 4M score/probe differences are too close relative to the observed 55-word auxiliary-charge mismatch in frozen82 short tail plan. No GPU run is justified from this validation alone; use it only if the pending mature-tail evidence requires a cleaner correspondence control.

Smoke outputs:
- `data/static_decoy_smoke/aligned_pairactive/scientific_metrics.json`
- `data/static_decoy_smoke/shuffled_pairactive/scientific_metrics.json`

## Update-path sanity check

A follow-up three-update aligned CPU smoke at `data/static_decoy_smoke/aligned_pairactive_3upd/` resolved the apparent zero-RMS issue from the two-update smoke. The training schedule has `lr=0` on update 1; update 2 logs `private_rms_max=0.0` because the forward used still-zero `up` weights before the first nonzero-LR optimizer step changes them. On update 3, `private_rms_max=0.0002826369018293917` and deterministic neutrality becomes `2.6624e-06`, confirming that private adapter parameters do update in the static-decoy script. This is consistent with the real 100-update frozen82 short tail plan tail runs, where private RMS reaches ~0.05.

## Inference-time private-scale handle

Prepared `scripts/materialize_private_scale.py` for future reversible admission tests of the learned private branch. CPU smoke materialized aligned-tail `private_adapter_scale=0.0` at `data/private_scale_smoke/aligned_scale0/` and compared it against the verified `chck_82M` reference on a deterministic text. Result: max and mean logit difference are exactly `0.0` despite different model classes (`AdapterDebertaV2ForMaskedLM` vs `FrozenSlowPrivateDebertaV2ForMaskedLM`). This proves that scale 0 is a safe no-private carrier and that intermediate private-scale endpoints can be created without retraining if the pending source-free probe shows useful private information but full scale erodes part of the score surface. No such scale evaluation is justified until the shuffled 4M tail and source-free probe results are read.
