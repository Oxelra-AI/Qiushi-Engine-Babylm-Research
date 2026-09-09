# scientific account three levels v5 candidate provenance package

Created: `2026-09-08T04:54:56Z`

**Export rule:** Export seed62064 as it is the higher complete endpoint; report seed62065 as fixed-policy replication; do not average into a new endpoint

## Validation checks

- `model_weight_seed64_integrity`: ✓
- `model_weight_seed65_integrity`: ✓
- `automodel_registration_valid`: ✓
- `source_configs_only_mlm`: ✓
- `tokenizer_identical_across_seeds`: ✓
- `repaired_configs_identical`: ✓
- `repaired_modeling_identical`: ✓
- `exposure_within_budget`: ✓
- `seed64_aoa_complete`: ✓
- `seed65_aoa_complete`: ✓
- `ms_aoa_complete`: ✓

**All passed:** ✓

## BabyLM Strict-Small exposure compliance

- Budget: 100,000,000 words
- Parent (trunk) exposure: 86,005,295
- Acquisition prefix words: 3,162,742
- Preservation additional counted: 517,332
- **Endpoint exposure (conservative):** 89,685,369 (89.685369M)
- Within budget: ✓

## Model weight integrity

- seed62064 `model.safetensors`: `1c6268a37a7774a78d725307f99bbaf1f5225b4315f9bfd9ddb2a5b47c680e4a`
  - Source = repaired: ✓
- seed62065 `model.safetensors`: `b814dd54340a9290c0ad92118adefa456f5ff6bad03d4aef641865bca609c8d8`
  - Source = repaired: ✓
- exact (M,S): `c14ac2c82470748aa6513585c12d0aac893dc96b7f87796e72c6c0d948a87ff4`
- coherent86: `e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8`

## Training method

- Method: `ms_acquisition_plus_ordinary_fullrow_parent_distillation_clean_rng`
- Updates: 80
- Parent: `models/frontier`
- λ_pres: 1.0, pres_student_mode: `eval`
- KL direction: `KL(teacher || student)`
- Preservation rendering: `ordinary-corruption parent distillation, not evidence-absent preservation`
- Focus lambda: 0.15
- Private adapter scale: 0.75 (8 layers)
- seed62064 train_seed: 62064, acq_seed: 135392207
- seed62065 train_seed: 62065, acq_seed: 974058884

## AutoModel registration

All repaired configs expose both `AutoModelForMaskedLM` and `AutoModel`: ✓
Source configs expose only `AutoModelForMaskedLM`: ✓

## AoA provenance

- `seed62064`: score=0.0, steps=18, rows=144090, estimator=`current_official_main_6f825c2_AoAEvaluator`, legitimate_zero=True
- `seed62065`: score=0.0, steps=18, rows=144090, estimator=`current_official_main_6f825c2_AoAEvaluator`, legitimate_zero=True
- `ms_acquisition`: score=0.0, steps=18, rows=144090, estimator=`current_official_main_6f825c2_AoAEvaluator`, legitimate_zero=True

Participant commit: `6f825c291e2c4c78ad33b1935fd64d45f52642dc`

Full JSON: `experiments/archive/functional_learning/data/v5_candidate_provenance/v5_candidate_provenance.json`
