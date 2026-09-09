# endpoint decision ledger and route quality endpoint decision ledger

This file records how the pending legal endpoints should be interpreted once the runtime delivers a terminal retrain result. It reads completed artifacts only and does not inspect active training run directories or managed-task state.

## Fixed references

- Visible leaderboard target used here: **41.800** Overall.
- Old-tokenizer compact-view-reinvest seed43022: **42.033134790075** Overall, margin **0.233135** over 41.8, but not submittable because its tokenizer was trained on 100M Strict text.
- For the old endpoint's cheap seven-column surface, SuperGLUE+AoA needed to match 41.8 would be **68.937836**; actual SuperGLUE+AoA was **71.036050**.
- Two old-tokenizer seeds both improved under compact-view reinvestment: treatment effects **0.685502** and **0.480465** Overall; the two-seed average is **0.582983**.

## Pending legal endpoints

### `complianttok_reinvest_seed43022`

- Model status: legal same-pool tokenizer endpoint with known newline unknown-token exposure.
- Tokenizer: `experiments/archive/frontier_consolidation/data/compliant_tokenizer`; SHA match: `True`.
- Run directory for completed-result inspection: `experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2`.
- Dry-run stages: inspect, cheap_columns, supplement_prediction_slices, continuation_projection, superglue_aoa, pristine_collate.
- Isolated pristine collation summary: `experiments/archive/frontier_consolidation/data/compliant_pristine_collate/complianttok_reinvest_seed43022/pristine_collate_complianttok_reinvest_seed43022_summary.json`.
- Evaluation command after training completion:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/compliant_postdelivery_driver.py --arm reinvest --target complianttok_reinvest_seed43022 --run-dir experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2 --expected-tokenizer experiments/archive/frontier_consolidation/data/compliant_tokenizer --expected-tokenizer-sha256 91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9 --gpu 0
```

### `bytealphatok_reinvest_seed43022`

- Model status: legal endpoint using standard full ByteLevel alphabet.
- Tokenizer: `experiments/archive/frontier_consolidation/data/compliant_tokenizer_bytealphabet`; SHA match: `True`.
- Run directory for completed-result inspection: `experiments/archive/frontier_consolidation/training/runs/bytealphatok_reinvest_seed43022`.
- Dry-run stages: inspect, cheap_columns, supplement_prediction_slices, continuation_projection, superglue_aoa, pristine_collate.
- Isolated pristine collation summary: `experiments/archive/frontier_consolidation/data/compliant_pristine_collate/bytealphatok_reinvest_seed43022/pristine_collate_bytealphatok_reinvest_seed43022_summary.json`.
- Evaluation command after training completion:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/compliant_postdelivery_driver.py --arm reinvest --target bytealphatok_reinvest_seed43022 --run-dir experiments/archive/frontier_consolidation/training/runs/bytealphatok_reinvest_seed43022 --expected-tokenizer experiments/archive/frontier_consolidation/data/compliant_tokenizer_bytealphabet --expected-tokenizer-sha256 b2b317e655a96f2c14eb559ce9fa7573f180904d84819cacfb5f61174cf355cf --gpu 1
```

## Why the next GPU evaluation is justified only after terminal training

The next expensive evaluation resolves:
- whether a legal 10M-tokenizer compact-view-reinvest endpoint exceeds the visible 41.8 Strict-Small Overall score
- whether the old-tokenizer 42.0331 scientific result survives under an end-to-end compliant tokenizer coordinate
- whether the byte-alphabet construction improves, preserves, or hurts the legal endpoint relative to the spatial repair route status same-pool tokenizer

Use the lowest-cost reliable sequence:
- collect authoritative terminal task result only after runtime delivery
- inspect delivered run metadata for 100M exposure, 10-pass corpus use, full checkpoint ladder, frozen recipe, `hf_model/chck_100M`, and exact tokenizer SHA
- run zero-shot and Reading columns first, including GlobalPIQA parallel/nonparallel combined as the official GlobalPIQA column
- run the posthoc Supplement slice join only to interpret the already-produced Supplement predictions
- apply the existing hard upper-bound continuation arithmetic before SuperGLUE and AoA
- run SuperGLUE, official min_context=0 AoA, and pristine collation only when the endpoint remains mathematically viable for the target being tested

Do not use these shortcuts:
- partial columns cannot stand in for the complete official-coordinate Overall
- old-tokenizer dynamics cannot predict the corrected-tokenizer endpoint
- newline unknown-token analysis cannot determine the sign of the spatial repair route status endpoint before official predictions
- tokenizer design is frozen; no vocabulary changes may be derived from benchmark strings

## Result-dependent route

- The byte-alphabet reinvest endpoint requires verified frozen-recipe training and pristine nine-column collation. If Overall exceeds 41.8, review compliance, evidence, evaluation coordinate, contamination safeguards and reproducibility; otherwise compare the complete column movements with the old 42.0331 reference and the other completed endpoint.
- The compliant-tokenizer reinvest arm remains a legal endpoint comparison. Use its isolated target name and pristine collation path; interpret the Supplement slice only after predictions are available to distinguish newline effects from broader learning.
- A resource/OOM failure before complete chck_100M production is not evidence about the learning principle. A retry must preserve corpus, tokenizer, seeds, batch, sequence length, optimizer, WWM and exposure.
- If no legal endpoint exceeds the visible leader under full official-coordinate scoring, compare relation stability, GlobalPIQA weakness, optimizer dynamics and data/representation hypotheses using cheaper discriminating tests before any new 100M run. Repeating the same tokenizer or spatial-lengthening variant under a new name is not a new scientific comparison.

## Interpretation assets

The dual compliant tokenizer endpoint policy and supplement newline symmetry and posthoc slices files explain possible score movement only after official predictions exist. They cannot choose the endpoint before scoring. The spatial repair route status tokenizer unknown-token exposure is concentrated in Supplement QA/turn-taking newlines and is structurally symmetric in good and bad alternatives, while old-tokenizer reinvestment's Supplement gain over clean came mainly from unaffected rows.

Full JSON: `experiments/archive/frontier_consolidation/data/endpoint_decision_ledger/endpoint_decision_ledger.json`
