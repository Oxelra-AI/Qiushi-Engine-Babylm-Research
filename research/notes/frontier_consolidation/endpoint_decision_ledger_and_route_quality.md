# endpoint decision ledger and route quality — endpoint decision ledger and route-quality correction

## Analysis Scope

No result was inferred from the incomplete retraining comparisons:

- Original same-pool-tokenizer compact-view reinvest retraining.
- Repaired byte-alphabet-tokenizer compact-view reinvest retraining.

It also did not inspect active training run directories, run GPU work, evaluate a model, mine additional evaluation subsets, alter tokenizers, or prepare final deliverables.



With endpoint evaluation already prepared, the remaining analysis made the decision criteria for the pending legal endpoints explicit, guarding against post-hoc endpoint preference, subset speculation, or unplanned 100M training.

## New artifact

Script:

- `experiments/archive/frontier_consolidation/scripts/endpoint_decision_ledger.py`

Outputs:

- `experiments/archive/frontier_consolidation/data/endpoint_decision_ledger/endpoint_decision_ledger.json`
- `research/documents/frontier_consolidation/data/endpoint_decision_ledger/endpoint_decision_ledger.md`

The script reads only completed summaries, tokenizer artifacts, and post-delivery dry-run manifests. It does not inspect active run directories.

Run result:

```json
{
  "status": "ENDPOINT_DECISION_LEDGER",
  "endpoints": [
    "complianttok_reinvest_seed43022",
    "bytealphatok_reinvest_seed43022"
  ],
  "old_reference_overall": 42.0331347900748,
  "two_seed_ate_overall": 0.582983
}
```

## Fixed evidence preserved

- Visible leaderboard reference used for immediate endpoint decisions: `41.8` Overall.
- Non-submittable old-tokenizer compact-view-reinvest seed43022 reference: `42.0331347900748` Overall, margin `+0.2331347900748` over 41.8.
- The old endpoint remains scientific evidence but not a legal submission endpoint because its tokenizer was trained on 100M Strict text.
- Two old-tokenizer seeds both showed positive within-seed treatment effect from compact-view reinvestment:
  - seed43022: `+0.685502` Overall.
  - seed43122: `+0.480465` Overall.
  - two-seed average: `+0.582983` Overall.

## Pending legal endpoints and commands

### `bytealphatok_reinvest_seed43022`

Resource-priority legal endpoint using the standard full ByteLevel alphabet tokenizer.

Evaluation command after byte-alphabet reinvest training completes:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/compliant_postdelivery_driver.py --arm reinvest --target bytealphatok_reinvest_seed43022 --run-dir experiments/archive/frontier_consolidation/training/runs/bytealphatok_reinvest_seed43022 --expected-tokenizer experiments/archive/frontier_consolidation/data/compliant_tokenizer_bytealphabet --expected-tokenizer-sha256 b2b317e655a96f2c14eb559ce9fa7573f180904d84819cacfb5f61174cf355cf --gpu 1
```

Expected isolated pristine collation:

- `experiments/archive/frontier_consolidation/data/compliant_pristine_collate/bytealphatok_reinvest_seed43022/pristine_collate_bytealphatok_reinvest_seed43022_summary.json`

### `complianttok_reinvest_seed43022`

Legal spatial repair route status same-pool tokenizer endpoint with known newline `<unk>` exposure; evaluate only if terminal and doing so does not delay the byte-alphabet priority path.

Evaluation command after the original compliant-tokenizer reinvest training completes:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/compliant_postdelivery_driver.py --arm reinvest --target complianttok_reinvest_seed43022 --run-dir experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2 --expected-tokenizer experiments/archive/frontier_consolidation/data/compliant_tokenizer --expected-tokenizer-sha256 91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9 --gpu 0
```

Expected isolated pristine collation:

- `experiments/archive/frontier_consolidation/data/compliant_pristine_collate/complianttok_reinvest_seed43022/pristine_collate_complianttok_reinvest_seed43022_summary.json`

## Expensive work discipline preserved

Once a terminal checkpoint exists, the next expensive official-compatible evaluation is justified because it decides:

1. whether a legal 10M-tokenizer compact-view-reinvest endpoint exceeds the visible 41.8 Strict-Small Overall;
2. whether the old-tokenizer 42.0331 scientific result survives under an end-to-end compliant tokenizer coordinate;
3. whether the byte-alphabet construction improves, preserves, or hurts the legal endpoint relative to the spatial repair route status same-pool tokenizer.

Lowest-cost reliable sequence:

1. collect the authoritative terminal task result only after runtime delivery;
2. inspect delivered run metadata for 100M exposure, 10-pass corpus use, full checkpoint ladder, frozen recipe, `hf_model/chck_100M`, and exact tokenizer SHA;
3. run zero-shot and Reading columns first;
4. run the already-built Supplement slice only as posthoc interpretation of produced predictions;
5. apply hard upper-bound continuation arithmetic before SuperGLUE and AoA;
6. run SuperGLUE, official min_context=0 AoA, and pristine collation only when mathematically warranted.

Not allowed as shortcuts:

- partial columns cannot stand in for the complete official-coordinate Overall;
- old-tokenizer dynamics cannot predict the corrected-tokenizer endpoint;
- newline `<unk>` analysis cannot determine the sign of the spatial repair route status endpoint before official predictions;
- tokenizer design is frozen; no vocabulary changes may be derived from benchmark strings.

## Route-quality consequence

The decisive evidence remains the completed legal 100M reinvest endpoints. If `bytealphatok_reinvest_seed43022` has a verified official-coordinate Overall above 41.8, compliance, scoring coordinate, contamination safeguards and reproducibility require independent review. If neither legal endpoint exceeds 41.8 after full official-compatible scoring, the exact score vector should guide a new discriminating comparison. Renaming the same tokenizer or spatial-lengthening variants does not create a new route.
