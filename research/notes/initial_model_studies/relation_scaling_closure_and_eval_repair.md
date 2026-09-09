# fineweb relation vs random 3m direct checkpoint trajectory — Repaired direct-checkpoint 3M trajectory: corrected interpretation

## Evaluation artifact confirmed and fixed

fineweb random quality 3m profile evaluated local parent `hf_model` with `--revision_name chck_1M/2M/3M`. For local
directories, transformers ignores `revision`, so fineweb random quality 3m profile scored the final (3M) weights for every
"checkpoint" on tasks loaded through that pattern — invalidating its 1M/2M points. fineweb relation vs random 3m direct checkpoint trajectory repairs
this by passing each exact checkpoint directory as `--model_path_or_name` with no revision.

Confirmation via model.safetensors hashes (all distinct per checkpoint, matching earlier analysis subdir hashes):
- random: chck_1M b1278fb5, chck_2M 739e7f7b, chck_3M 9c5879d4
- relation: chck_1M 2b58cb5f, chck_2M fd85e7ae, chck_3M e3714f87

## Corrected relation_explicit − random_quality trajectory (real, per-checkpoint weights)

| checkpoint | ΔBLiMP | ΔSupplement | ΔEWoK | ΔEntity | ΔCOMPS | ΔReading |
|---|---:|---:|---:|---:|---:|---:|
| chck_1M | +0.63 | +1.20 | **+3.63** | +0.12 | -0.03 | -0.49 |
| chck_2M | -0.01 | -2.80 | **-2.09** | -0.05 | -0.23 | +0.09 |
| chck_3M | -0.90 | +1.60 | **+0.54** | +0.30 | +0.31 | +0.09 |

## Scientific reading

1. **The relation-explicit EWoK gain is real at 1M (+3.63, close to fineweb relation vs random 1m profile's +3.27) but does NOT
   persist and is highly volatile.** It swings to −2.09 at 2M and settles at only +0.54 at 3M. This is
   not a clean monotone decay; the two arms cross. On the 100-item-per-domain fast EWoK set, single-seed
   checkpoint-to-checkpoint noise is large (EWoK moves ~5.7 points between the arms across 1M→2M).

2. **Relation-explicit does not move Entity at any exposure** (+0.12/−0.05/+0.30 ≈ 0). Confirmed again
   across a genuine 1M/2M/3M sweep.

3. **Conclusion: relation-explicit FineWeb selection is not a durable, scalable multi-column lever.**
   The 1M EWoK bump is an early-exposure effect that does not survive to 3M and carries no Entity
   benefit. Scaling relation-explicit data alone will not close the Overall SOTA gap.

4. **Measurement caution now load-bearing:** single-seed fast-EWoK deltas are too noisy for route
   decisions. Any future EWoK claim needs ≥2 seeds and, ideally, per-domain/item-level bootstrap. The
   fast set (11 domains × 100) is a screening tool, not a scoring surface.

## Reproducibility action taken / needed

- Fixed: fineweb relation vs random 3m direct checkpoint trajectory direct-checkpoint evaluator (`relation_3m_direct_checkpoint_eval.py`) is the
  correct pattern for all future local multi-checkpoint evaluation. Do not pass parent `hf_model` +
  `--revision_name` for local checkpoints.
- Needed audit: any prior multi-checkpoint local trajectory or endpoint-selection result (e.g. the
  "chck_80M best endpoint" current-best coordinate) that used parent+revision loading must be
  re-checked. Single-checkpoint runs (fineweb relation vs random 1m profile, pair discriminator tokenaware 1m profile) are unaffected because parent == final == the
  only checkpoint.

## Route consequence

Two data-composition levers are now closed as durable SOTA paths: paired-restatement (pair discriminator tokenaware 1m profile) and
relation-explicit scaling (fineweb relation vs random 3m direct checkpoint trajectory). Data-selection filters alone do not close the Entity/EWoK gap.
The research must move to a mechanism that changes cross-sentence state formation or credit assignment
to the MLM logits — and it must be genuinely distinct from the already-negative RMEC (relation-word
priority masking) and MCEC (downstream-content masking). Any such mechanism must
carry a pre-registered mechanism probe on non-evaluation text showing earlier context causally controls
later predictions, plus ≥2 seeds for any fast-EWoK claim.
