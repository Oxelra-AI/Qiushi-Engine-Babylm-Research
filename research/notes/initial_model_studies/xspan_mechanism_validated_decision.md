# relational xspan compact v4 filter — Relational XSpan mechanism validated; proceed to primary-objective trainer

## Decision
The compact v4 rule-based XSpan targets show a genuine, broad, s1-specific primary-likelihood signal under the protected 100M WWM model. This clears the stated likelihood criterion (one compact repair → true/wrong/no-s1 likelihood test). **Stop stacking span-boundary rules. Build the XSpan primary-objective trainer.**

## Evidence (`data/xspan_compact_v4_s1_likelihood_probe.json`)
- Overall true−wrong mean **+0.717**, median +0.491, positive fraction **76.4%** (n=212 heldout targets).
- true−no mean +0.944; **wrong−no mean only +0.227** (median +0.026). The true-s1 advantage is specific to the correct previous sentence, not "any preceding sentence."
- Broad across target types: action/object/result +1.309 (84.6% pos), semantic continuation +0.843 (76.1%), location +0.587 (71.4%), definition/property +0.533 (80.4%).
- **Not template-concentrated:** top-decile prefixes are diverse (`is in`, `distributed by`, `black beak`, `shares in`, `awarded since`, `three movements`, `live single`, `released on`...). One legitimate `is in southwestern France` geographic cluster exists but is small and semantically valid.
- Negatives are interpretable: they occur when s1 is generic (`The population was 1,725...`) or s2 is a common collocation (`in Great Britain`), i.e. where s1 truly adds little. This is the shape of a real signal.

## Contrast with closed routes
- Pronoun/deictic token targets (counterfactual propagation closed): true−wrong ≈ **+0.0004** (no specificity). Rejected.
- Counterfactual perturbation-detection (cfprop readout 200k comparison): signal not localized to dependent token; local-anomaly shortcut. Closed.
- XSpan semantic spans (relational xspan compact v4 filter): true−wrong **+0.717**, broad, specific. **Proceed.**

## Why this is a primary-objective route, not another auxiliary
The XSpan objective replaces a controlled fraction of ordinary MLM masks with these s2 semantic spans, so reducing the loss requires the model to use s1 through the main MLM head — measurable by the same true/wrong/no-s1 likelihood drop. There is no separate discriminative head that could be minimized by local anomaly.

## Proposed trainer design
Extend the trusted WWM full-cycle trainer:
1. Keep ordinary WWM as the base objective on the official corpus, unchanged, with exact word-exposure accounting.
2. On a separate counted XSpan stream (compact v4 JSONL), mask the designated s2 target span and train the **primary MLM head** to predict it, conditioned on true s1.
3. Combined loss L = (1−ρ)·L_WWM + ρ·L_XSpan, ρ small (e.g. 0.1–0.25). No train-only head.
4. Count all XSpan true-context words toward exposure (`words` field). Keep wrong-s1/no-s1 only for evaluation probes, not training exposure.
5. Preserve standard HF checkpoints so the official evaluator loads without fine-tuning.

## Required matched controls (2×2)
| data | objective |
|---|---|
| official uniform | WWM |
| official uniform | WWM + XSpan(true-s1) |
| official uniform | WWM + XSpan(wrong-s1 control) |
| official + relational mix | WWM + XSpan(true-s1) |

Plus: same seed/geometry/exposure; report at each checkpoint the true/wrong/no-s1 likelihood drop on heldout XSpan, and direct-checkpoint BabyLM task scores. Gate to 10M/20M only if Entity + (EWoK or GlobalPIQA) move while BLiMP/Supplement/Reading are guarded; require multi-seed before any 100M commitment.

## Data artifact
- Trainable XSpan JSONL: `data/xspan_revision_118/relational_xspan_compact_v4_from_v3_seed117_target200000_actual.jsonl` (4,235 rows, 101,777 true-context words, 0 mechanical flags).
- Fields: `text`/`target_span_text` (true-s1), `text_wrong_s1`/`target_span_wrong_s1`, `text_no_s1`/`target_span_no_s1`, `target_type`, `split`.
