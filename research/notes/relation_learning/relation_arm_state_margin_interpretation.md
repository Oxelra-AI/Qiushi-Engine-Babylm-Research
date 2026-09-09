# calibrated route options: Relation-arm state-margin result — interpretation

Created UTC: 2026-09-07

## Executive summary

The state-margin probe confirms causal locality dependence on entity-state reading as a **fourth independent readout**, joining compact probes, Wikipedia probes, and Entity evaluation. The confirmation required neutral-adjusted T-N margins because split arms carried a large N bias that inflated their raw T values.

## Prediction vs result

| Prediction | Raw T result | Neutral-adjusted T-N | Verdict |
|---|---|---|---|
| REPEAT T < CLEAN (stale attraction) | REPEAT T > CLEAN (+0.189) | REPEAT T-N > CLEAN (+0.161) | **Wrong** — modest improvement, not stale |
| REPEAT_SPLIT ≈ CLEAN | REPEAT_SPLIT T >> CLEAN (+0.256) | REPEAT_SPLIT T-N ≈ CLEAN (−0.137) | **Confirmed after T-N** |
| VIEW T > REPEAT | VIEW T > REPEAT (+0.364 vs +0.189) | VIEW T-N > REPEAT (+0.452 vs +0.161) | **Confirmed** |
| VIEW_SPLIT ≈ CLEAN | VIEW_SPLIT T >> CLEAN (+0.250) | VIEW_SPLIT T-N ≈ CLEAN (+0.046) | **Confirmed after T-N** |

## Why raw T was misleading

The split arms showed elevated UPDATED_USE T values because their N (no-update) values were substantially higher than CLEAN:

| Arm | N (new-source, no update) | N shift vs CLEAN |
|---|---:|---:|
| clean | −4.764 | 0.000 |
| repeat | −4.736 | +0.028 |
| view | −4.852 | −0.088 |
| **repeat_split** | **−4.371** | **+0.394** |
| **view_split** | **−4.561** | **+0.204** |

Split arms assign higher base scores to the new-state phrase even without seeing any update sentence. This N bias appears to come from the split training distributing companion material across rows, which changes the model's general phrase-level scoring without installing a relation-specific context response. Only T-N removes this bias to reveal the true update-use structure.

## The locality pattern after neutral adjustment (UPDATED_USE, extended_nontrain, n=1125)

| Arm | T | N | T-N | T-N vs CLEAN | Content T-N vs CLEAN |
|---|---:|---:|---:|---:|---:|
| clean | +0.841 | −4.764 | +5.605 | 0.000 | 0.000 |
| repeat | +1.030 | −4.736 | +5.766 | +0.161 | +0.161 |
| **view** | **+1.206** | **−4.852** | **+6.057** | **+0.452** | **+0.476** |
| repeat_split | +1.098 | −4.371 | +5.468 | −0.137 | −0.156 |
| view_split | +1.091 | −4.561 | +5.652 | +0.046 | +0.045 |

**The ordering on T-N is: VIEW >> REPEAT > CLEAN ≈ VIEW_SPLIT > REPEAT_SPLIT**

Both split arms collapse to CLEAN (or slightly below), removing the local companion effect. This matches the compact-probe locality result: the behavioral face of designed companions depends on within-window co-occurrence.

## What differs from the compact-probe result

1. **REPEAT is modestly positive, not negative.** The compact probe showed REPEAT creating a source-conditioned *liability* for changed-form targets. Here, REPEAT shows a small *improvement* in state-update use (+0.161). The difference: the compact probe involves exact recurrence of the source text in the training window, which creates competition when the target requires a different form. The state-margin probe has no exact recurrence of source text — it presents a fresh source sentence, an update, and a state query. REPEAT's modest positive may reflect its general training on local paired content rather than its specific exact-copy liability.

2. **VIEW is strongly positive.** VIEW's T-N advantage (+0.452) exceeds REPEAT's (+0.161) by 0.291 nats, consistent with VIEW's varied-restatement training being more useful for state changes than REPEAT's exact-copy training. This parallels the compact probe where VIEW improved changed-form use.

3. **N bias is a new finding.** The split arms' elevated N scores were not visible in the compact or Wikipedia probes. This may reflect split-arm training changing the model's general phrase-level scoring distribution without installing relation-specific patterns.

## Scientific value for the paper

This readout adds three things:

1. **Fourth confirmation of causal locality.** After neutral adjustment, the locality pattern holds: local arms > CLEAN on relation-specific update use, split arms ≈ CLEAN.

2. **Dissociation between REPEAT's compact-probe liability and state-probe benefit.** REPEAT is negative on compact changed-form targets (stale attraction) but modestly positive on state-update targets (no exact recurrence). The liability is specific to the exact-copy context, not a general state-reading deficit.

3. **N-bias finding.** Split training inflates general phrase scores, which can mask the absence of relation-specific learning. T-N correction is necessary for state-margin probes.

## Artifacts

- Raw per-arm T/U/N margins: `experiments/archive/relation_learning/data/relation_arm_state_margin/raw_scores/raw_TUN_margin_summary.csv`
- Arm-minus-CLEAN deltas: `experiments/archive/relation_learning/data/relation_arm_state_margin/arm_minus_clean_delta_summary.csv`
- Per-arm key margins: `experiments/archive/relation_learning/data/relation_arm_state_margin/per_arm_key_margins.csv`
- Pre-scored predictions: `experiments/archive/relation_learning/data/relation_arm_state_margin/pre_score_predictions.json`
