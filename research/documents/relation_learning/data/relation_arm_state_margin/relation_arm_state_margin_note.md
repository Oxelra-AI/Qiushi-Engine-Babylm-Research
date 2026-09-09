# calibrated route options relation-arm state-margin result

Created UTC: 2026-09-06T22:39:25Z

## Purpose

Connect the state-margin recency-reader characterization to the established causal locality result. If REPEAT shows stale-state retention and REPEAT_SPLIT collapses toward CLEAN, the state-margin probe is a fourth independent readout.

## Pre-scored predictions

- REPEAT T(orig_new - source) < CLEAN T(orig_new - source) for UPDATED_USE packets: stale attraction
- REPEAT_SPLIT T(orig_new - source) ≈ CLEAN T(orig_new - source): locality removed
- VIEW T(orig_new - source) > REPEAT T(orig_new - source): better changed-form handling
- VIEW_SPLIT ≈ CLEAN: locality removed
- On DISTRACTOR, REPEAT should show stronger source-state retention (T/U retention margins)

Decision rule: If predictions 1-2 hold, the state-margin probe confirms causal locality on state behavior; this joins the paper. If 1 fails, the state-margin probe is insensitive to the relation arm axis.

## Key result: T(orig_new - source) for UPDATED_USE packets

This is the primary state-update-use margin. More positive = model prefers new state after update. Less positive / negative = stale attraction.

| comparison | probe_set | n | full Δ(arm-clean) | SE | content Δ | SE |
|---|---|---:|---:|---:|---:|---:|
| repeat_minus_clean | balanced_heldout | 200 | +0.1391 | +0.0850 | +0.1240 | +0.0969 |
| repeat_minus_clean | extended_nontrain | 1125 | +0.1887 | +0.0365 | +0.1884 | +0.0403 |
| repeat_split_minus_clean | balanced_heldout | 200 | +0.2395 | +0.0964 | +0.2782 | +0.1056 |
| repeat_split_minus_clean | extended_nontrain | 1125 | +0.2564 | +0.0374 | +0.3081 | +0.0410 |
| view_minus_clean | balanced_heldout | 200 | +0.2485 | +0.0997 | +0.2513 | +0.1076 |
| view_minus_clean | extended_nontrain | 1125 | +0.3644 | +0.0399 | +0.3694 | +0.0432 |
| view_split_minus_clean | balanced_heldout | 200 | +0.1747 | +0.0850 | +0.1893 | +0.0965 |
| view_split_minus_clean | extended_nontrain | 1125 | +0.2499 | +0.0367 | +0.2921 | +0.0402 |

## Key result: source retention for UNCHANGED_DISTRACTOR_USE packets

| comparison | condition | probe_set | n | full Δ(source-new) | SE | content Δ | SE |
|---|---|---|---:|---:|---:|---:|---:|
| repeat_minus_clean | T | balanced_heldout | 200 | -0.0011 | +0.0602 | +0.0084 | +0.0699 |
| repeat_minus_clean | T | extended_nontrain | 200 | -0.0011 | +0.0602 | +0.0084 | +0.0699 |
| repeat_minus_clean | U | balanced_heldout | 200 | +0.1366 | +0.0624 | +0.1831 | +0.0748 |
| repeat_minus_clean | U | extended_nontrain | 200 | +0.1366 | +0.0624 | +0.1831 | +0.0748 |
| repeat_split_minus_clean | T | balanced_heldout | 200 | +0.0020 | +0.0790 | +0.0237 | +0.0898 |
| repeat_split_minus_clean | T | extended_nontrain | 200 | +0.0020 | +0.0790 | +0.0237 | +0.0898 |
| repeat_split_minus_clean | U | balanced_heldout | 200 | -0.1746 | +0.0748 | -0.2030 | +0.0915 |
| repeat_split_minus_clean | U | extended_nontrain | 200 | -0.1746 | +0.0748 | -0.2030 | +0.0915 |
| view_minus_clean | T | balanced_heldout | 200 | -0.0359 | +0.0711 | +0.0379 | +0.0842 |
| view_minus_clean | T | extended_nontrain | 200 | -0.0359 | +0.0711 | +0.0379 | +0.0842 |
| view_minus_clean | U | balanced_heldout | 200 | +0.1927 | +0.0639 | +0.2765 | +0.0773 |
| view_minus_clean | U | extended_nontrain | 200 | +0.1927 | +0.0639 | +0.2765 | +0.0773 |
| view_split_minus_clean | T | balanced_heldout | 200 | -0.0757 | +0.0590 | -0.0400 | +0.0731 |
| view_split_minus_clean | T | extended_nontrain | 200 | -0.0757 | +0.0590 | -0.0400 | +0.0731 |
| view_split_minus_clean | U | balanced_heldout | 200 | -0.1584 | +0.0668 | -0.1453 | +0.0794 |
| view_split_minus_clean | U | extended_nontrain | 200 | -0.1584 | +0.0668 | -0.1453 | +0.0794 |

## Interpretation

Read the delta tables against the predictions. If locality predictions hold, this readout joins the paper alongside compact probes, Wikipedia probes, and Entity official evaluation as the fourth independent confirmation of relation-typed source-conditioned readout with causal locality dependence.
