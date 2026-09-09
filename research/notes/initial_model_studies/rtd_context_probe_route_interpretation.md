# event binding timescale — Matched RTD+MLM vs WWM context probe: route interpretation

## Evidence files

- Script: `scripts/rtd_wwm_context_probe_tokenlevel.py`
- JSON: `data/rtd_wwm_context_probe_tokenlevel.json`
- Rows: `data/context_probe_tokenlevel_rows.csv`
- Note: `notes/rtd_wwm_context_probe_tokenlevel.md`

## Matched 1M setup being interpreted

Both models are matched on:

- official 1M corpus slice: BNC spoken 762,080 words + CHILDES 237,920 words
- DeBERTa-v2 8×480, n_head=8, baseline16k tokenizer
- WWM p=0.15, batch 128, 49 steps, seeds 42/456/789
- identical tokenization and masking counts (`masked_tokens_total = 199,522`)

RTD shortcut metrics at 1M:

- replaced recall: 0.2554
- above-majority RTD accuracy: +0.0186
- predicted-original rate: 0.9413
- label-original rate: 0.8486

This means RTD partially escapes the all-original shortcut but remains weak and highly original-biased.

## Fast-profile outcome at chck_1M

From completed counterfactual binding credit gate direct-checkpoint evaluation:

| metric | RTD − WWM |
|---|---:|
| BLiMP fast | +0.17 |
| Supplement fast | −2.80 |
| EWoK fast | +2.18 |
| Entity fast | −1.39 |
| COMPS | +0.13 |
| Reading mean | +0.045 |

This is a one-column EWoK gain with Entity and Supplement damage. It cannot justify scaling unless the mechanism probe shows that RTD changed cross-sentence credit assignment beyond WWM.

## Context-intervention design

The repaired token-level probe reconstructs exact prior data event binding panel consumed examples, splits each example into an earlier prefix and fixed later suffix, chooses a fixed later tokenizer token as the MLM target, and applies identical variants to both WWM and RTD+MLM:

- `full`: true prefix + later suffix with fixed target masked
- `deleted`: no prefix + same later suffix and target
- `shuffled`: block-shuffled true prefix + same later suffix and target
- `unrelated`: unrelated prefix of similar length + same later suffix and target

The route-relevant quantities are **RTD-minus-WWM increments** and **specificity relative to unrelated prefix**. Single-model sensitivity is not enough because WWM already contains recoverable history signal.

## Results (240 fixed-token cases)

| quantity | mean | median | positive fraction |
|---|---:|---:|---:|
| WWM full−deleted | +0.00002 | +0.00000 | 0.525 |
| RTD full−deleted | +0.00183 | +0.00132 | 0.546 |
| RTD−WWM full−deleted | +0.00181 | +0.00142 | 0.542 |
| RTD−WWM full−shuffled | −0.00001 | +0.00005 | 0.542 |
| RTD−WWM full−unrelated | +0.00286 | +0.00089 | 0.542 |
| RTD−WWM specificity deleted vs unrelated | −0.00106 | +0.00091 | 0.533 |
| RTD−WWM specificity shuffled vs unrelated | −0.00287 | −0.00097 | 0.475 |

## Scientific interpretation

The necessary evidence for RTD as a cross-sentence credit-assignment improvement is absent.

1. **WWM is essentially prefix-insensitive in this token-level probe** at 1M. Its full−deleted/shuffled/unrelated deltas are near 0. This means the probe does not show WWM using these prefixes for the fixed later tokens.

2. **RTD+MLM has slightly larger prefix sensitivity than WWM for deletion**, but the effect is tiny (mean +0.0018 log-prob) and not specific to the true prefix.

3. **The unrelated-prefix effect is larger than the deletion effect.** RTD−WWM full−unrelated mean is +0.00286, while RTD−WWM full−deleted mean is +0.00181. Therefore the apparent RTD sensitivity is not evidence that true earlier context supports later prediction; it is more consistent with generic prefix/position/distribution sensitivity.

4. **Specificity relative to unrelated prefix is slightly negative**, especially for shuffled vs unrelated (mean −0.00287, median −0.00097, positive fraction 0.475). This directly fails the experimental criterion that RTD change earlier-context credit assignment relative to WWM and relative to unrelated-prefix perturbations.

## Route consequence

Do **not** scale the current RTD+MLM configuration to 100M. Do not launch seed43 or 3M/10M merely from the 1M EWoK gain. The current RTD result is best interpreted as a weak generic signal-density / local plausibility effect:

- minimal shortcut escape: replaced recall 0.255, above-majority +0.019;
- EWoK fast +2.18 at 1M;
- Entity −1.39 and Supplement −2.80;
- no matched context-specific credit-assignment gain over WWM.

This does not close all RTD-family ideas, but it closes the current simple hybrid RTD+MLM configuration as a scale-ready SOTA route.

## Next scientific need

The next mechanism must change **which training events force history to control later logits**, not only increase token-level signal density. It should be designed with a pre-registered matched probe from the start:

- same interventions applied to WWM and the candidate model;
- fixed later MLM target;
- unrelated-prefix control;
- route survives only if candidate − WWM specificity is positive and task-profile damage is absent.

Possible repair directions should address the failure mode directly: RTD learned generic anomaly/prefix sensitivity, not true-prefix specificity. Candidate mechanisms need hard negatives whose wrongness depends on earlier context, not generator replacements that are mostly local plausibility anomalies.
