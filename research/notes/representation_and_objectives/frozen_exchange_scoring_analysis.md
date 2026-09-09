# frozen exchange scoring analysis — Frozen exchange scoring: role-switch packets on legal40k and reheat endpoints

## Key finding: strong unsaturation confirmed

Both compliant endpoints fail massively on the role-switch packets. The model cannot consistently bind the same alternatives correctly under both context directions.

## Scoring design

For each of 520 pairs, scored s(context + consequence_with_alternative) for 3 contexts (AB, BA, erased) × 2 alternatives = 6 forward passes per pair per checkpoint. Four-cell margin M = margin_AB + margin_BA, where margin = s(correct) − s(wrong). M > 0 means context-conditioned binding; M = 0 is target-prior-only. Context-erasure |bias| measures the target prior without context. Both_correct = fraction where both directions have positive margins.

All target words tokenize as 2 tokens (space + word) with the legal40k byte-BPE. The space token (ID 225) is shared across all alternatives at the same position, so its log-probability cancels exactly in margin computation.

## Comparative results

| Metric | legal40k_100M | reheat_100M | Δ |
|--------|-------------|-------------|---|
| Overall M | +1.169 | +1.487 | +0.318 |
| M positive fraction | 0.594 | 0.612 | +0.018 |
| erased |bias| | 2.732 | 2.622 | −0.110 |
| acc_AB | 0.492 | 0.500 | +0.008 |
| acc_BA | 0.592 | 0.606 | +0.014 |
| **both_correct** | **0.281** | **0.308** | **+0.027** |

Chance both_correct (two independent binaries) = 0.25. Both models are only marginally above chance.

## By family

| Family | legal40k both | reheat both | legal40k M | reheat M |
|--------|---------------|-------------|------------|----------|
| spatial | 0.183 | 0.208 | +0.273 | +0.692 |
| temporal | 0.310 | 0.310 | −0.779 | −0.840 |
| transfer | 0.487 | 0.562 | +4.819 | +6.783 |
| comparative | 0.450 | 0.438 | +4.286 | +4.129 |
| state_change | 0.212 | 0.300 | +2.464 | +2.268 |
| **container (HO)** | **0.017** | **0.000** | **−4.545** | **−4.675** |

Container is the fully held-out family. Both models score near or at zero both_correct — no generalization to unseen relation types.

## By template style and entity split

| Split | legal40k both | reheat both |
|-------|---------------|-------------|
| style=train | 0.241 | 0.279 |
| style=held_out | 0.356 | 0.361 |
| entity=train | 0.281 | 0.313 |
| entity=held_out | 0.278 | 0.278 |

Entity transfer is flat: train and held-out entities perform identically. This is positive for training prospects — the model generalizes across entities but not across relation types.

Held-out templates (paraphrase transfer) perform BETTER than training templates. This suggests training templates are not harder; the difference is in which families have more held-out templates.

## What this establishes

1. **Strong unsaturation**: 72% of pairs fail both-correct. The model cannot bind alternatives under competing contexts. This is the exact deficit identified in EWoK and GlobalPIQA.

2. **Genuine context conditioning exists but is insufficient**: M > 0 overall (the model uses context), but both_correct ≈ 28% shows it cannot consistently reverse bindings when context changes.

3. **Family-specific patterns**:
   - Transfer (give/receive) is best: the model has partial transfer-relation knowledge (49–56% both)
   - Comparative (more/fewer) is second: 44–45% both
   - Temporal is NEGATIVE: M < 0 means context makes binding WORSE. Strong name priors (|bias|=3.4–3.7) with negative context effect.
   - Spatial is weak but positive: 18–21% both, near chance
   - State_change is moderate: 21–30% both
   - Container (held out) is zero: complete failure on unseen relation

4. **Target priors are substantial** (|bias| = 2.6–2.7 nats) but cancel in the four-cell margin by construction. The scoring correctly isolates context-conditioned binding.

5. **Reheat is marginally better** (+0.027 both_correct), consistent with its higher cheap7 but insufficient for SOTA.

## Decision for Route B training

The frozen-scoring gate is **passed**:
- ✅ Models are unsaturated (28% both vs 25% chance)
- ✅ Context conditioning is positive but insufficient (M > 0)  
- ✅ No provenance overlap with evaluation text (0 n-grams)
- ✅ Entity transfer is flat (train/held-out equal)
- ✅ Held-out family is at zero (massive room for transfer)
- ✅ WWM useful density ~1.2% (low but nonzero)

Concerns requiring mitigation before training:
- ⚠️ Target priors are high (|bias| = 2.6) — training must not merely reinforce priors
- ⚠️ Temporal M is negative — temporal templates may need revision or exclusion
- ⚠️ 17k words is only 0.17% of 10M budget — may need more packets or concentrated placement

## Files

- Scoring pairs: `data/role_switch_packets/scoring_pairs.jsonl`
- Legal40k results: `data/frozen_exchange_scoring/legal40k_100M_pair_results.jsonl`
- Reheat results: `data/frozen_exchange_scoring/reheat_100M_pair_results.jsonl`
- Summary: `data/frozen_exchange_scoring/frozen_exchange_summary.json`
- Packets manifest: `data/role_switch_packets/packet_manifest.json`
