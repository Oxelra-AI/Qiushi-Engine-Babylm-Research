# Complete Mechanism: Two-Channel Model of Experience Value in Data-Efficient Learning

## The principle

Under a fixed experience budget, each training example installs learning through two separable channels:

**Channel 1 — Representational facilitation (weight-mediated, always positive)**
Structured experience builds entity/item representations that transfer to related tasks through model weights. This channel is active regardless of in-context attention during training. It improves identification of the correct answer within the appropriate output family.

**Channel 2 — Relation-specific routing (attention-mediated, in-window, relation-specific)**
When source and target co-occur in the same training window with unblocked attention, the model installs a cross-span computation matching the practiced relation. Identity practice installs copy routing; content-conditioned practice installs transformation routing. The routing fires specifically when the model recognizes the source in-window at evaluation time.

The net effect on any evaluation depends on: (1) representational benefit from Channel 1 (always positive), (2) match/mismatch between the installed routing and the evaluation task (Channel 2, sign depends on task), and (3) budget displacement of alternative evidence.

## Evidence map

### Channel 1: Weight-mediated facilitation

| Evidence | Source | What it shows |
|----------|--------|---------------|
| ident_masked ≈ ident_full on hcg | corr acquisition result (synthetic) | +3.25 vs +3.07, Δ=−0.18: facilitation is weight-based |
| ident_masked ≈ ident_full on within-family identification | corr acquisition result/10 | Blocking attention during identity training doesn't remove within-family improvement |
| Facilitation independent of task cue | Step10b | Within-family Δ: −3.38 (typed) vs −3.00 (constant) — same regardless of task specification |
| Identity exposure helps even vs noise baseline | corr acquisition result/10 | ident_full within=6.6 vs neutral within=9.8 (typed cue, Step10b) |
| Wrong correspondence actively hurts | corr acquisition result | wrong hcg=+0.51 < neutral hcg=+1.67: wrong evidence degrades even weight-level identification |

### Channel 2: Attention-mediated relation routing

| Evidence | Source | What it shows |
|----------|--------|---------------|
| ident_full hcp_s = 10.15 vs ident_masked hcp_s = 15.91 | corr acquisition result | Full attention during identity training installs strong copy routing (5.76 nats difference) |
| ident_full hc_s = 13.13 vs ident_masked hc_s = 9.89 | corr acquisition result | Copy routing hurts correspondence NLL (+3.24 nats) |
| REPEAT_MASKED = 0 copy/content gain | identity shortcut bridge result | Blocking attention at evaluation eliminates the routing computation entirely |
| REPEAT_SPLIT eliminates recurrence cost | BabyLM relation-learning integration | Token-nonoverlap cost drops from −0.75/−1.04 to −0.03 when in-window co-occurrence removed |
| REPEAT_SPLIT eliminates copy benefit | BabyLM relation-learning integration | Copy gain drops from +0.50/+0.67 to −0.20 |
| 11-14× source specificity ratio | BabyLM relation-learning integration | Copy routing fires specifically on recognized source (T condition), not unrelated text (U condition) |
| REPEAT suppresses target; VIEW improves target | BabyLM relation-learning integration | Same source-mass routing, opposite effect on target: identity misfire vs content-conditioned reading |
| Adversarial constraints reverse h1 orientation | topology phase1 result | Wrong relation content installs wrong computation: h1 margin +11.9→−20.8, selective to h1-involving edges |

### Channel interaction and task dependence

| Evidence | Source | What it shows |
|----------|--------|---------------|
| Entity 0-ops: REPEAT +9.42 over VIEW | BabyLM Entity-depth comparison | Direct retrieval: both channels help (facilitation + copy routing matches task) |
| Entity 3-4 ops: VIEW +7.54/+8.74 over REPEAT | BabyLM Entity-depth comparison | Multi-op transformation: facilitation helps but copy routing hurts (mismatch) |
| Task cue reverses absolute NLL: ident 6.78 vs neutral 10.06 | Repaired typed cue | When model knows to rewrite, Channel 1 facilitation dominates |
| Without task cue: ident 13.29 vs neutral 9.89 | Matched constant cue | Without specification, Channel 2 routing shifts output family |

## Why earlier interpretations were partially wrong

**corr acquisition result "facilitation + output competition"**: Correctly identified facilitation, incorrectly attributed output competition to a learning mechanism. The output competition was partly task-ambiguity artifact (family-level) and partly real routing (token-level). Step10b separated these.

**matched support source contrast result "no source-specific excess"**: The synthetic substrate had same attribute tokens across source and rewrite, so the nonoverlap condition wasn't faithful. When properly disjoint (Step8b), the excess was still near zero because the synthetic task lacked the in-window identity training that creates source-specific routing in BabyLM.

**task ambiguity decomposition result original "pure opportunity cost for BabyLM"**: Incorrect. the BabyLM T/U decomposition shows the BabyLM cost is source-specific (+0.45-0.69 true source vs −0.30-0.35 unrelated source), which opportunity cost cannot explain. The cost is recognition-triggered routing.

## Unified principle statement

**In data-efficient learning, experience value decomposes into representational structure (what the model learns about items) and installed computations (what the model learns to do between items). Representational structure transfers broadly through weights and is always beneficial. Installed computations are relation-specific, require in-window co-occurrence, fire on source recognition, and help or hurt depending on whether the evaluation task matches the practiced relation. Data efficiency is maximized when the practiced relation matches the target relation, or when task specification can override mismatched routing.**

## What remains

1. **VIEW_SPLIT** (in progress at the time): Does content-conditioned routing also require in-window co-occurrence? This determines whether both sides of the principle are local.

2. **Formal mathematical statement**: The two channels can be expressed as additive contributions to the log-probability decomposition, with the routing channel conditional on source recognition.

3. **Architecture generality**: The routing channel should depend on attention mechanisms; the representation channel should be more universal. Testing with causal-LM and different attention patterns would strengthen the principle.

4. **Scale transfer to BabyLM**: Whether the measured synthetic magnitudes predict the BabyLM magnitudes under realistic mixing ratios.

## Key files

- corr acquisition result (shared backbone): `notes/corr_acquisition_result.md`, `data/corr_acquisition/summary.json`
- task ambiguity decomposition result (NLL decomposition): `notes/task_ambiguity_decomposition_result.md`
- Repaired cue + constant control: `data/revision_010b_repaired_cued/results.json`
- identity shortcut bridge result (attention as causal variable): `notes/identity_shortcut_bridge_result.md`
- topology phase1 result (adversarial constraint reversal): `notes/topology_phase1_result.md`
- BabyLM relation-learning integration (locality + specificity): `research/notes/relation_learning/decisive_results_synthesis.md`
