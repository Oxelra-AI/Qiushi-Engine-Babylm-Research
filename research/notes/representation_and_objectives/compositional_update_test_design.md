# compositional update test design — Compositional Entity-Event-State Update: Mechanism Validation and Test Design

## 1. Context and Motivation

The compact-view route produced a strong DeBERTa-specific effect (+2.4164 equal7 over literal
repetition) but failed to transfer to RoBERTa (factorial interaction item review factorial: EWoK+Entity item interaction
−0.005 pp, stable-family aggregate +0.012 pp). Frozen transfer prediction (frozen transfer test and route decision) showed
DeBERTa compact gains anti-predict RoBERTa movement. The compact-view advantage remains a real
DeBERTa practical asset but is not the central transferable principle.

The persistent natural binding failure (route2 factorial causal review) shows that across checkpoints, DeBERTa:
- Represents "which entity was acted on" (affected-entity code at 0.825 accuracy, layer 6)
- But cannot compose entity-identity + action-result → output-usable final state (0.541, chance)
- Always outputs the prior/initial state for both entities (unaffected accuracy 1.0, affected 0.0)

This is the same computational deficit that EWoK conditional reversals and Entity Tracking probe:
the model knows local compatibility but fails context-conditioned binding when entities exchange
roles under competing contexts. The missing operation is a **main-path compositional update**
over (entity, event, result_state, time).

## 2. Completed CPU Mechanism Validation

### 2.1 Corpus repair

The earlier analysis counterbalanced corpus was correctly designed (96 held quartets all with mixed
answers, majority class 6.2%, no answer shortcut) but the trainer expected an `events` field
that the generator never wrote, making the EntityMemory module completely inert (all events
invalid → no writes).

**Repair**: compositional update test design annotation script extracted events from text templates:
- 12,000 train: 11,040 binding (1 event each), 384 transition_acq (1 event), 384 state_acq
  (0 events), 192 identity_acq (0 events)
- 1,088 eval: 896 binding (1 event each), 192 multi_event (3 events each)
- Zero validation errors. Annotated corpus: `data/annotated_corpus/`

### 2.2 Three-arm comparison (1,000 train records, 8 epochs, CPU, seed 43022)

| Arm | Params | Loss | Held Recomb | Affected | Unaffected | Quartet All | Multi-Event |
|-----|--------|------|-------------|----------|------------|-------------|-------------|
| vanilla | 4,975,296 | 1.143 | 0.497 | 0.458 | 0.536 | 0.000 | 0.500 |
| shared_key oracle | 5,271,745 | 0.068 | **0.776** | **1.000** | 0.552 | 0.552 | 0.531 |
| shared_key learned | 5,271,745 | 0.042 | **0.833** | **1.000** | **0.667** | **0.656** | 0.500 |

Write-permutation control (entity-to-slot swap):
| Arm | WP Held Recomb | WP Affected | WP Unaffected |
|-----|----------------|-------------|---------------|
| shared_key oracle | 0.221 | 0.443 | 0.000 |
| shared_key learned | 0.167 | 0.333 | 0.000 |

### 2.3 What this establishes

1. **Entity-keyed memory enables compositional binding**: Both memory arms dramatically
   outperform vanilla on *held* entity×transition recombinations (0.776/0.833 vs 0.497).
   The effect is not memorization — held combinations never appeared in training.

2. **The mechanism can be learned**: Learned attention outperforms oracle (0.833 vs 0.776),
   showing soft attention discovers flexible entity-event associations. This is essential for
   natural-text integration where oracle positions are unavailable.

3. **Write-permutation confirms selective updating**: Permuting writes destroys performance
   (0.167-0.221), and specifically inverts the affected/unaffected asymmetry (unaffected → 0.0).

4. **Multi-event temporal overwrite remains unsolved**: Neither arm exceeds chance (0.50) on
   interleaved reversals. The multi-event items are eval-only; with training exposure or
   more data, this may improve. This is a known limitation of the current corpus/training.

5. **Unaffected-entity preservation is the bottleneck**: affected=1.0 for both memory arms,
   but unaffected = 0.552 (oracle) / 0.667 (learned). The memory correctly updates the acted-on
   entity but imperfectly preserves the unacted entity's state.

## 3. Natural Bridge: The Binding Failure Connection

### 3.1 The same failure mode

route2 factorial causal review measured DeBERTa chck_82M on structurally identical binding probes:
- Unaffected accuracy: **1.0** (correct — preserves prior state)
- Affected accuracy: **0.0** (wrong — outputs initial instead of result state)
- Hidden affected-entity code: 0.825 accuracy at layer 6
- Hidden final-state polarity: 0.541 (chance)

This is the exact complement of the TinyMLM mechanism validation:
- The entity-keyed memory module achieves affected=1.0 and lifts the unaffected bottleneck
- DeBERTa without memory achieves unaffected=1.0 but affected=0.0
- The missing computation is the same: compositional state update

### 3.2 The internal signature

**Entity-Event Binding Fidelity (EEBF)**: For two entities with the same initial state where
one is acted upon and the other is not:

- EEBF_affected = P(model predicts result_state | query=acted entity)
- EEBF_unaffected = P(model predicts initial_state | query=unacted entity)
- EEBF_composite = (EEBF_affected + EEBF_unaffected) / 2

**Baseline (route2 factorial causal review)**: DeBERTa chck_82M has EEBF_affected=0.0, EEBF_unaffected=1.0,
EEBF_composite=0.50. The model cannot perform the compositional update.

**Mechanism validation**: TinyMLM+learned memory has EEBF_affected=1.0,
EEBF_unaffected=0.667, EEBF_composite=0.833.

**Natural EWoK bridge**: EWoK conditional items where the correct answer requires tracking
which entity was affected by an event (social_properties, physical_dynamics, spatial_relations,
physical_relations — the same domains that showed the largest correctness transition analysis transitions). Improvement
on these domains after memory integration would constitute natural bridge evidence.

## 4. Full Experiment Protocol

### Phase 1: DeBERTa Baseline (no training, GPU inference only)

Run chck_82M on the full annotated eval corpus to establish the natural DeBERTa binding failure
on the counterbalanced substrate:
- Affected accuracy (expected: ~0.0 based on route2 factorial causal review)
- Unaffected accuracy (expected: ~1.0)
- Per-family accuracy
- Multi-event accuracy

This confirms the binding deficit on the exact test substrate.

### Phase 2: Frozen DeBERTa + Memory Module (short GPU training)

- Take chck_82M DeBERTa as frozen backbone
- Insert EntityMemory at layer 2 (same fraction as TinyMLM layer 2/4)
- Train ONLY the memory module parameters (~296K new params) on the full 12,000
  synthetic training records for 8-16 epochs
- Evaluate on:
  - Synthetic held_recomb: EEBF_affected, EEBF_unaffected, quartet_all_rate
  - Official-compatible EWoK (especially social_properties, physical_dynamics)
  - Entity Tracking
  - Full BabyLM selected evaluation

**Predeclared readout rule**: The frozen-backbone test succeeds if:
- Synthetic EEBF_composite > 0.70 (above vanilla 0.50 baseline)
- Official EWoK improves by > +0.5 pp compared to chck_82M
- If synthetic succeeds but EWoK does not improve → the mechanism does not bridge to
  natural text from frozen representations; proceed to Phase 3
- If synthetic fails → the DeBERTa representation cannot support entity-keyed memory
  even with ground-truth-like structure; close the route

### Phase 3: Joint DeBERTa Training (full GPU, if Phase 2 succeeds or partially succeeds)

- Initialize fresh DeBERTa 8×480 with EntityMemory at layer 2
- Train on the standard 100M-word compact-view stream + synthetic binding supplement
  (12K binding examples repeated to ~1% of total training tokens)
- Same optimizer, LR schedule, WWM parameters as the historical runs
- Evaluate on:
  - Synthetic EEBF (held recombination + multi-event)
  - Full official-compatible BabyLM evaluation
  - Item-level comparison with chck_82M on EWoK conditional items

**Predeclared readout rule**: The joint training succeeds if:
- Official equal7 > compact_view_reinvest equal7 (44.2886)
- EWoK > compact_view_reinvest EWoK (+2.18 over repeat)
- Entity > compact_view_reinvest Entity (+7.90 over repeat)
- Synthetic EEBF_composite > 0.80

If it fails to beat the compact-view triangle despite having the memory mechanism, the
binding mechanism is real but doesn't improve sample efficiency for natural language.

### Phase 4: Cross-Architecture Transfer (conditional on Phase 3 success)

- Repeat Phase 3 with RoBERTa 8×480 architecture
- Use the same training stream + synthetic supplement
- Compare with the RoBERTa factorial and RoBERTa selected results

**Predeclared readout rule**: Transfer succeeds if RoBERTa EWoK+Entity interaction is
positive and exceeds the earlier analysis factorial baseline (+0.012 pp → needs > +0.5 pp).

## 5. DeBERTa Integration Architecture

### Insert point: after layer 2 (of 8)

route2 factorial causal review found the affected-entity code concentrated at layers 5-6, but the initial entity
representation forms by layer 2. The memory module should see early entity representations
and provide updated state information to layers 3-8.

### Memory module adaptation for DeBERTa

The existing EntityMemory module expects:
- `slot_pos`: integer positions of entity mentions in the sequence
- `state_pos`: positions of initial state words
- `event_pos`: positions of verb and actor entity
- `actor_slots`: ground-truth entity-to-slot mapping (oracle only)
- `query_pos`: position of queried entity
- `mask_pos`: position of [MASK] token

For natural text, these structured positions are unavailable. The DeBERTa adaptation needs:
1. **Learned entity detection**: attention over all positions to find entity-like tokens
2. **Learned event detection**: attention over all positions to find event triggers
3. **Self-supervised entity-event association**: the memory write attention learns to
   associate events with their affected entities from context

This can be implemented as:
```
entity_scores = linear(h).softmax(dim=1)  # (B, L, n_slots) → soft entity assignment
event_scores = linear(h).softmax(dim=1)   # (B, L) → soft event detection
```

The memory module then uses these soft scores instead of hard positions.

### Parameter budget

EntityMemory with d=480 (DeBERTa hidden size):
- slot_key: 480×480 = 230,400
- write_query: 480×480 = 230,400 (or shared)
- read_query: 480×480 = 230,400 (or shared)
- init_value: 480×480+480 = 230,880
- event_value: 960×480+480+480×480+480 = 692,160
- gate: 960×1+1 = 961
- out: 480×480+480 = 230,880
- norm: 960
- Total: ~1.85M (shared) to ~2.08M (independent)

This is ~5-6% of the base DeBERTa-v2 8×480 model (~35.5M params), a modest structural addition.

## 6. Risk Analysis

### Risk 1: Synthetic-only success

The mechanism works on the counterbalanced binding corpus but not on natural text. This
would be a repeat of the role-switch packet failure (frozen exchange scoring analysis). Mitigation: Phase 2
tests frozen DeBERTa + memory on EWoK before full retraining.

### Risk 2: The binding failure is an output-head problem, not a representation problem

If DeBERTa already represents the correct state internally (route2 factorial causal review suggests 0.825 for
affected-entity identity but 0.541 for state polarity), the issue might be in how the
MLM head reads the representation, not in the representation itself. Mitigation: Phase 2
evaluates both synthetic accuracy and official EWoK/Entity; if the memory module improves
synthetic but not official, the bottleneck is elsewhere.

### Risk 3: The memory module interferes with existing competence

Adding structural modifications might improve EWoK/Entity at the cost of BLiMP or
Supplement. Mitigation: predeclared readout requires holding BLiMP+Supplement to rule
out redistributive rather than genuine improvement.

### Risk 4: Insufficient training signal from 12K synthetic examples

The synthetic corpus (12K records, ~218K words) is tiny compared to 100M training words.
The memory module might not receive enough gradient signal. Mitigation: in Phase 3,
increase synthetic supplement proportion or use curriculum (synthetic first, natural second).

## 7. Files

- Annotated corpus: `data/annotated_corpus/{train,eval}.jsonl`
- Mini corpus for validation: `data/annotated_corpus_mini/{train,eval}.jsonl`
- Annotation script: `scripts/annotate_corpus_events.py`
- CPU validation results:
  - Vanilla: `data/mini_vanilla/result.json`
  - Oracle: `data/mini_shared_oracle/result.json`
  - Learned: `data/mini_shared_learned/result.json`
- Original (broken) trainer: `training/scripts/train_entity_memory_v2.py`
- Original corpus: `training/data/counterbalanced_corpus/`

## 8. Next babylm2026 live surface. **Immediate (Phase 1)**: Run chck_82M on the annotated eval corpus (GPU inference, ~2 min)
   to confirm the binding failure on the counterbalanced substrate
2. **Short-term (Phase 2)**: Build the frozen-DeBERTa + memory integration and train on
   the synthetic corpus; evaluate on EWoK/Entity (GPU, ~30 min)
3. **If Phase 2 shows promise**: Phase 3 joint training (GPU, ~1-2 hours)
4. **If Phase 3 succeeds**: Phase 4 cross-architecture transfer
