# earlier analysis — Role-binding mechanism route: entity-state slots + nonce-symbol update micro-world

## What binding learning response v2 results closed and what it did not

binding learning response v2 results (repaired downstream-query binding learning-response, 4 arms × 3 seeds) showed that the current pairwise-margin masked-target contrastive loss on 48 template pairs produces 100% training accuracy but zero held-out binding generalization. All margins are ≤0.003, seed variance dominates arm differences, and no arm separates from chance.

This closes the specific naive template-contrastive objective. It does NOT close entity-state binding as a learnable mechanism. The proposed follow-up must change role-binding representation and credit assignment, not repeat a template margin loss.

## Why score-chasing through S1 known columns is not the route

S1 12×384 known-seven sum is 296.525, ~0.37 below protected 8×480. To exceed leader Overall 41.80, S1 would need SuperGLUE+AoA >79.675, about +11.83 above the protected model's 67.847. No evidence supports such a jump. The 1.27 Overall gap cannot be closed through existing-column advantages alone.

## The next mechanism: Writable Entity-State Slots (WESS)

The proposed mechanism is: persistent, entity-indexed state slots with gated event-driven updates and bottlenecked query readout. This directly addresses the credit-assignment gap identified across structure density conclusion and objective pivot, 167, 171, 173, and 177.

### Core architecture

For each episode with entities \(e_1,\dots,e_K\) and events \(t=1,\dots,T\):

**Slot initialization:** \(m_{k,0} = \text{Pool}\{h_i : i \in \text{first\_span}(e_k)\}\) — entity slots are initialized from their first mention, not from fixed name embeddings.

**Event-driven update:**
\[
z_{k,t} = \text{Attn}(W_q m_{k,t-1}, W_K H_t, W_V H_t)
\]
\[
g_{k,t} = \sigma(w^\top[m_{k,t-1}; z_{k,t}; r_{k,t}])
\]
\[
\tilde{m}_{k,t} = \text{GRU}(m_{k,t-1}, z_{k,t})
\]
\[
m_{k,t} = (1-g_{k,t})m_{k,t-1} + g_{k,t}\tilde{m}_{k,t}
\]

where \(r_{k,t}\) includes mention-span, relative position, and co-reference features. For the first experiment, gold entity spans and event-participant routing are provided.

**Persistence constraint:** \(L_{\text{persist}} = \sum_{t,k}(1-a_{k,t})\|m_{k,t} - \text{sg}(m_{k,t-1})\|_2^2\) where \(a_{k,t}\) is the event-entity participation label.

**Bottlenecked query readout:** The query can only access query tokens, target entity slot, and non-answer structural information — NOT the setup state words directly. This forces the model to route through the state slots.

**Slot-MLM loss:** \(L_{\text{slotMLM}} = -\sum_{t,k} \log p(s_{k,t} \mid m_{k,t}, q_k)\) — supervise state prediction at every event, not just the final query.

**Total loss:** \(L = L_{\text{MLM}} + \lambda_b L_{\text{slotMLM}} + \lambda_r L_{\text{route}} + \lambda_p L_{\text{persist}}\)

## First decisive experiment: nonce-symbol update micro-world

### Data design

Each episode:
- 4 nonce entities, 4 nonce states
- 3–6 events with at least one entity overwritten (s₁→s₂)
- At least 2 distractor entities
- Query randomly selects an entity (not always the most recent)
- Entity names, state names, mention order, and templates randomized per episode

Example:
```
dax is at lup.
wug is at fen.
dax moves to mor.
wug moves to lup.
dax moves to tib.
Where is dax?  tib
```

### Anti-shortcut extrapolation axes

Train on 1–3 updates, test on 4–8 updates. Evaluate separately on:
1. Unseen entities / seen states
2. Seen entities / unseen states
3. Unseen entities / unseen states
4. Unseen entity–state combinations
5. Unseen event orders
6. Unseen template families
7. 1, 3, 7 distractor entities
8. Distractors sharing candidate state words
9. Same entity overwritten 2–4 times
10. Two entities swapping states (bag-preserving)
11. Multiple entities sharing the same state
12. Active/passive and pronoun/full-name alternation
13. Train short sequences, test long sequences

### Three-arm comparison (≥5 seeds each)

1. **Endpoint-only WWM**: standard DeBERTa-v2 MLM on the episodes, query target masked
2. **Explicit step-state supervision**: same Transformer, but after each event, supervise the full entity-state table
3. **WESS with bottlenecked readout**: entity slots with gated updates, persistence loss, slot-MLM, and query-only-slot access

All arms matched on initialization, tokenizer, update count, mask schedule, and masked-token exposure. Add a parameter-matched deeper Transformer control.

### Primary metrics (not binary margin)

1. Per-step per-entity state accuracy
2. Non-participating entity persistence rate
3. Target-entity selective write accuracy
4. Overwrite accuracy (new state replaces old)
5. Endpoint query accuracy
6. Length extrapolation curve (accuracy vs. number of updates)
7. Entity and state renaming equivariance

With K=4 states, random endpoint baseline is 25%.

### Causal intervention evidence (required before claiming mechanism)

1. **Final-state swap**: swap two entity slots before query readout; answer should follow the slot, not the text
2. **Single-step write intervention**: replace the new state written at event t from s to s'; only the target entity's subsequent states should change
3. **Bottleneck ablation**: remove the bottleneck (allow query to attend to full text); if performance improves, the model was bypassing slots
4. **Write ablation**: zero out the last overwrite's Δm for the target entity; answer should revert to previous state
5. **Entity-renaming equivariance**: apply an arbitrary permutation to all entity names; slot contents should permute accordingly
6. **Activation patching**: patch the update at a single event; causal effect should be selective to the target entity

## What success looks like

WESS must show:
- Stable held-out state retrieval on unseen entities, unseen states, and unseen combinations
- Correct overwrite behavior (new state replaces old, not additive)
- Persistence under interference (distractor updates don't change target entity state)
- Length extrapolation (accuracy degrades gracefully, not catastrophically)
- Slot-swap intervention transfers the answer
- Write-ablation reverts to previous state
- Entity-renaming produces slot-permutation equivariance

If only the step-state supervision model passes, the bottleneck is credit assignment, not architecture. If only WESS passes, the bottleneck is persistent entity-indexed storage. If neither passes, entity-state binding is not learnable under the current DeBERTa-v2 encoder and a fundamentally different architecture is needed.

## Relation to BabyLM columns

The nonce-symbol micro-world directly tests the shared mechanism behind:
- **Entity**: identity→attribute/location binding with interference
- **EWoK**: event-driven world-state updates and subsequent inference
- **COMPS**: attribute composition with correct entity, avoiding neighbor misbinding
- **GlobalPIQA**: multi-step action effects on object location, state, and affordance

## Implementation note

The WESS module is a small extension to the DeBERTa-v2 MLM trainer:
- A lightweight GRU or gated MLP for slot updates
- Event-to-slot write attention
- Query-to-slot read attention
- Route and persistence auxiliary losses
- The standard MLM head remains unchanged

The first experiment should use gold entity spans and event-participant labels to isolate the state-tracking mechanism from entity detection/coreference. If the mechanism works, later work can add learned mention detection and routing.

## Next execution

The proposed implementation comprises the nonce-symbol update micro-world data generator and the three-arm training script. The data generator must produce episodes with controlled overwrite, interference, and extrapolation axes. The training script must implement WESS with bottlenecked readout, step-state supervision, and the causal intervention evaluation suite. This is a small experiment (hundreds of episodes, small model, fast training) that will produce decisive evidence about whether entity-indexed state tracking is learnable under this architecture.
