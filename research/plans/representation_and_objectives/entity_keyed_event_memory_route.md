# Entity-keyed event-memory route after orthogonal route closures

## Updated scientific state

The missing operation is now sharply factorized. route2 factorial causal review showed that the model can encode whether the queried object is the acted-on object, but it does not compose that identity with the event-result polarity into an output-usable state on held-out state/action families. The older requirement that a failing checkpoint already contain a latent recoverable solution is no longer the right standard for the next architecture route: a representation-forming mechanism is meant to create the operation that the old model failed to form.

A bounded read of graph packet v0 found useful relation-preserving compaction but not the needed operation. The packet has 50 rows and does preserve more entity/relation words than ordinary compaction, but the extracted graphs have 0 nonempty before/after state entries, 0 `state_change` relations, 0 `action` relations, and 0 entity-state-update rows with explicit state/action structure. It should not receive H100 training as a mechanism route unless rebuilt into explicit entity-keyed event-state transitions with true correspondence and matched correspondence-destroying comparisons.

## Minimal main-path operation to test

Add an entity-keyed event memory block inside the encoder, not as a late detachable head.

At layer `l`, with contextual token states `h_i`:

1. **Entity slots from tokens.** Each token proposes an entity key `k_i = W_E h_i`; a learned scalar selector `s_i` lets names, nouns, pronouns, and repeated mentions contribute without an external tagger. Consecutive pieces can be pooled locally, but the first implementation can stay token-level.
2. **Event-conditioned write.** Event/result tokens produce a write value `u_t = W_U h_t` and a write query `q_t = W_Q h_t`. The write distribution to entity slots is
   `a_{t,i} = softmax_i(q_t^T k_i + b(relative_position(t,i)))` over entity-like tokens in the same sequence. The slot value is `m_i = sum_t a_{t,i} u_t`, with an order-sensitive bias favoring plausible event-to-argument spans and later events replacing earlier evidence.
3. **Query-conditioned read.** A query token or mask-context token produces `r_j = W_R h_j` and reads `z_j = sum_i softmax_i(r_j^T k_i) m_i`. The read vector is injected before the next self-attention/FFN block: `h_j <- h_j + W_O z_j` followed by layer norm.
4. **Main-path training.** All weights, including the base encoder and the memory block, train together under MLM. The block's read vector is present before later layers, so later attention and the MLM decoder must learn with the bound entity-event-state representation rather than receiving a post-hoc residual.

The important bottleneck is the shared entity key: an event update reaches the same slot that a later query reads. If the event belongs to cup A and the query asks about cup B, the read should retrieve B's unchanged state; if the event belongs to cup B, it should retrieve B's updated state. This directly targets the route2 factorial causal review failure.

## Tight low-cost experiment

### Data object
Create a controlled two-entity corpus with matched sequence geometry:

- Two objects and two initial states are always stated.
- One action verb changes exactly one object's state.
- The query asks either the affected or unaffected object.
- Entities, verbs, and state families have held-out splits.
- State families include reversible pairs such as open/closed, empty/full, clean/dirty, wet/dry, hot/cold, lit/dark, inflated/flat, locked/unlocked.
- Text is natural enough to use the same tokenizer and MLM machinery, but the generation is fully recorded and counted as synthetic research data.

Example pair with geometry held fixed:

`The cup was empty. The bowl was full. Emma filled the cup. The cup is now [MASK].`

`The cup was empty. The bowl was full. Emma filled the bowl. The cup is now [MASK].`

Only the entity-action binding changes; target and foil words are the same. This is the exact factor that route2 factorial causal review found missing.

### Arms
Run the smallest reliable comparison first:

- Vanilla tiny DeBERTa-style encoder with matched depth/width and parameter count.
- Same encoder plus the entity-keyed event memory block inserted in middle layers.
- A parameter-matched non-keyed adapter arm if implementation time permits.

### Training scale
Start with a toy 0.2M-0.5M word synthetic run for implementation and mechanism formation. If the memory arm learns held-out entity/verb/family composition and the vanilla arm does not, run a second small mixed run with the exact legal compact stream plus a counted synthetic state-update slice, still far below endpoint training cost. Do not start a full 100M run from this result alone.

### Measurements
Use unchanged readouts, not templates selected after seeing results:

1. Synthetic held-out entity/verb/state-family accuracy on affected-query and unaffected-query cases.
2. Target-swapped and entity-swapped comparisons showing the model follows the entity-event correspondence rather than target prior.
3. Internal write/read sanity: read vector for a query should be closer to the slot of the queried object than the other object, and event value should change with the action verb/state pair.
4. The same existing natural interaction surfaces: orthogonal route screen summary cross-context property surface, EWoK transition subsets, GlobalPIQA hard/parallel subsets. These are readouts only, not training material.
5. Broad cheap7 columns only after the mechanism screen shows held-out composition; otherwise stop before spending H100 time on official-style evaluation.

### Results that would move the route forward
The route becomes worth a small BabyLM-mixed training run if the memory arm, with matched tokens and held-out families, produces selective affected-query recovery while preserving unaffected-query accuracy, and the same operation survives entity/target-swapped comparisons. It must also avoid immediate collapse on the unchanged natural interaction readouts. A synthetic-only win without natural-surface movement is useful for architecture formation but not enough for endpoint training.

## Practical endpoint note

alpha0.75 private-scale coherent86 is the strongest currently materialized endpoint by projected arithmetic: cheap7 44.18142857142857, SuperGLUE 69.81922238969935, Overall(AoA=0) 42.1210247099666, model SHA `e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8`, carrier SHA `40181994810e21bc823474a3e4ac84c8eb42213e03904d36a60a4698477d1994`.

A direct official AoA runner preflight on `experiments/archive/frontier_consolidation/training/runs/coherent86_private_scale_0p75/hf_model` found that the model root contains only `final` and is missing all 19 required AoA checkpoint directories. The existing AoA runner therefore cannot be used directly on that root. The next endpoint work is to decide how to represent the actual learning history truthfully: either keep AoA as scalar zero in the already validated carrier, or construct a documented lifelong model-root from the real underlying 1M-80M anchor states plus the real coherent tail states if the official submission format can faithfully accept the nonstandard 86M stopping point. This endpoint work is separate from the scientific mechanism route.

## Next construction work

Implement the entity-keyed event-memory operation as a small model block with a controlled data/freezing plan. Runnable code and fixed readout files are prerequisites for the tiny matched experiment. The first run compares vanilla and memory arms on both H100s in parallel, to determine whether a main-path entity-keyed write/read operation can form the missing composition at all.
