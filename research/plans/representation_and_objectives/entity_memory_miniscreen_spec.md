# Implementable mini-screen for an entity-keyed event-state memory block

## Scientific target

Recent evidence reduces the unsolved computation to one product:

`same_entity(query, event_argument) × event_result_state -> query_answer_state`.

route2 factorial causal review showed that old checkpoints can often represent the first term, but not the product. The next route should therefore create a main-path operation that separates persistent entity identity from mutable event-conditioned state and forces write and read to share the entity address. It should not require the old checkpoints to hide a recoverable answer.

## Architecture arms for the first small screen

All arms use the same tokenizer, sequence length, optimizer family, training text order, mask seed policy, depth/width as closely as possible, and word/token exposure accounting.

1. **Vanilla tiny DeBERTa-style MLM.** Matched baseline.
2. **Shared-key recurrent entity memory.** Mid-encoder insertion. Each mention/token emits an address key `k`; mentions of the same entity are softly consolidated. Initial-state clauses initialize mutable slot values. Event tokens write through the same key space, with an event/value vector and an update gate. Query/mask context reads through the same key space and injects the read vector before later encoder layers.
3. **Independent-key memory.** Same size and placement, but write and read use non-shared key spaces. This separates extra capacity from shared entity addressing.
4. **Non-persistent event cross-attention.** Event tokens can influence query tokens but there is no maintained entity state. This separates one-hop lexical routing from state memory.
5. **Parameter-matched FFN/adapter.** Same approximate parameters and placement, no entity-keyed write/read path.

If implementation time is tight, arms 1, 2, and 3 are mandatory; arm 4 or 5 can be added next.

## Preferred shared-key recurrent block

At an insertion layer, let contextual token states be `h_i`.

- Address: `k_i = W_k h_i`.
- Mention selection: `p_i = sigmoid(w_s^T h_i)`; no external tagger is used.
- Mention consolidation: repeated names/noun heads share slot content through a soft coreference kernel over keys and surface spans. First version may restrict training examples to repeated explicit nouns, then evaluate a pronoun split separately.
- Initial slot value: phrases such as `The cup was empty` initialize `m_cup^0`.
- Sequential event update: for each event token/span `t`, write address mass
  `a_ti = softmax_i((W_q h_t)^T k_i + b_relpos(t,i))`.
  Event value `v_t = W_v h_t` updates each slot by
  `m_ti = (1-g_ti)m_(t-1)i + g_ti phi(m_(t-1)i, v_t, m_(t-1)i ⊙ v_t)`.
  This makes overwrite and prior-dependent transitions representable.
- Query read: query/mask context reads
  `rho_ji = softmax_i((W_r h_j)^T k_i)` and injects `h_j <- LN(h_j + alpha W_o sum_i rho_ji m_Ti)` before later normal encoder layers.

Important design: addresses are stable entity identities; slot values are mutable state. Later layers see the read result in the main path.

## Frozen synthetic data object

Generate a counted synthetic corpus of two-entity state-update stories, with full JSON provenance.

Core quartet, with target/foil, action/state tokens, length, and local geometry matched:

- `The cup was empty. The bowl was full. Emma filled the cup. The cup is now [MASK].`
- `The cup was empty. The bowl was full. Emma filled the bowl. The cup is now [MASK].`
- entity order swapped;
- query entity swapped;
- event direction reversed.

Families: open/closed, empty/full, clean/dirty, wet/dry, hot/cold, lit/dark, inflated/flat, locked/unlocked, broken/fixed, raised/lowered.

Held-out splits:

- entity surface forms and noun heads;
- action lemmas;
- complete state/action families;
- templates and word order;
- direct repeated noun versus pronoun/description;
- single event versus multi-event.

Add multi-event cases early enough to test state maintenance, not only verb-to-answer lookup:

- A updated, B untouched;
- A updated then reversed;
- A and B updated in interleaved order;
- no-op and failed/negated actions (`tried to`, `did not`, `failed to`, `may have`).

## Measurements that make the screen meaningful

Use the same frozen readout set for every arm and seed.

1. Affected-query accuracy rises while unaffected-query accuracy stays high.
2. Paired update margin is positive on held-out families:
   `[logit(updated)-logit(prior)] when queried entity is affected` minus the same quantity when the other entity is affected.
3. Entity swap changes the predicted state only for the swapped entity; target-swap reverses the margin sign.
4. Shared-key memory beats independent-key memory and non-persistent cross-attention on paired entity/event/state combinations, not only aggregate accuracy.
5. Write mass concentrates on the acted-on entity; read mass concentrates on the queried entity; the product transport from event to query through an entity slot is higher for matched entity pairs.
6. Permuting write-to-entity correspondence at inference damages the memory arm; perturbing an irrelevant slot has small effect.
7. Formerly absent route2 factorial causal review-style paired event-polarity signal appears at the memory block output and downstream layers, and is connected to MLM output movement.
8. The existing natural readouts, unchanged and never trained on, do not collapse: orthogonal route screen summary property surface, EWoK transition subsets, GlobalPIQA hard/parallel subsets.

A synthetic-only win forms an architecture object but does not yet prove BabyLM endpoint value. Natural-surface movement without the shared-key comparisons does not identify the mechanism.

## Training scale and use of H100s

First implementation run: 0.2M-0.5M synthetic words, two seeds if runtime allows. Use both H100s for matched arms in parallel. This is justified because it decides whether the proposed main-path operation can form the missing product at all. Do not launch a full 100M BabyLM run from this alone.

If the shared-key memory arm shows the expected held-out composition pattern and the non-shared/capacity arms do not, the next small run can mix the legal compact-view stream with a counted synthetic state-update slice and evaluate unchanged natural interaction readouts. Only after that would broad cheap7/full official measurement be scientifically warranted.

## Implementation plan

The proposed construction requires:

- a tiny synthetic data generator with fixed partitions and SHA hashes;
- a lightweight DeBERTa-compatible memory block inserted into a tiny model first;
- arm configs for vanilla, shared-key memory, independent-key memory, and one non-persistent/capacity comparison;
- training scripts that record word counts, token counts, mask counts, seeds, and model hashes;
- the frozen paired readout script, including entity/target swap files and internal write/read transport summaries.

Training remains conditional on those files existing and a CPU correctness check of the data and tensor shapes.

independent_review support: `data/external/independent_review01_generator1_integration.md` and `data/external/independent_review01_verifier1_integration.md`.
