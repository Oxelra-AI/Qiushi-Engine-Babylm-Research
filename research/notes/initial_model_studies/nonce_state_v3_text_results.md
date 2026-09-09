# nonce state v3 text results — Text-interface v3: WESS mechanism survives natural-language encoding

## Evidence

- Script: `scripts/nonce_state_v3_text.py`
- Results: `data/nonce_state_tracking/v3_text_interface.json`

## Design change from v2

Events are now rendered as natural-language tokens (`dax moves to lup.`) processed
by a shared 3-layer Transformer encoder. WESS extracts entity and state representations
from encoder hidden states at gold span positions. The endpoint and step-supervised
arms use the same encoder. This tests whether the slot mechanism works with
contextualized distributed representations, not isolated symbol embeddings.

## Five-seed aggregate results (random baseline = 0.25)

| split | endpoint | step_supervised | wess_gold | wess_shuffled |
|---|---:|---:|---:|---:|
| iid | 0.723 ± 0.023 | 0.733 ± 0.030 | **0.999 ± 0.002** | 0.996 ± 0.008 |
| new_ent | 0.748 ± 0.014 | 0.741 ± 0.019 | **1.000 ± 0.000** | 0.997 ± 0.006 |
| new_state | 0.406 ± 0.069 | 0.426 ± 0.035 | **0.848 ± 0.021** | 0.790 ± 0.041 |
| new_both | 0.418 ± 0.067 | 0.396 ± 0.036 | **0.832 ± 0.035** | 0.783 ± 0.047 |
| long | 0.239 ± 0.028 | 0.259 ± 0.027 | **0.984 ± 0.007** | 0.985 ± 0.011 |
| overwrite | 0.299 ± 0.034 | 0.296 ± 0.050 | **0.987 ± 0.007** | 0.988 ± 0.007 |

## Causal interventions (overwrite split)

| intervention | wess_gold | wess_shuffled |
|---|---:|---:|
| Slot swap transfer | **0.904** | 0.316 (≈ random) |
| Last-write ablation revert | **0.996** | 0.996 |

## Scientific findings

### 1. WESS mechanism survives the text interface

The v2 structured-symbol mechanism (100% IID, perfect interventions) degrades only
slightly when moved to text: IID 0.999, long 0.984, overwrite 0.987. The slot-swap
transfer drops from 1.000 to 0.904 (still strongly above chance at 0.25). The
mechanism's core algorithm — initialize slots from entity mentions, update through
GRU, read through entity-indexed projection — works with contextualized representations.

### 2. Standard Transformer and step supervision both FAIL

Endpoint (0.723 IID, 0.239 long, 0.299 overwrite) and step-supervised (0.733 IID,
0.259 long, 0.296 overwrite) are near-random on the critical long and overwrite splits.
Step supervision does NOT help — providing explicit per-event state labels through a
standard encoder does not teach the overwrite algorithm.

**The bottleneck is not credit assignment alone but persistent entity-indexed storage.**

### 3. Critical new diagnostic: shuffled routing reveals mechanism semantics

wess_shuffled achieves nearly identical accuracy to wess_gold (long 0.985 vs 0.984,
overwrite 0.988 vs 0.987) but slot-swap transfer drops to 0.316 (near-random for
K=4). This means:
- The shuffled model learned to compensate for wrong initial routing
- Its slots contain useful state information (high accuracy)
- But the entity→slot mapping is not preserved (swap doesn't transfer)
- The gold-routed model maintains interpretable entity indexing (swap at 0.904)

This demonstrates that gold routing is important for **causal interpretability** of
the mechanism, not just for task performance. A model can achieve the same accuracy
with internally rerouted representations, but only gold routing produces a mechanism
whose internal states can be intervened on meaningfully.

### 4. Unseen states are the remaining weakness

new_state (0.848) and new_both (0.832) are the weakest splits. Since candidate
retrieval scores against encoder embeddings of state tokens, unseen state tokens
have less-trained representations. This is an artifact of the shared embedding
table and limited vocabulary, not of the slot mechanism itself.

## Relation to BabyLM SOTA goal

This result establishes:
1. Entity-indexed persistent slots solve the state-update algorithm where
   standard Transformers fail, even with a shared text encoder.
2. The mechanism generalizes to unseen entities, unseen states, long sequences,
   heavy overwrite, and is causally intervened (slot swap, write ablation).
3. Step supervision alone (credit assignment) is insufficient — the inductive
   bias of persistent entity-indexed storage is the key ingredient.

## Remaining gaps before BabyLM integration

1. **Gold span positions**: still provided. Next test: detect entity/state spans
   from tokens using attention or learned span heads.
2. **Formulaic templates**: "X moves to Y" is repetitive. Need diverse event
   phrasing to test robustness.
3. **DeBERTa-v2 encoder**: v3 uses a simple 3-layer Transformer. Need to test
   with DeBERTa-v2 architecture and larger hidden size.
4. **MLM integration**: BabyLM uses masked-language modeling. The WESS module
   must be added as an auxiliary route within an MLM-compatible trainer.
5. **Natural-language state words**: in BabyLM, states are real words (cities,
   rooms, attributes), not nonce symbols. Candidate retrieval must work over
   the full vocabulary or use span-based decoding.
6. **Budget compliance**: all synthetic episode text counts toward the 10M-word
   limit. Need to determine the optimal proportion and integration strategy.
