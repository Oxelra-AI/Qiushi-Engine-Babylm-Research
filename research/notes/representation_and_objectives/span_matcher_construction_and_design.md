# span matcher construction and design — Context-free SpanMatcher gauge probe: construction and design

## Scientific purpose

raw name binding construction and design-293 showed that replacing `<cand>/<other>` with raw character-encoded names + query attention fails to generalize candidate matching to held names: even with full training fit (train_state 1.0, train_cmp 1.0), held-name graph_same reaches only 0.688 (shared_trunk) vs 1.0 (supplied harness). The clean-init recheck (clean init and attention execution plan) confirmed that the supplied-harness mechanism is not a sequential-reuse artifact.

**Diagnosis**: The query-attention approach entangles name identity with contextual position in the GRU hidden states. When the GRU processes character-by-character name tokens in context, the resulting attention patterns are position/context-specific and don't transfer to new character sequences at different positions.

**Design insight**: Separate candidate binding from relational reasoning by computing matching OUTSIDE the main sequence encoder:

1. **Context-free SpanMatcher**: A tiny character-GRU (26 chars × 16 dim, 16-dim hidden = ~1K params) encodes each name STRING independently (not in sequence context), then matches query to event names via scaled dot-product + softmax.

2. **Match-gated embeddings**: At name token positions in the main sequence, replace the token embedding with `p * cand_embed + (1-p) * other_embed` where p is the SpanMatcher's match probability. This is functionally identical to `<cand>/<other>` replacement when matching is correct.

3. **Standard GRU + shared/untied heads**: The main encoder (GRU → mean pool → scalar head) processes the gated sequence exactly as in causal gauge experimental design.

## Key design choice: single-token name positions

Unlike raw name binding construction and design/293 which character-encoded names INTO the main sequence (producing 28+ token sequences), this design keeps names as single `<name>` placeholder tokens (like causal gauge experimental design's `<cand>`/`<other>`). The character-level processing happens entirely inside the SpanMatcher, separate from the main GRU. This means:

- The main sequence length is identical to causal gauge experimental design (~12 tokens per event)
- The GRU never sees character-level name information
- The ONLY information about candidate identity comes from the match-gated embedding
- This cleanly separates "who is the candidate?" (SpanMatcher) from "what happened to them?" (GRU + heads)

## SpanMatcher architecture

```
CharGRUMatcher:
  char_embed: Embedding(26, 16)    # a-z → 16-dim vectors
  gru: GRU(16, 16, batch=True)    # sequential character encoder
  
  encode_name(char_ids) → 16-dim vector
  forward(query_chars, [name1_chars, name2_chars]) → softmax probabilities
```

The character GRU encodes names as ORDERED character sequences (not bags), so "Omar" ≠ "Maro" even though they share characters. The GRU hidden state captures character identity + order. Since the character set (a-z) is shared between training and evaluation names, new names composed of known characters should produce reasonable encodings.

The temperature-scaled softmax (×5.0) makes match probabilities near-binary for well-separated names.

## Oracle control

Oracle mode uses exact character equality: match_score = 1.0 if query_chars == event_name_chars, else 0.0. This produces exactly the same binary gating as `<cand>/<other>` replacement, serving as:
- Validation that the architecture reproduces causal gauge experimental design results
- Upper bound for the learned matcher

## Conditions and controls

Same as causal gauge experimental design:
- **shared_trunk**: Shared MatchGatedEncoder (GRU + SpanMatcher), separate scalar heads
- **tied**: Single MatchScorer for both state and comparison
- **untied**: Separate MatchGatedEncoders (separate GRUs AND separate SpanMatchers)
- **bridge_sign ±1**: Only flips h0/h2 changed-state anchor scores

Additional diagnostic: **matching_accuracy** reports how accurately the SpanMatcher identifies the query candidate on held-name evaluation rows.

## Potential confounds

1. In `shared_trunk`, the SpanMatcher is shared between state and comparison tasks (same encoder object). This is analogous to the shared GRU, not an additional confound, because candidate matching is the SAME operation for both tasks.

2. In `untied`, the SpanMatchers are SEPARATE (separate encoder objects). This means untied has no shared parameters at all, matching the causal gauge experimental design untied control.

3. `cand_embed` and `other_embed` are learned nn.Parameters (not vocabulary embeddings). This is functionally equivalent to causal gauge experimental design's `<cand>`/`<other>` tokens but initialized differently (randn×0.1 vs standard embedding init).

## Expected results

- **Oracle**: Should closely reproduce causal gauge experimental design gauge-transport results (shared_trunk transports, untied does not)
- **Learned + matching acc 1.0**: If CharGRU matches correctly on held names AND gauge transport works → the mechanism survives learned binding
- **Learned + matching acc < 1.0**: If CharGRU fails on held names → character generalization failure (need better matcher)
- **Learned + matching acc 1.0 but no transport**: If matching is correct but gauge transport fails → the soft gating pathway disrupts the representation (would be surprising)

## Files

- Script: `training/scripts/span_matcher_gauge_probe.py`
- Oracle output: `data/oracle_gauge/`
- Learned output: `data/learned_gauge/`
- This note: `notes/span_matcher_construction_and_design.md`
