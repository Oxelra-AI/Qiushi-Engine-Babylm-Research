# span matcher construction and design — Context-free SpanMatcher: Gauge transport with learned candidate binding

## Result summary

The context-free SpanMatcher experiment demonstrates that the shared-representation
gauge-transport mechanism **survives when candidate binding is learned from character
strings rather than supplied as a vocabulary token**.

### Key findings

1. **CharGRU matching generalizes perfectly to held names**.
   State match accuracy = 1.0, comparison match accuracy = 1.0, on 16 completely
   disjoint eval names (zero overlap with 16 train names). The tiny character-GRU
   (26 chars × 16 dim, 16 hidden ≈ 1K params) compositionally encodes novel
   character sequences and correctly identifies query candidates.

2. **Gauge transport works with learned binding**.
   Learned shared_trunk:
   - bs+1: graph_same=1.0, unchanged=1.0, hh_closure=1.0, pair_both=1.0
   - bs-1: graph_same=0.0, unchanged=1.0, hh_closure=1.0, pair_both=0.0
   - graph_same_margin reverses from +18.214 to -17.449
   
   Oracle shared_trunk (validation):
   - bs+1: graph_same=1.0, unchanged=1.0, hh_closure=1.0
   - bs-1: graph_same=0.0, unchanged=1.0, hh_closure=1.0
   - graph_same_margin reverses from +17.986 to -17.449

3. **Row-paired sign reversal is complete**.
   256/256 graph-transfer d_e values reverse between bs+1 and bs-1 for both
   oracle and learned matchers. Learned bs+1 d_e coordinates are perfectly
   aligned with oracle bs+1 (256/256 same sign).

### What this establishes

The candidate-binding interface (`<cand>`/`<other>` token replacement) is the first
supplied interface to be successfully naturalized. The gauge mechanism works when:
- Candidate identity is determined by a context-free character-level matcher
- Match probabilities gate single-token name embeddings
- The gated representation feeds the shared GRU for relational reasoning

The result confirms that the gauge mechanism is **not dependent on a vocabulary-level
addressing primitive**. A compositional matching process can substitute, provided it
achieves correct binding on held identities.

### What this does not establish

- The SpanMatcher uses explicit `<name>` token positions. In natural text, entity
  positions are not delimited.
- The matching is binary (two participants per event). Natural events involve
  variable numbers of entities.
- Comparison semantics (same/different ownership) are still supplied.
- Changed-fact routing (separate static scorer) is still supplied.
- The result is at the controlled probe scale, not BabyLM or transformer scale.

### Architecture detail

**Context-free SpanMatcher**: Names are encoded as character sequences OUTSIDE the
main GRU. The SpanMatcher has its own tiny character GRU that processes characters
independently of the event context. Match scores are computed via scaled dot product
(temperature=5.0) + softmax.

**Match-gated embeddings**: At name token positions, the token embedding is replaced
with `p * cand_embed + (1-p) * other_embed`, where p is the match probability. With
correct matching (p≈1.0 for candidate, p≈0.0 for other), this produces embeddings
functionally identical to `<cand>`/`<other>` tokens.

**Key design choice**: Single-token `<name>` placeholders (not character-by-character
encoding in the main sequence). This keeps the main GRU architecture identical to
causal gauge experimental design's supplied harness, isolating the binding mechanism as the only change.

### Remaining controls to run

- Learned untied ± (running on GPU1): should fail gauge transport (no shared representation)
- Fresh-rename with novel names: tests generalization beyond train+eval pools
- Comparison-only filter: corrected per corrected multiseed gauge affine and next raw binding independent_review

### Files

- Script: `training/scripts/span_matcher_gauge_probe.py`
- Oracle output: `data/oracle_gauge/`
- Learned bs+1 output: `data/learned_gauge/`
- Learned bs-1 output: `data/learned_gauge_bsminus/`
- Learned untied output: `data/learned_gauge_untied/` (pending)
- Construction notes: `notes/span_matcher_construction_and_design.md`
- This synthesis: `notes/span_matcher_learned_binding_result.md`
