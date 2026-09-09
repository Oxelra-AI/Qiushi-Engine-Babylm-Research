# raw span discovery design — Raw span discovery: removing supplied entity positions

## Scientific purpose

span matcher construction and design showed that candidate IDENTITY binding can be naturalized (context-free
CharGRU matching replaces supplied `<cand>`/`<other>` tokens). raw span discovery design tests
the next level: can entity POSITION discovery also be learned?

In the current mechanism chain:
1. **Entity discovery**: which tokens in the text are entity names? (previously: `<name>` markers or `<cand>`/`<other>` tokens)
2. **Candidate binding**: which entity is the queried candidate? (span matcher construction and design naturalized this)
3. **Relational reasoning**: shared representation + anchors + graph → gauge transport

raw span discovery design removes the supplied entity positions entirely. The model receives raw
lowercase text with actual names embedded as ordinary tokens. A per-token
CharGRU matcher computes character-level similarity against candidate/other
name strings, and a three-way softmax (candidate/other/neither) gates each
token embedding. Non-name tokens should have high P(neither) and retain their
word embeddings. Name tokens should have high P(candidate) or P(other) and
receive the learned candidate/other parameter embeddings.

## Architecture: SpanGatedEncoder

Key differences from span matcher construction and design SpanMatcher:
- span matcher construction and design: `<name>` tokens at known positions → matcher operates on 2 marked positions
- raw span discovery design: all tokens eligible → matcher operates on ALL seq_len positions
- span matcher construction and design: word embeddings have `<name>` token → tells model where entities are
- raw span discovery design: word embeddings have actual name tokens (or `<unk>` for eval names)

Components:
1. **CharGRUEncoder** (shared, context-free): 28-char vocab, 16-dim embedding, 16-dim GRU
2. **Per-token matching**: dot(token_char_vec, cand_char_vec), dot(token_char_vec, other_char_vec)
3. **Three-way gating**: softmax([score_cand, score_other, neither_bias]) with neither_bias=2.0
4. **Gated embedding**: P(neither)·word_emb + P(cand)·cand_param + P(other)·other_param
5. **Main GRU**: bidirectional, 64 hidden → mean pool → scalar head

The rest of the architecture (shared_trunk/untied, bridge_sign, train/eval split,
evaluation pipeline) is identical to causal gauge experimental design/294.

## Key design choices

- `neither_bias` initialized to 2.0: at init, most tokens classified as "neither"
- Special tokens (`<bos>`, `<eos>`, `<hyp>`) forced to "neither" via -1e6 masking
- Character IDs: lowercase a=1..z=26, other=27, padding=0
- Vocabulary: <pad>, <unk>, <bos>, <eos>, <hyp>, plus all content tokens including
  train names. Eval names become <unk>.
- Oracle mode: char IDs at name positions replaced with candidate/other char IDs
  (makes matching trivially correct → validates architecture)

## CPU verification

Exact raw-substring scan produces identical token sequences to causal gauge experimental design's
`normalize_event_for_candidate` on all 1951 unique event+candidate combinations.
No mismatches. 32 unique names, all single-token. One substring pair (Eli/Felix)
provides a natural stress test for character matching. The information needed for
causal gauge experimental design-level canonicalization is fully available in raw text + name strings.

Saved at: `data/exact_scan_verification/`

## Experiments

Seed 29600 for all runs.

| GPU | Mode | Conditions | Bridge signs | Expected time |
|---|---|---|---|---|
| 0 | learned | shared_trunk | +1, -1 | ~40 min |
| 1 | learned | untied | +1, -1 | ~40 min |

### Decisive readouts

For learned shared_trunk vs untied under bridge_sign ±1:
1. **State matching accuracy**: P(cand) > 0.5 at candidate position, P(other) > 0.5 at other position (both train and eval names)
2. **Train fit**: train_state_acc, train_cmp_acc should reach 1.0
3. **Graph state transport**: graph_same should be 1.0 for bs+1, 0.0 for bs-1 (shared_trunk only)
4. **Unchanged preservation**: unchanged should be 1.0
5. **Held-held closure**: hh_closure should be 1.0
6. **Row-paired sign reversal**: all graph-transfer changed d_e values reverse between bridge signs
7. **Mixed held-seen**: separate diagnostic (may differ from causal gauge experimental design)
8. **Matching diagnostics**: trained matching accuracy on eval (held) names

### Interpretation framework

| Outcome | What it means |
|---|---|
| Learned shared_trunk: full gauge fingerprint + high matching | Entity position discovery works; the mechanism survives without supplied entity markers |
| Learned shared_trunk: train fit + matching OK but no gauge transport | Architecture or gradient flow issue in SpanGatedEncoder |
| Learned shared_trunk: matching fails on eval names | Character matcher doesn't generalize to novel name strings |
| Untied: matches train, no gauge transport | Same as causal gauge experimental design/294: shared representation is necessary |
| Oracle shared_trunk: full gauge fingerprint | Architecture validation (SpanGatedEncoder preserves mechanism) |

## Connection to principle

If successful, this shows that the gauge-transport mechanism does NOT require:
1. Supplied candidate/other identity tokens (removed in span matcher construction and design)
2. Supplied entity position markers (removed in raw span discovery design)

The remaining supplied interfaces would be:
- Binary ownership (exactly two participants)
- Comparison semantics (same/different label structure)
- Changed-fact routing (separate static scorer)
- The separate static scorer itself

## Files

- CPU verification: `data/exact_scan_verification/`
- Script: `training/scripts/raw_span_discovery_probe.py`
- Design note: this file
- Output (shared_trunk): `data/learned_shared/`
- Output (untied): `data/learned_untied/`
