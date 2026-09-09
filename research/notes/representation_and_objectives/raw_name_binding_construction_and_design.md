# raw name binding construction and design — Raw-name binding probe: construction and design

## Scientific purpose

The causal gauge experimental design/291 causal gauge-transport result established three necessary factors
for coordinate transport: state anchors + comparison graph + shared representation.
However, the harness supplies candidate addressing by replacing participant names
with `<cand>`/`<other>`. This experiment removes EXACTLY that interface to test
whether the mechanism survives when the model must learn to bind a queried
participant to its occurrences in event text.

## Design changes from causal gauge experimental design

### 1. Character-based name encoding with query

**causal gauge experimental design**: `<bos> during the tablet episode <cand> daxed <other> <eos>`

**raw name binding construction and design**: `<bos> during the tablet episode <N> m i r a </N> daxed <N> o m a r </N> <QRY> <N> m i r a </N> <eos>`

Key properties:
- Names are character-encoded inside `<N>`/`</N>` delimiters
- All 26 lowercase letters are always in the vocabulary (held names representable)
- `<QRY>` separates the event text from the candidate query
- The model must match the query character pattern to event name occurrences
- Both candidates see the same event text; only the query differs

### 2. Vocabulary

Base: `<pad>`, `<unk>`, `<bos>`, `<eos>`, `<N>`, `</N>`, `<QRY>`, `<hyp>` + `a-z`
Word tokens: collected from training data only (same as causal gauge experimental design train-only scope)
Total: 88 tokens (8 special + 26 chars + 54 word tokens)

Eval object words map to `<unk>` (same as causal gauge experimental design train-only condition).
Eval names are representable through character sequences despite never appearing
as word tokens.

### 3. Narrow comparison-only filter (--no-bridge-changed-only)

causal gauge experimental design's `--no-bridge-anchors` removed ALL h0/h2 arm state rows (128/320),
including both changed and unchanged companions.

raw name binding construction and design's `--no-bridge-changed-only` removes ONLY changed h0/h2 bridge anchors
(32/64 arm StateQuery objects), keeping unchanged h0/h2 companions. This is the
corrected filter requested by independent_review in corrected multiseed gauge affine and next raw binding.

### 4. Fresh renaming evaluation (--fresh-rename)

After training, all eval participant names are bijectively mapped to fresh names
(`Bix`, `Caz`, `Dex`, ...) that never appeared in training or eval. If binding
is character-pattern-based and variable-like, performance should be preserved.

### 5. Identical controls

- bridge_sign connector on h0/h2 changed-state anchor training loss
- shared_trunk vs tied vs untied architecture comparison
- Paired initialization (all modes start from identical weights per seed)
- Graph evidence (held-held comparison rows)
- Dropout=0 for deterministic comparison
- Per-module gradient clipping at 5.0
- AdamW lr=3e-3, weight_decay=1e-4
- 220 epochs
- All eval suites: paired_state_conservation, heldheld_unseen_edge_closure,
  mixed_held_seen_orientation, cross_template_state_readout

## Expected outcomes

### If the mechanism survives
- shared_trunk/tied: train fit 1.0, h1/h3 graph-transfer same-initial state choice
  follows bridge_sign, held-held closure invariant, mixed orientation reverses
- untied: local fit but no coherent graph transport
- Fresh rename: metrics preserved (binding is variable-based)

This would show that learned candidate binding can support the same gauge-transport
mechanism, removing the strongest supplied-interface objection.

### If binding fails (train fit < 1.0)
- The GRU cannot learn character-pattern matching with this architecture
- Distinguish: does comparison learning also fail, or only state updating?
- Consider adding attention or a matching module
- The failure localizes to binding, not to the transport mechanism itself

### If binding succeeds but transport fails
- The model learns to match names but the shared coordinate does not transport
- This would suggest the `<cand>/<other>` replacement provides more than just
  addressing: it may regularize the representation by removing identity information

## GPU experiments launched

1. **Primary** (GPU0): full graph + anchors, 3 conditions × 2 signs
   → `data/primary_raw_name/`

2. **Comparison-only** (GPU1): narrow filter (no changed h0/h2),
   3 conditions × 2 signs
   → `data/comparison_only_raw_name/`

## Substrate

Same as causal gauge experimental design/291: `data/information_budget_substrate/replace_k16_spread`

## Script

`training/scripts/raw_name_binding_probe.py`

## Decision tree

| Primary tied/shared_trunk | Comparison-only | Interpretation |
|---|---|---|
| Transport + binding | No transport | Gauge mechanism survives removal of <cand>/<other>; bridge anchors still required |
| Transport + binding | Transport | Anchors not needed (unexpected, examine carefully) |
| Binding but no transport | — | Shared coordinate doesn't transport through char-based binding |
| No binding | — | GRU inadequate for char-pattern matching at this scale |
