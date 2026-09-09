# posalign matcher design and capacity — Position-aligned matcher: architectural inductive bias for entity binding

## Scientific question

raw span identity routing result showed that the CharGRU dot-product matcher can discriminate between
different names at the same position (eval relative accuracy 0.97) but cannot
reliably detect which tokens ARE entity names among all tokens (eval hard-assignment
accuracy ~0.5). The posalign matcher design and capacity capacity pilot confirmed this is not a capacity issue:
char_hidden 16/32/64 all plateau at ~0.5 eval hard-assignment despite train reaching
1.0. The CharGRU compresses character sequences into fixed vectors, conflating exact
string equality with approximate character similarity.

## Key finding: position-aligned comparison inherently generalizes

A `PositionAlignedMatcher` computes per-position character similarity using learned
embeddings, then takes the product over positions (soft AND). This has the correct
inductive bias for string equality:
- Same characters at each position → high per-position similarity → high product
- Any different character → low per-position similarity → low product  
- Different lengths → length penalty → low score

**Before any training, the position-aligned matcher achieves 1.000 hard-assignment
accuracy on both train (16 names) and eval (16 disjoint names).** This confirms
that the architectural inductive bias is the key factor, not the amount of equality
training.

## Experimental comparison

| Matcher | Architecture | Eval hard-assign | Eval relative | Entity detection | Entity identification |
|---|---|---|---|---|---|
| CharGRU-16 | GRU→fixed vector→dot product | 0.50 | 0.91 | ✗ | ✓ |
| CharGRU-32 | GRU→fixed vector→dot product | 0.56 | 1.00 | ✗ | ✓ |
| CharGRU-64 | GRU→fixed vector→dot product | 0.53 | 0.94 | ✗ | ✓ |
| PosAlign-16 | per-position emb→dot→product | 1.00 | 1.00 | ✓ | ✓ |
| Hard oracle | exact string match | 1.00 | 1.00 | ✓ | ✓ |

## What this establishes

Entity binding from raw text requires two distinct capabilities:
1. **Entity detection**: finding entity tokens among all tokens (the "needle")
2. **Entity identification**: matching detected entities to their query roles

The CharGRU's compression bottleneck prevents entity detection generalization because
its continuous embedding space conflates exact equality with approximate similarity.
The position-aligned matcher maintains the full character sequence structure, enabling
exact equality detection for unseen strings.

This adds a fourth condition to the gauge-transport principle:
> Shared representation + anchors + connected graph + **structurally correct binding**
> → absolute gauge transport

"Structurally correct" means the binding interface must maintain enough positional
structure to distinguish exact entity matches from approximate character similarity.
Compression-based representations (GRU, attention pooling) may learn inter-name
discrimination but fail entity detection without structural comparison inductive biases.

## GPU experiment status

Shared-trunk bs+1 and bs-1 are running on H100s with:
- Position-aligned matcher (frozen, 0 pretraining epochs)
- No-name vocabulary (vocab 60)
- 150 relational training epochs
- Same substrate, seeds, and evaluation as causal gauge experimental design

If the gauge fingerprint is restored (graph_same 1.0/0.0, unchanged 1.0, hh_closure
1.0, row-paired sign reversal 1.0), this completes the naturalization chain:
supplied `<cand>/<other>` → SpanMatcher at supplied positions → position-aligned
matching on raw text.

## Files

- CharGRU capacity pilot: `scripts/charenc_capacity_pilot.py`,
  `data/charenc_capacity_pilot/`
- Position-aligned probe: `training/scripts/posalign_gauge_probe.py`
- Hard-assignment probe (initial attempt): `training/scripts/hard_assignment_gauge_probe.py`
- CPU smoke: `data/posalign_cpu_smoke/`
- GPU results (pending): `data/posalign_shared_bsplus/`,
  `data/posalign_shared_bsminus/`
