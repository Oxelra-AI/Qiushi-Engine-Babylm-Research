# Topology Experiment Design v2: Corrected After independent_review + relation_learning Review

## Critical corrections from review

### 1. DETACH interpretation (independent_review verifier)
earlier analysis established that comparison-only runs are BIT-IDENTICAL across bridge signs
+1 and -1. Held-held comparison targets are gauge-invariant. Therefore:
- Under DETACH, the trunk receives gradients ONLY from comparison loss
- Comparison loss is IDENTICAL for bs+1 and bs-1
- **The trunk should be IDENTICAL between paired runs**
- Only the state head differs (it receives bridge-sign-dependent targets)

DETACH does NOT test "indirect anchor orientation through comparison loss."
DETACH tests: **can a state head, trained only on h0/h2 anchor labels,
generalize to h1/h3 using a comparison-trained trunk representation?**
This is a clean test of readout calibration (mechanism 3).

**Verification check:** trunk parameter hashes should be identical between
DETACH bs+1 and DETACH bs-1. If they differ, there's a bug.

### 2. Selective shuffle limitation (independent_review + relation_learning)
Random label shuffling gives 50% correct info on average, not zero.
Shuffled labels also inject gradient noise into the shared trunk, which
can actively damage representations for ALL relations, not just h1.
Therefore h1 failure under shuffle conflates "absence of correct info"
with "active representational damage."

### 3. Graph distance correction (independent_review)
h3 is distance 1 from h2 (an anchor), not distance 2 from the anchor SET.
The existing harness has both h0 and h2 as anchors, so all four relations
are within distance 1 of an anchor. The per-relation analysis confirmed
perfect transport for all relations.

---

## Scientific purpose (revised)

This experiment answers two questions:

**Q1 (DETACH):** Does the comparison-trained trunk representation support
state-readout calibration from anchors to unanchored relations?
If yes: finite absolute labels scale with coordinate components, not entities.
If no: state-loss gradients must shape the trunk for usable state readouts.

**Q2 (DISCONN variants):** Does correct relational content on comparison edges
determine orientation, or does mere exposure/shared parameters suffice?
This requires triangulation across multiple disconnection methods to separate
correct-info removal from gradient damage.

---

## Three mechanisms to separate

1. **Representation formation** — shared trunk + comparison loss organizes coordinates
2. **Graph-mediated orientation** — correct comparison labels propagate anchor sign
3. **Readout calibration** — state head calibrates from anchors to unanchored relations
   using a trunk representation that is itself bridge-sign-invariant

---

## Current harness data structure

### Training state queries (576 total)
- h0_dax: 32 direct anchors (bridge_sign flips changed rows)
- h2_norp: 32 direct anchors
- s_give: 256 common seen
- s_receive: 256 common seen
- h1_mep: 0, h3_ziv: 0

### Training comparison edges (192, balanced 96T/96F)
- h0↔h1: 48 (24T/24F)
- h0↔h2: 48 (24T/24F)
- h1↔h2: 48 (24T/24F)
- h2↔h3: 48 (24T/24F)

### Per-relation comparison exposure
h0: 96, h1: 96, h2: 144, h3: 48

### Graph structure
h0--h1, h0--h2, h1--h2, h2--h3 (all connected, diameter 2)
Both h0 and h2 are anchored; all relations within distance 1 of an anchor.

---

## Experimental conditions

### DETACH (primary, cleanest)

Full original graph, but state loss detached from shared trunk.

Implementation: in state scoring, `h = trunk.forward_gated(...)` then
`scores = head(h.detach())`. Comparison scoring uses normal `head(h)`.
Effect: trunk ← comparison loss only; state head ← state loss only.

Expected behavior:
- Trunk IDENTICAL between bs+1 and bs-1 (verified by parameter hash)
- State head DIFFERS between bs+1 and bs-1 (anchor labels differ)
- Question: can state head generalize anchor orientation to h1/h3?

Verification: compare trunk parameter hashes across paired runs.
State accuracy reported separately for {direct anchors, common seen, h1/h3 graph transfer}.

### DISCONN triple (exposure/info/damage separation)

All three variants target h1 by modifying h0↔h1 and h1↔h2 edges
while preserving h0↔h2 and h2↔h3 correct labels. h3 serves as the
correctly-connected control.

**DISCONN-ADVERSARIAL:** Invert h0↔h1 and h1↔h2 labels (True→False, False→True).
- h1 gets maximally WRONG relational info
- Prediction if graph info determines orientation: h1 should have INVERTED
  orientation (anti-transport), not chance
- Expected row-paired signature: h1 signs still flip with bridge_sign,
  but in the OPPOSITE direction from FULL

**DISCONN-SHUFFLE:** Randomly permute h0↔h1 and h1↔h2 labels.
- h1 gets noisy but not zero relational info (50% correct on average)
- Multiple shuffle seeds to estimate sensitivity
- Prediction: h1 at or near chance if noise dominates; partial transport
  if 50% correct info is sufficient

**DISCONN-DELETE:** Remove h0↔h1 and h1↔h2 rows entirely (96 rows removed).
- h1 gets ZERO exposure (same as earlier analysis no_cmp for h1 specifically,
  but h2↔h3 and h0↔h2 remain, so h3 should still follow)
- Total comparison rows: 96 (not 192)
- Prediction: h1 at chance (no exposure); h3 should follow (connected to h2)

### Triangulation logic

| Condition | h1 exposure | h1 graph info | h1 gradient effect | Predicted h1 |
|---|---|---|---|---|
| FULL | 96 rows | Correct | Normal | Follows bridge_sign |
| ADVERSARIAL | 96 rows | Inverted | Maximally wrong | Anti-follows or damaged |
| SHUFFLE | 96 rows | ~50% correct | Noisy | Chance or partial |
| DELETE | 0 rows | None | None | Chance |

**If ADVERSARIAL gives anti-transport AND DELETE gives chance:**
→ Correct label CONTENT determines orientation (not just exposure)
→ ADVERSARIAL doesn't merely damage; it actively (mis)orients

**If ADVERSARIAL and DELETE both give chance:**
→ Either: labels are needed AND noise damage is negligible
→ Or: labels are needed but adversarial also damages (need to check h3)

**If h3 fails in ADVERSARIAL but not DELETE:**
→ Adversarial on h1 edges damages shared trunk (affects h3 through parameters)
→ DELETE is cleaner because it doesn't inject noise

**If h3 succeeds in ALL variants:**
→ h1-specific edge modifications don't damage the broader trunk
→ h1-specific results are interpretable

---

## Combined run plan

### Phase 1: decisive pair (4 GPU runs, estimated ~30 min on 2 GPUs)
DETACH at bridge signs +1 and -1.
DISCONN-DELETE at bridge signs +1 and -1.

DETACH: tests readout calibration.
DELETE: tests whether h1 needs any exposure while h3 control stays connected.

### Phase 2 — Triangulation (4-6 more runs):
DISCONN-ADVERSARIAL bs+1, bs-1
DISCONN-SHUFFLE bs+1, bs-1 (×1-3 shuffle seeds)
FULL control bs+1, bs-1 (reproduction with seed 30000)

### Post-hoc on ALL conditions:
- Per-relation h1 vs h3 row-paired sign analysis
- Trunk parameter hash comparison (especially for DETACH pair)
- Frozen post-hoc affine calibration from h0/h2 to h1/h3
- Comparison accuracy and per-edge loss/gradient diagnostics

---

## Implementation notes

### DETACH implementation (in training loop)
```python
# For changed state queries through model.event_state:
if detach_trunk:
    h, probs = model.event_state.trunk.forward_gated(ids_t, mask, tcb, cc, oc, isp)
    scores = model.event_state.head(h.detach())  # ← key change
else:
    scores, probs = model.event_state.score_gated(ids_t, mask, tcb, cc, oc, isp)
# Comparison scoring always uses normal (non-detached) path
```

For static queries through model.static (separate trunk): unchanged, since
model.static has its own trunk that doesn't share parameters with event scorers.

Verification: after first backward pass, check that all shared trunk parameters
have zero grad contribution from state loss.

### ADVERSARIAL implementation
```python
def invert_edge_labels(tc, target_edge_types):
    """Invert labels on specified edge types."""
    result = []
    for c in tc:
        if (c.relation1, c.relation2) in target_edge_types:
            c_new = copy.copy(c)
            c_new.label = not c.label
            result.append(c_new)
        else:
            result.append(c)
    return result

H1_EDGES = {('h0_dax', 'h1_mep'), ('h1_mep', 'h2_norp')}
tc_adversarial = invert_edge_labels(tc, H1_EDGES)
```

### DELETE implementation
```python
def delete_edges(tc, target_edge_types):
    return [c for c in tc if (c.relation1, c.relation2) not in target_edge_types]

tc_delete = delete_edges(tc, H1_EDGES)  # 96 rows remaining
```

---

## Relation to broader data-efficient learning principle

### DETACH implications
If readout calibration works: the trunk learns reusable relational coordinates
from comparison structure alone, and absolute orientation is a lightweight
calibration problem. The number of absolute labels scales with coordinate
components (disconnected components), not with entities or relations.

### DISCONN implications
If correct relational labels determine orientation: finite experience value
depends on adding CORRECT relational constraints, not just exposure. An
experience that adds wrong or noisy constraints has negative or zero value.
This directly informs the fixed-budget substitution framework: admitted material
is valuable when it carries correct learnable constraints, not when it
merely increases exposure.

### Seed×data correction
The balanced seed-by-data analysis supersedes the initial correlation interpretation. The low cross-seed loss-change correlation and high absolute-loss correlation are correlation observations, not a measured fraction of private learning or proof of identical competence. In a balanced DeBERTa 2×2 seed×data design, same-data cross-seed and same-seed cross-data partial correlations are similar, and the independent VIEW-CLEAN heldout-row data-effect vector is weak across seeds. Therefore the fixed-budget substitution conversion term remains open: a replicated benchmark effect would point toward representation-level conversion not captured by this row-loss instrument, while a non-replicated benchmark effect would shrink the substitution evidence to seed-stable sub-benchmark facts. These loss correlations are not premises for the controlled topology experiment.
