# invariance probe design: Identity-Under-Variation Representation Probe Design

## Background

If V-C benchmark movement replicates across seeds while VIEW is a reproducibly
worse LM, conversion lives in representation, not in row-level fitting.  The
mechanism predicted by the representation_and_objectives gauge-transport work is specific: VIEW differs
from REPEAT in exactly one respect — it presents the same content in distinct
surface forms.  Exact repetition provides no information about what is invariant.
VIEW's paraphrased renderings provide identity-under-variation evidence.

## Testable prediction (pre-stated)

In VIEW-trained models, representations of paired renderings of the same content
should be more invariant (higher cosine similarity, or more linearly aligned) than
in CLEAN or REPEAT models.  The gain should:
  1. Agree across seeds (beyond a permutation null)
  2. Predict benchmark movement on tasks needing surface-invariant knowledge
     (EWoK, COMPS, entity tracking) rather than form-sensitive BLiMP
  3. Be absent in REPEAT arms (which see the same surface form multiple times)

BLiMP being the one seed-stable V-C result (near zero in seed43022) is consistent:
BLiMP tests syntactic acceptability, which depends on surface form rather than
invariance to it.

## Data for the probe

The matched-rowholdout data pools contain paired text:
- `compact_view_dose2p64x_100M.jsonl` — view versions of training rows
- `cleanqwen_lengthmatched_dose2p64x_100M.jsonl` — clean versions of training rows
- `compact_view_dose2p64x_changed_block_rows_meta.jsonl` — metadata for rows
  where VIEW and CLEAN actually differ (the "changed block")

For each changed-block row, both versions (clean and view) exist.  These are the
identity-under-variation pairs.

Additionally: `heldout_cleanqwen_rows.jsonl` contains 6,992 held-out rows.
If view versions of these exist in the full compact_view pool, they provide the
cleanest test (neither model trained on them in either form).

## Measurement protocol

1. Select ~500 changed-block rows (or 500 heldout rows with view pairs)
2. Tokenize both clean and view versions using the shared tokenizer
3. Forward-pass through these checkpoints at 100M:
   - D_V_43022, D_V_43122, D_C_43022, D_C_43122, D_R_43122
4. Extract mean-pooled hidden states from the final transformer layer
5. Compute per-row cosine similarity: sim(rep_clean, rep_view)
6. Aggregate: mean similarity per arm

## Controls

- **Random-pair baseline**: For each row, pair clean(row_i) with view(row_j≠i).
  This should show LOW similarity in all arms.
- **Same-text ceiling**: Pass clean(row_i) twice.  Should show similarity ≈ 1.0
  (or exactly 1.0 with deterministic inference).
- **Cross-seed replication**: VIEW_43022 - CLEAN_43022 gap should match
  VIEW_43122 - CLEAN_43122 gap in sign and approximate magnitude.

## Pre-stated null

- If VIEW similarity = CLEAN similarity for clean-view pairs, VIEW did not learn
  surface invariance; conversion operates through a different mechanism.
- If VIEW > CLEAN for clean-view pairs BUT also VIEW > CLEAN for random pairs,
  the effect is a general representation-geometry artifact, not specific invariance.
- If VIEW > CLEAN specifically for clean-view pairs AND this correlates with
  benchmark columns needing invariance but not BLiMP, the identity-under-variation
  mechanism is supported.

## Confound control

VIEW model saw view versions during training; CLEAN model saw clean versions.
The metric is CROSS-FORM SIMILARITY (cosine between clean rep and view rep),
not absolute activation.  VIEW's advantage, if it exists, is that training on
diverse surface forms of the same content taught the model that meaning is
invariant to surface change.  CLEAN never had this pairing signal.

For the strongest confound control, use held-out rows (never trained on by either
model in either form).  If held-out view versions are unavailable, the changed-block
test remains valid because the metric is similarity, not recognition.

## Implementation estimate

- Forward pass: 500 rows × 2 versions × 5 arms × 1 checkpoint = 5,000 forward passes
- Each pass: ~0.01s on H100 → total ~50 seconds
- Total with loading: ~5 minutes
- Depends on: V-C replication result (invariance probe design evaluation)

## Files to produce

- `scripts/invariance_probe.py`
- `data/invariance_probe/` — per-arm similarity arrays
- `notes/005_invariance_probe_result.md`

## Connection to the principle

If both V-C replication and the invariance probe survive:
  Under a fixed budget, admitted experience is valuable in proportion to the
  identity-under-variation evidence it supplies about content the learner must
  reuse, given a shared representation to carry it.

This unifies representation_and_objectives (gauge transport requires identity evidence) with frontier_consolidation
(fixed-budget substitution where the value depends on conversion) into a single
testable, measurable principle.
