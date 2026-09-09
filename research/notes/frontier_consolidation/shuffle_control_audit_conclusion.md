# shuffle control audit conclusion — shuffled-control semantics resolved

## Question
The repaired dual-view trainer's `shuffled` mode uses a **per-batch derangement** of
the realized AuxUnit source sequences (`source_assignment`, seed `43022 + 1000003*step`),
not the static precomputed `decoy_source_ids` stored in the pair-data file. It is necessary to quantify whether this control is contaminated by within-row /
within-document correspondence before the aligned-vs-shuffled 20M panel is read.

## Method
`scripts/dualview_shuffle_control_audit.py` (CPU) replays the same
batch order (batch 256, `shuffle=False` loader), the same derangement rule, and the
same seed over the first 20M main words. For every constituent pair it records whether
the assigned false source is the same pair, same packed row, or same source document,
and the source-word length mismatch. Approximation: it assumes all pairs in each pair
row fire (real WWM masking fires a subset), so it is an **upper bound** on within-row /
within-document collision opportunity per batch.

Output: `data/dualview_shuffle_control_audit/shuffle_control_audit.json`.

## Result (upper bound over 24,310 unit assignments, 506 batches)
- same_pair: 0.000%
- same_row (false source from the same packed 256-word row): **6.664%** (1,620)
- same_doc (false source from the same source document): **0.016%** (4)
- different_doc: **99.98%**
- source-word mismatch: mean abs 8.70, median 7, p90 18, p95 23, max 37

## Structural cause (decisive)
Independent count over the 12,155 legal compact pairs:
- 88.76% of pairs belong to documents that contribute multiple pairs (up to 17/doc),
  but those pairs are distributed across many different packed rows.
- Only **4 of 3,005** packed rows contain >=2 pairs from the same document.

Therefore pairs sharing a document almost never share a packed row or a batch, so a
within-batch derangement almost never draws a false source from the target's own
document. The 0.016% same_doc rate is a genuine property, not an artifact of the
"all pairs fire" approximation; the realized masked-subset control fires fewer units
per batch and can only reduce collision further.

## Conclusion
The `shuffled` arm is a clean **different-document source-correspondence control**,
consistent with the document-disjoint transfer that source free transfer synthesis validated. The
worst-case failure mode — a small/null aligned-vs-shuffled gap being misread as
refuting transfer when it is only a mislabeled within-document control — does NOT apply:
99.98% of false sources are from different documents.

The one residual imperfection is source-length mismatch (mean 8.7 words). This is not a
correspondence contamination: both arms charge identical total source words per batch
(exact multiset preservation), and the mismatch cannot manufacture a spurious aligned
advantage. If a later stricter contrast is wanted, the precomputed length-matched
different-document `decoy_source_ids` in the pair-data file remain available, but the
current realized control is already a valid different-document contrast and does not
require re-running the shuffled arm.

## Decision
Reading the aligned-vs-shuffled 20M panel as-is is scientifically valid. No re-run of
the shuffled arm is required on contamination grounds. The exact `mlm_only` no-aux
reference must still be run under the true charged-exposure definition (not the adapter matched horizon plan
approximation) before any 100M commitment.
