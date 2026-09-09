# sparse relation aux calibration — sparse relation auxiliary calibration decides against the two-branch launch

## What was tested

Before spending two 70M→80M continuations (semantic vs anchor_permuted), I calibrated the
exact sparse relation aux preflight contrast on the identical exact 70M→80M segment (init compact `chck_70M`,
tail SHA `19304f23...`, labels SHA `39ed2b8e...`, tokenizer SHA `e70d167f...`), performing no
optimizer update. Script: `scripts/sparse_relation_aux_calibration.py`.

Two runs, identical except event probability:
- `data/sparse_relation_aux_calibration/` (event_prob 0.35, as sparse relation aux preflight default)
- `data/sparse_relation_aux_calibration_eventprob1/` (event_prob 1.0)

Each scanned the full 250-batch / 64,000-row / 9,971,289-word segment for coverage, used 32
identical-WWM batches for pre-update margins, and 4 batches for MLM-vs-aux backbone gradients.

## Load-bearing findings (both runs agree)

1. **The paired target is already trivially recoverable from visible context in BOTH arms.**
   Raw hidden-state cosine success (true pivot / matched anchor picks its own dependent target
   over the structure-matched cross-event target):
   - event_prob 0.35: semantic 0.8836, anchor_permuted 0.8047 (delta 0.0789)
   - event_prob 1.00: semantic 0.8867, anchor_permuted 0.8278 (delta 0.0589)
   The untrained rank-64 head is near chance in both arms (~0.485–0.488), so the only
   distinguishing signal is a ~0.06–0.08 raw-cosine success gap. Per relation family the
   semantic−anchor raw-cosine delta is 0.02–0.11 and is not consistently positive on the
   trained-head margin (temporal +0.056, causal −0.052, negation −0.058, spatial −0.015).

2. **The auxiliary is not a masked-recovery task — targets/negatives are visible.**
   Under ordinary WWM, target and negative-target groups are fully visible original tokens in
   ~83% of events, pivots/anchors in ~88%. So the auxiliary mostly reinforces representation
   lookup the backbone already encodes, not conditional prediction of a hidden consequence.

3. **Microbatch-local structure matching retains only ~20–23% of selected events.**
   event_prob 0.35: 19,847 selected → 4,000 used (16.0/batch). event_prob 1.0: 24,000 → 5,505
   (22.0/batch). Raising event_prob does not fix retention; the bottleneck is same-microbatch
   structure-group availability, not selection probability. Comparative never survives (0 used
   in both runs); physical_change is ~2.5% of used events. Match level 4 (loosest) is ~51%.

4. **Backbone gradient pressure is small and, more importantly, nearly identical between arms.**
   Weighted aux/MLM model-gradient L2 ratio ~0.13–0.14 for both semantic and anchor_permuted
   (asymmetry well under 2×). The two arms exert essentially the same backbone pressure, so a
   downstream difference could not be attributed to gradient-scale asymmetry — but it also means
   the semantic arm carries almost no additional true-relation signal over the matched control.

## Decision

The two 20M-word continuations would NOT resolve the mechanism. A semantic-arm advantage, if
any, could not be attributed to true pivot–consequence structure: the target is already
recoverable from visible same-sentence context in the matched-anchor arm at 0.83, the added
semantic raw-cosine gap is ~0.06, the trained-head margins are near chance and not consistently
semantic-favoring per family, coverage of the physically grounded families is thin, and the two
arms apply near-identical backbone pressure. These results do not justify a paired H100
continuation, so this exact formulation is stopped.

## What this negative result teaches (evidence-continuous)

The relation supervision must operate where the consequence is NOT already visible. A contrastive
auxiliary over unmasked same-sentence spans reduces to representation lookup and cannot pressure
conditional world-relation formation. This points the next mechanism toward making the dependent
consequence a genuine prediction target under conditioning on the pivot — i.e. the consequence
group should be masked (removed from the visible context) while the pivot is the only cue, so the
model must produce the consequence from relation knowledge, not retrieve a co-present token. This
is distinct from dense PVDM (which shifted target mass and manipulated anchors and damaged relation
surfaces): it would keep ordinary WWM intact and add, sparsely, a masked-consequence-given-pivot
objective on legal pvdm compliance and control design events, with the matched control masking the same consequence but
conditioning on the matched nonrelation anchor. The sparse relation aux calibration calibration harness can be reused to
require, before any launch: (a) the consequence target is masked in both arms, (b) the pre-update
recovery gap between true-pivot and matched-anchor conditioning is non-trivial and semantic-favoring,
(c) family coverage including physical_change/comparative is adequate, (d) comparable backbone
pressure. Only then is a paired continuation worth GPU time.

## Files
- `scripts/sparse_relation_aux_calibration.py`
- `data/sparse_relation_aux_calibration/calibration_summary.json`
- `data/sparse_relation_aux_calibration_eventprob1/calibration_summary.json`
- `notes/sparse_relation_aux_calibration.md`, `sparse_relation_aux_calibration_eventprob1.md`
