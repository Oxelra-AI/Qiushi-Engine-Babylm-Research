# shared gauge review and next causal controls — Review of shared factorization result synthesis shared-gauge evidence and next causal controls

## Scientific question

The shared factorization result synthesis result is the first positive controlled mechanism after the post-SOTA failures: a tied candidate-event scorer transported held h1/h3 state orientation from h0/h2 anchors through held-held comparison evidence, while an equal-or-greater-capacity untied model fit the same observed rows without transporting the state coordinate. The current reviewer question is sharper: does this reflect transport of an **absolute shared gauge** rather than generic parameter sharing?

## Files read and produced

Read/checked:
- shared factorization result synthesis merged result: `data/shared_coordinate_all_merged_analysis/shared_coordinate_merged_analysis_summary.md/json`.
- Heldheld-only anchor control: `data/shared_coordinate_trainvocab_heldheld_anchor_control/shared_coordinate_summary.md`, merged into `data/shared_coordinate_heldheld_anchor_analysis/`.
- Input-access check: `data/factorized_probe_input_audit/input_audit_summary.md`.
- shared factorization result synthesis script: `training/scripts/shared_relation_coordinate_probe.py`.
- shared factorization result synthesis independent_review verifier: .

Produced:
- CPU saved-output review: `scripts/gauge_transport_review_analysis.py` -> `data/gauge_transport_review_analysis/gauge_transport_review_analysis_summary.md/json`.
- CPU training-diff/event-coordinate audit: `scripts/training_diff_and_event_coordinate_audit.py` -> `data/training_diff_and_event_coordinate_audit/training_diff_and_event_coordinate_audit_summary.md/json`.
- independent_review design/check memos: , .

## What survives review

### 1. Saved prediction integrity is clean for the train-only replication

`gauge_transport_review_analysis.py` found, for the eight train-only runs it read:
- `parse_errors=0` in every run;
- `eval_state_groups=768` in every run;
- no groups with bad length, no non-one-hot true/arm labels, and no score ties;
- train state and comparison fit are 1.000 in the four aligned/inverted seed28801 tied/untied runs.

This makes a merge-table or missing-row explanation unlikely for the shared factorization result synthesis train-only result.

### 2. Aligned/inverted arm readouts show a real signed response in tied models

For train-only seed28801, paired by evaluation row:

- Tied, true target, paired-state graph-transfer same-initial changed rows: aligned correctness 1.000, inverted correctness 0.000; every one of 64 paired margins changed sign. Mean margins were +16.708 aligned and −14.754 inverted.
- Tied, arm target: aligned and inverted are both correct, with positive margins in both arms (+16.708 and +14.754). This is the expected installed-coordinate behavior.
- Mixed held-seen comparison rows show the same pattern: under true target, tied aligned is correct and tied inverted is wrong with 512/512 opposite-signed margins; under each arm target both are correct.
- Untied models do not produce the same state coupling: graph-transfer same-initial changed-state correctness is only 0.250 aligned and 0.500 inverted under the true target, despite direct anchors and held-held comparison closure fitting.

This supports a genuine orientation variable in the tied pathway.

### 3. Raw event-coordinate signs show subset-gauge behavior, not a global flip

`training_diff_and_event_coordinate_audit.py` reconstructed raw candidate-event score differences `d = score(candidate0) - score(candidate1)` from saved comparison predictions for the aligned vs inverted train-only seed28801 runs.

For the tied model:
- held direct-anchor event coordinates reverse between aligned and inverted arms (`opposite_sign=1.000`, correlation with negative coordinate about 0.998);
- held graph-transfer event coordinates also reverse (`opposite_sign=1.000`, correlation with negative coordinate about 0.997);
- seen-reference coordinates in mixed held-seen comparisons remain same-signed (`same_sign=1.000`, correlation about 0.99998).

For the untied comparison branch:
- held and seen event coordinates remain same-signed between aligned and inverted arms (`same_sign=1.000`) because the comparison rows are identical and state-anchor gradients cannot reach the comparison scorer.

This is a stronger reading than aggregate accuracy: tied state-anchor changes alter the held component coordinate while preserving seen-reference orientation, so the effect is not just a whole-network sign flip.

### 4. Heldheld-only does not set a correct absolute state gauge

The heldheld-only train-only control uses the same 192 held-held comparison rows and no held direct state anchors. It still includes common seen state rows, but there are no h0/h2 bridge state rows. Its summary:
- tied seed28801: train state/comparison 1.000, held-held closure 1.000, but direct_same=0.000, graph_same=0.000, mixed_acc=0.000, pair_both_graph_same=0.000;
- tied seed28802: train_state=1.000 but train_cmp=0.875, graph_same=0.250, mixed_acc=0.125;
- untied heldheld-only also does not produce a correct absolute held-state coordinate.

Thus relative held-held comparison closure alone is not enough to pick the absolute state orientation. Bridge state anchors are necessary in the current harness.

## Important limits that remain

The supported statement is narrow:

> With participant candidates, binary ownership, same-owner comparison, and changed-fact routing supplied by the harness, a shared candidate-event scoring pathway lets direct state anchors choose a held-event gauge that comparison connectivity can transport to held h1/h3 state predictions. Separate pathways fit local evidence but need not align their state and comparison gauges.

It is not yet evidence for full natural role induction because:
- candidate discovery and candidate/other marking are supplied by a regex/parser;
- cross-event candidate correspondence is supplied;
- the two-candidate ownership simplex and same-owner comparison operator are hard-coded;
- changed versus unchanged routing is supplied by `event_object`/`parse_hypothesis`;
- unchanged facts use a separate static scorer, so pair-both currently duplicates changed-state accuracy rather than proving a unified updater;
- changed-state scoring ignores the initial-state premise, so the anti-copy/copy-initial alternatives that mattered in Steps281–284 are outside this restricted hypothesis class.

There is also one nuance in the aligned/inverted comparison: the arm-specific held-held comparison rows are identical, but the bridge-state rows are separate arm inventories, not simple row-id label flips. For the changed-state event scorer this still operates as an absolute-anchor reversal because only event text and target index enter the changed branch. But the next test should be cleaner: keep the data fixed and impose a controlled connector/permutation in the state-anchor loss.

## Next decisive causal experiment

The next run should make the transported gauge causally visible by a **fixed-data state-anchor connector** and a **graph-evidence × state-anchor factorial**, before any parser-free natural-language attempt.

### A. State-anchor connector/permutation on fixed data

Use one fixed arm, preferably `aligned_state_bridge`, with train-only vocabulary. Add a parameter `bridge_sign ∈ {+1, -1}` in the runner. For direct held changed-state training rows only (`h0_dax`, `h2_norp`):

```python
scores = state_scores(model, vocab, q, device)
if bridge_sign == -1 and q.is_changed and q.relation in {"h0_dax", "h2_norp"} and q.suite contains sparse_state_bridge:
    scores = scores.flip(0)
loss = cross_entropy(scores, target)
```

Do not apply this connector to:
- relation-comparison rows;
- common seen state rows;
- held h1/h3 evaluation;
- unchanged/static rows.

Predicted fingerprint for the tied model under connected graph + anchors:
- local post-connector direct-anchor fit remains high for `bridge_sign=+1` and `-1`;
- unpermuted canonical h1/h3 state margins reverse when `bridge_sign` flips;
- mixed held-seen comparison products reverse because held coordinates flip while seen-reference coordinates remain stable;
- held-held comparison products remain stable because all held events in the connected component flip together.

This is stronger than the current aligned/inverted arm comparison because raw rows/labels need not change; only the interface convention at direct anchors changes.

### B. Graph-evidence × state-anchor factorial

Run, at minimum, these fixed-inventory cells under `bridge_sign=+1/-1`:

| graph evidence | held direct state anchors | purpose |
|---|---|---|
| present | present | current positive cell; should show reversible h1/h3 transport in tied |
| present | absent | already partly represented by heldheld-only; should have high relative closure but no reliable absolute held-state orientation and no bridge-sign effect |
| absent | present | bridge-only cell; tests whether h0/h2 state anchors alone teach a surface rule that directly generalizes to h1/h3 without graph transport |
| absent | absent | baseline for default/surface orientation |

A sharper version uses a degree-matched cut graph instead of simply deleting comparisons. Current k16 comparison edges are:
- h0_dax–h1_mep: 48 rows
- h0_dax–h2_norp: 48 rows
- h1_mep–h2_norp: 48 rows
- h2_norp–h3_ziv: 48 rows

For a graph-cut control, replace anchor-to-held paths with matched within-component comparisons, e.g. anchored component h0–h2 and unanchored component h1–h3, preserving row counts, voice/template balance, same/different label proportions, and lexical exposure. A tied model should propagate bridge-sign only into the anchor-connected component, not into the disconnected h1/h3 component.

### C. Model variants needed to separate gauge identity from generic sharing

Run paired-initialized variants:
1. fully tied event scorer (current positive mechanism);
2. shared encoder/trunk with separate comparison and state scalar heads (generic representation sharing but independent scalar gauges);
3. fully untied scorers (current negative control).

Use cloned initial weights, dropout 0 or deterministic paired dropout, and no global cross-branch gradient clipping (or clip modules separately). Current shared factorization result synthesis did not guarantee paired initialization and used global norm clipping, which is acceptable for the first result but weaker for the causal test.

### D. Required readouts

Save raw event candidate differences `d_e`, not just clamped comparison logits. Report:
- train fit by objective family;
- post-connector direct-anchor margins;
- unpermuted canonical state margins for h0/h2 and h1/h3;
- mixed held-seen product signs and seen-reference coordinate stability;
- held-held product invariance;
- relation × voice × template splits;
- row-paired `bridge_sign=+1` vs `-1` opposite-sign fractions and correlations;
- no parse omissions, exactly two candidates per state query, one-hot labels, and tie counts.

## Research consequence

shared factorization result synthesis now survives a serious review as a real controlled shared-gauge effect inside a supplied candidate-event harness. The work is still far from a final general data-efficient learning principle: it has not yet shown parser-free role discovery, unified state conservation, architecture-general transfer, or BabyLM-scale benefit. The next scientific pressure should be the fixed-data connector plus graph×anchor factorial. If this succeeds, the candidate principle becomes much sharper: finite evidence becomes reusable when sparse absolute anchors and relative comparisons are forced through a shared computational gauge connected to the target behavior. If it fails, shared factorization result synthesis should be reframed as broad pathway sharing or an artifact of the current arm inventories rather than causal gauge transport.
