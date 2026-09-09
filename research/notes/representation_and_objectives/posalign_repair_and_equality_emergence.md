# posalign repair and equality emergence — Position-sensitive equality: repaired connector and finite-emergence evidence

## Scientific Motivation was needed

posalign matcher design and capacity found a striking positive cell: a frozen position-aligned matcher on raw text with no name vocabulary gave perfect shared-trunk gauge transport for `bridge_sign=+1`. The negative-sign repair and untied control were still in progress. A repaired negative-sign run could not be causally paired with the old positive cell if seed or implementation changed; even a clean pair would show sufficiency of a supplied frozen equality operator rather than learned natural binding.

## Delivered posalign matcher design and capacity tasks

- `data/posalign_untied_bsplus/`: untied `bs+1`, seed29800, fitted local state/comparison and matching at 1.0 but did not transport the graph coordinate (`graph_same=0.25`, `mixed_acc=0.5`, `unchanged=1.0`, `hh_closure=1.0`). This keeps the shared-representation condition active under raw text with a position-sensitive matcher.
- `data/posalign_shared_bsminus_r2/`: shared-trunk `bs-1`, seed29801, did **not** fit locally (`train_state=0.694`, `train_cmp=0.875`, `graph_same=0.5`, `hh_closure=0.625`). It is not evidence against gauge transport.

## Repair found in posalign matcher design and capacity script

The negative-sign failure was not just an optimization issue. In `training/scripts/posalign_gauge_probe.py`, posalign matcher design and capacity trained `bridge_sign=-1` using:

```python
inv_t = q.inverted_target_by_name[q.names[1]]
actual = target if bridge_sign == 1 else (1 - target if inv_t else target)
```

This differs from the established causal gauge experimental design/296 connector, which flips only sparse direct-anchor changed-state rows:

```python
if bridge_sign == -1 and q.is_direct_anchor:
    scores = scores.flip(0)
```

The posalign matcher design and capacity script was repaired to this direct-anchor-only connector. A CPU smoke at `data/posalign_repaired_bridge_cpu_smoke/` syntax-checked and ran one epoch. The repaired `bs-1` smoke now starts with the same healthy train behavior as the old positive cell (`train_state=0.941`, `train_changed=0.944`) rather than the broken broad-flip behavior (`train_state≈0.694`). This justifies a small matched H100 rerun.

## Occurrence audit

I suspected posalign matcher design and capacity's hard assignment might be wrong because it canonicalizes one argmax occurrence per query rather than every occurrence. `data/posalign_occurrence_audit/occurrence_audit.md` refuted that explanation for the current substrate: every train/eval state and comparison event has exactly one occurrence of each name. Therefore the major bug was the target connector, not repeated-token canonicalization. The occurrence distinction remains important for later more natural substrates.

## Finite emergence of the equality operator

`data/equality_operator_emergence_pilot/` and `data/equality_operator_breakdown/` test the equality route upstream of relational training without GPUs.

Main table from `equality_operator_emergence.md`:

| model | train events | official held | fresh known chars | fresh unseen chars | near known names | held rel-name | held true-maxnonname |
|---|---:|---:|---:|---:|---:|---:|---:|
| shared_pos_initial | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.9107 |
| dual_pos_initial | 0.485 | 0.341 | 0.312 | 0.312 | 0.188 | 0.590 | -0.0171 |
| dual_pos_name_level_train | 1.000 | 0.818 | 1.000 | 0.562 | 1.000 | 0.872 | 0.4963 |
| dual_pos_train_alphabet_char_pairs | 1.000 | 0.903 | 1.000 | 0.625 | 1.000 | 1.000 | 0.5656 |
| dual_pos_full_alphabet_char_pairs | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.7101 |

The official held names contain letters absent from train names (`b,c,g,u,y,z`). The breakdown shows:

- name-level finite evidence solves held names composed only of observed train letters: official held seen-letter-only accuracy 1.0, but candidates with unseen letters 0.676.
- character-pair evidence over the train alphabet also solves seen-letter-only names and improves official held to 0.903, but still fails some unseen-letter candidates.
- finite character-pair evidence over the full alphabet gives exact held assignment 1.0 across official held, fresh known-character, fresh unseen-character, and near-name tests.

This clarifies the equality component: position-wise structure is necessary, but a same-symbol coordinate must either be supplied by shared character embeddings or learned from finite lower-level equality evidence covering the primitive alphabet. Name-level relational exposure can induce a partial equality route but does not guarantee out-of-support primitive characters.

## Prepared next relational script

`training/scripts/dualpos_learned_equality_gauge_probe.py` removes the zero-shot same-symbol prior by using separate token-side and query-side character embeddings, then pretrains them on finite character-pair equality evidence before relational training. CPU smoke at `data/dualpos_cpu_smoke/` confirms it runs and that with only 2 character-pair epochs matching remains low, so the script is not accidentally using a hidden shared-equality prior.

## H100 work launched in posalign repair and equality emergence

After the connector repair and CPU smoke, two minimal matched repaired cells were launched:

- `data/posalign_repaired_shared_bsplus_seed29920/`, shared_trunk `bs+1`, repaired script, seed29920, 180 epochs, lr 5e-3, GPU0.
- `data/posalign_repaired_shared_bsminus_seed29920/`, shared_trunk `bs-1`, repaired script, seed29920, 180 epochs, lr 5e-3, GPU1.

These decide whether frozen position-sensitive equality is sufficient for raw-text/no-name gauge transport under the correct sparse direct-anchor sign intervention. Because both signs use the same repaired code and seed with fresh untouched initialization, row-paired sign reversal is interpretable if both cells fit state and comparison locally.

## Interpretation Before Further Results

The current supported principle is sharper but still bounded:

1. Shared relational computation, sparse absolute anchors, and connected comparison constraints can transport a coordinate gauge in the supplied candidate harness.
2. Raw text can feed this mechanism when entity tokens are canonicalized into candidate/other coordinates.
3. A frozen position-sensitive equality operator with same-character prior is sufficient for this canonicalization in the present marker-free synthetic strings.
4. Finite lower-level equality evidence can create the same kind of operator when it covers the primitive character alphabet; finite name-level exposure is partial and bounded by primitive coverage.
5. The evidence still does not show broad natural binding, parser-free coreference, learned semantic role induction, or BabyLM-scale benefit.

The next scientific transition is to make equality computation emerge from finite experience while candidate ambiguity and lexical alternatives remain available, then test whether shared relational representation still transports the gauge. The dual-position learned-equality script is prepared for this, but the currently running repaired frozen-equality sign pair should be collected and analyzed first.

## New result: finite full-alphabet equality evidence feeds gauge transport

The dual-position learned-equality sign pair completed after the initial synthesis. It used separate token-side and query-side character embeddings, so the zero-shot same-symbol prior was removed. Before character-pair evidence, both train and eval matching were 0.0. Full-alphabet finite character-pair equality evidence (`a`–`z`, 60 epochs) installed a reusable equality coordinate: train/eval matching became 1.0 for both signs, with identical pretraining losses and init hash `dfe5516bc0f5738a`.

Relational training then recovered the full shared-trunk sign intervention under raw text/no-name vocabulary:

- `data/dualpos_fullalpha_shared_bsplus_seed29930/`: shared_trunk `bs+1`, train_state=1.0, train_cmp=1.0, graph_same=1.0, pair_both_graph_same=1.0, unchanged=1.0, held-held closure=1.0, mixed=1.0.
- `data/dualpos_fullalpha_shared_bsminus_seed29930/`: shared_trunk `bs-1`, train_state=1.0, train_cmp=1.0, graph_same=0.0, pair_both_graph_same=0.0, unchanged=1.0, held-held closure=1.0, mixed=0.0.
- `data/dualpos_fullalpha_pair_analysis/`: same init hash true, both match-after true, both train-fit true; row-paired direct-anchor opposition 192/192, graph-transfer opposition 192/192, unchanged same-sign 384/384, held-held comparison closure 1.0 for both signs.

This moves the evidence beyond a frozen equality prior: finite lower-level equality evidence can create the reusable identity coordinate that raw-text gauge transport requires. It is still a controlled synthetic result because the character-pair task explicitly supplies equality over the primitive alphabet, and hard assignment/frozen matching still protect the equality route during relational learning. It does not yet show that ordinary relational training discovers equality in the presence of lexical alternatives.

## Pending control

`s299_t41_tool2` is the matched dual-position/full-alphabet **untied** `bs+1` control. It tests whether finite-emergent equality plus local state/comparison fit is enough by itself. The expected mechanistic contrast is: equality matching should be 1.0 and local train tasks should fit, but graph transfer should fail without a shared relational representation. If it unexpectedly transports the graph coordinate, the downstream shared-representation condition would need revision.

## Completed control: finite equality without shared relational representation

The dual-position/full-alphabet untied `bs+1` cell completed at `data/dualpos_fullalpha_untied_bsplus_seed29930/`. It confirms equality formation but not local comparison learning in that branch:

- before character-pair evidence: train/eval matching 0.0;
- after full-alphabet character-pair evidence: train/eval matching 1.0;
- relational train state: 1.0;
- relational train comparison: only 0.552 after 180 epochs;
- eval graph_same 0.5, unchanged 1.0, held-held closure 0.656, mixed 0.531.

`data/dualpos_untied_failure_audit/` shows the comparison branch stayed essentially at the symmetric point: train comparison `prob_same≈0.5`, near-half fraction 1.0, mean absolute logit about `2.3e-7`, and event comparison `|d_e|` around `1e-3`. This means the dual-position untied run is not a local-fit-without-transport contrast. Instead it exposes another meaningful boundary: when the comparison scorer is disconnected from state anchors, the comparison objective can stay in a symmetric zero-coordinate solution even with perfect equality matching. The frozen posalign matcher design and capacity untied cell remains the clean raw-text local-fit contrast for the supplied-equality setting; for the finite-learned-equality setting, a future untied comparison would need a symmetry-breaking initialization or a direct state-seeded comparison branch if a pure local-fit contrast is required.

## posalign repair and equality emergence strongest result after all runs

The step established two connected facts in the current controlled substrate:

1. **Frozen position-sensitive equality is sufficient for marker-free raw-string input.** With the corrected sparse direct-anchor connector and matched seed29920, shared-trunk `bs+1/bs-1` cells have identical init hash `89c5d0c1e435f56d`, train state/comparison 1.0, row-paired direct-anchor opposition 192/192, graph-transfer opposition 192/192, unchanged same-sign 384/384, and held-held closure 1.0 for both signs.
2. **Finite primitive equality evidence can create that identity coordinate.** With separate token/query character embeddings and no initial same-symbol prior, full-alphabet character-pair evidence moves matching from 0.0/0.0 to 1.0/1.0 and the shared-trunk relational sign pair with matched init `dfe5516bc0f5738a` recovers the same gauge transport fingerprint: direct and graph sign opposition 1.0, unchanged stability 1.0, and held-held closure 1.0.

The scientific boundary remains clear: this is not ordinary language discovering reference. It is a controlled demonstration that reusable behavior can be built by composing a lower-level learned equality coordinate with a shared higher-level relational coordinate. The next transition should test whether this route survives when lexical name channels and trainable soft routing are available, and whether equality can be protected early enough to avoid being bypassed by identity-specific shortcuts.

## Independent verification notes from independent_review

independent_review verifier integration  agrees that posalign repair and equality emergence supports a narrow two-stage mechanism: finite primitive character-pair supervision can install a reusable string-identity kernel, and once hard identity routing feeds a shared event representation, sparse anchor orientation controls unanchored event decisions while held-held relative structure remains stable.

The verifier added important scientific cautions:

- The graph-edge mechanism is behaviorally consistent but not yet causally isolated in the raw-text learned-equality setting. A shared trunk may still exploit relation/template regularities unless comparison edges are removed, randomized, rewired, split into components, or varied by anchor distance.
- The learned-equality untied run is not a clean nonsharing contrast because its comparison scorer stayed at a symmetric point (`p_same≈0.5`) rather than fitting comparison rows. It shows a comparison-branch symmetry problem, not merely absence of graph transport.
- The current logit recording sets saturated probabilities to logit 0, so margin summaries based on `logit_same` are unreliable at saturation; accuracy and row-paired sign results remain usable, but future scripts should clamp probabilities before taking logits.
- The current task still receives candidate names, tokenized strings, single mentions, hard assignment, and frozen equality after pretraining; it excludes candidate discovery, repeated mentions, aliases, pronouns, ambiguous spans, and ordinary language reference.

The most informative next work is therefore not more seeds of the same positive cell. It is to intervene on the graph/comparison channel and then reintroduce lexical identity channels and trainable soft routing in a controlled way while measuring whether the equality route remains used.
