# permuted companion correspondence control MAX permuted-companion correspondence control

This note records CPU-only construction of the missing permuted-correspondence control. No GPU training or evaluation was launched.

## Scientific role

The existing MAX view/repeat dose arm tests aligned compact view versus exact duplicate recurrence. The MAX breadth arm tests aligned compact view versus additional same-population FineWeb sentences, but breadth is not a pure correspondence control: it changes token fertility, unigram distribution, sentence boundaries, and sentence length selection. The new permuted companion correspondence control arm holds the compact-view text multiset itself fixed and breaks only source-to-own-view correspondence:

- same 33,291 MAX source sentences as matched max dose execution note MAX view/repeat;
- same exact multiset of 33,291 accepted compact rewrites as MAX view;
- every source receives a different pair's rewrite;
- donor rewrite is never from the same source row or from any source document represented in the target row;
- row companion word sums, full row word sequence, suffix/topup/common filler, heldout split, fixed spatial repair route status tokenizer, and 100M pass-order formula are preserved.

If an aligned view model substantially exceeds this permuted arm on the Entity/state carrier, the result supports source-conditioned record re-addressing rather than compact text as generic filler. If aligned and permuted are similar, the record-addressability interpretation should be retired or narrowed, and the useful principle shifts toward compact companion text / regularization / token-distribution effects.

## Materialized files

- materializer: `experiments/archive/frontier_consolidation/scripts/materialize_permuted_companion_max.py`
- 10M pool: `experiments/archive/frontier_consolidation/data/dose_2p64x_permuted_companion_rowholdout_pools/compact_permuted_view_dose2p64x_10M.jsonl`
- 100M stream: `experiments/archive/frontier_consolidation/data/dose_2p64x_permuted_companion_rowholdout_pools/compact_permuted_view_dose2p64x_100M.jsonl`
- assignment sidecar: `experiments/archive/frontier_consolidation/data/dose_2p64x_permuted_companion_rowholdout_pools/compact_permuted_view_dose2p64x_companion_assignment.jsonl`
- row metadata: `experiments/archive/frontier_consolidation/data/dose_2p64x_permuted_companion_rowholdout_pools/compact_permuted_view_dose2p64x_changed_block_rows_meta.jsonl`
- metadata: `experiments/archive/frontier_consolidation/data/dose_2p64x_permuted_companion_rowholdout_pools/permuted_companion_rowholdout_metadata.json`
- summary: `research/documents/frontier_consolidation/data/dose_2p64x_permuted_companion_rowholdout_pools/permuted_companion_rowholdout_summary.md`
- independent audit: `experiments/archive/frontier_consolidation/scripts/audit_permuted_companion_control.py`, outputs under `experiments/archive/frontier_consolidation/data/permuted_companion_independent_audit`

Key hashes:

- 10M pool SHA256: `c834cab0310aff7a5223079861695310c641e9a0d368e79e5d201ac6a8c97bb9`
- 100M stream SHA256: `f5416621a1647f867d2112e1bfcb175b2461ea2c6dbc5413e8ed4daf53122edc`
- assignment SHA256: `c62b27c5af51c7d7c69e71918955d55301316acaf7990ceba175095b728dfbe7`

## Construction details and edge cases

A slot-level same-rewrite-length derangement is possible for 33,269 / 33,291 slots. Three rewrite lengths are globally singleton (39, 40, 42 words), so exact per-slot length preservation is combinatorially impossible for those rows. The materializer uses a documented eight-row row-sum repair for exactly 22 slots:

- exception rows: 2969, 4023, 4025;
- helper rows: 6007, 2231, 3501, 4382, 4496;
- each selected row receives the same number of donor rewrites and the same total companion words as before;
- no donor row/doc violates the correspondence-breaking constraints.

Thus only 22 / 33,291 slots change individual rewrite length, while every changed row retains exact total words and the whole training row sequence is identical to MAX view.

## Audits

Producer audit (`permuted_companion_rowholdout_metadata.json`):

- pair slots 33,291; changed rows 7,923; suffix rows reused from MAX view 57,390;
- exact 10M words and row-length sequence matches MAX view;
- companion text multiset identical to MAX view;
- assignment is a permutation;
- same-pair, same-doc, same-row, and donor-doc-in-target-row-docset counts are all zero;
- all row companion word sums are preserved;
- 100M stream written with matched max dose execution note pass-order seed formula.

Token/WWM geometry versus aligned MAX view (`permuted_companion_token_geometry.json`):

- legal16k token delta: 0 (+0.000000%);
- visible seq256 token delta: +1,629 (+0.011399%);
- WWM group delta: +881 (+0.008973%).

Independent audit (`permuted_companion_independent_audit/independent_audit_summary.json`) passed and reconstructed the control without reusing the materializer's construction functions:

- target set and donor set are all 33,291 pairs;
- zero same-pair, same-doc, same-row, or donor-doc-in-target-row-docset assignments;
- reconstructed prefix exactly matches the written 10M pool;
- suffix rows exactly equal the MAX view pool suffix;
- source and rewrite multiset digests match MAX view;
- metadata 100M stream SHA matches actual stream;
- 100M training order matches the matched max dose execution note formula.

## Conditional training and readout

Training wrapper prepared and dry-run checked, but not launched:

- `experiments/archive/frontier_consolidation/scripts/train_deberta_permuted_companion.py`
- dry-run status: `PERMUTED_COMPANION_TRAIN_PREFLIGHT_OK`
- exact stream: 653,130 rows, 100,000,000 words, 2,552 expected batch256 updates
- fixed tokenizer SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`
- full p2c+c2p DeBERTa: 34,467,424 parameters, vocab 16,384
- default run dir: `experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_permuted_view_dose2p64x_matched_rowholdout_deberta100M_seed43022`

CPU-safe EWoK/Entity evaluator path was extended in `experiments/archive/frontier_consolidation/scripts/cpu_safe_arm_ladder_eval.py` with arm key `max_permuted`; plan check correctly reports 10 checkpoints missing until training occurs. File-only readout prepared in `experiments/archive/frontier_consolidation/scripts/permuted_correspondence_readout.py`; incomplete run wrote `experiments/archive/frontier_consolidation/data/permuted_correspondence_readout/permuted_correspondence_readout_summary.json` with 20 missing rows and no conclusion.

Do not train this arm merely because it exists. Launch one H100 run only if the already-running second-basin MAX DeBERTa view/repeat readout reproduces a substantial Entity-positive aligned effect. Then score only EWoK+Entity first and decompose Entity as:

\[
V-R = (V-P) + (P-R),
\]

where \(V\) is aligned compact view, \(P\) is permuted compact view, and \(R\) is exact repetition. Large \(V-P\) is the correspondence/record-addressability signal; large \(P-R\) is compact companion text without correspondence.
