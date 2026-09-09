# compact core joint visibility audit compact_view_reinvest pair visibility audit

This CPU audit supports the active compact-view-density route while the GPU full-evaluation and independent-seed jobs run. It does not use a GPU and does not add a new model result.

JSON: `experiments/archive/representation_and_objectives/data/compact_reinvest_pair_visibility_audit/compact_reinvest_pair_visibility_audit.json`

## Pair-level compact signal

- Reinvest selected pairs: 12155 = core 10094 + added 2061; ID union check: True.
- Source words 261803, rewrite words 161708, pair words 423511, rewrite/source word ratio 0.6177.
- Mean content/entity/number recall: 0.6631 / 0.9960 / 1.0000; pairs with entity recall <1: 218.
- Individual source+rewrite pair token length mean/median/p95/max: 50.9781 / 47.0000 / 89.0000 / 167.0000 (no special tokens).
- Individual pairs fitting within 254/256 tokens: 12155 / 12155 of 12155.

## Changed-block row visibility

| arm | changed rows | words | token mean | token p95 | token max | rows >256 with special | visible token fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| view_reinvest | 3006 | 423520 | 208.1377 | 250.0000 | 330.0000 | 91 | 0.998026 |
| repeat_reinvest | 3006 | 423520 | 199.7186 | 240.0000 | 311.0000 | 39 | 0.999187 |
| lengthmatched_reinvest | 3006 | 423520 | 213.2272 | 270.0000 | 395.0000 | 280 | 0.991528 |

## View-vs-repeat token geometry within the changed block

- Row-paired view-minus-repeat token delta (no special) mean/median/p95/sum: 8.4192 / 8.0000 / 19.0000 / 25308.0000.
- Rows with view token length greater/less/equal than repeat: 2771 / 157 / 78.

## Scientific reading

- The active GPU work will decide downstream value. This audit checks whether the proposed adjacent source+compact-view signal is actually available inside the seq256 trainer interface and whether reinvestment changes token/truncation geometry enough to threaten interpretation.
- If full/seed evaluations preserve the fast component pattern, these visibility facts make a matched repeat-reinvest seed run interpretable as a mechanism test rather than a rescue run.
