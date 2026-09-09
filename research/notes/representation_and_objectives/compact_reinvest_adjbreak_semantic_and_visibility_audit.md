# compact reinvest adjbreak control adjbreak semantic and visibility audit

## Semantic break strength

Pairs audited: 12155.

- original source-own-rewrite content recall mean/median: 0.6512 / 0.6364
- adjbreak source-donor content recall mean/median/p95: 0.0061 / 0.0000 / 0.0625
- original content Jaccard mean/median: 0.5810 / 0.5652
- adjbreak content Jaccard mean/median/p95: 0.0037 / 0.0000 / 0.0357
- adjbreak bigram/trigram Jaccard mean: 0.0009 / 0.0001
- high donor-similarity counts: {'donor_content_recall_ge_0p60': 5, 'donor_content_jaccard_ge_0p50': 5, 'donor_bigram_jaccard_ge_0p30': 3, 'donor_trigram_jaccard_ge_0p20': 3, 'donor_entity_recall_ge_0p75_with_entities': 7, 'donor_number_recall_ge_0p75_with_numbers': 6}

## Visibility transitions

- full in both: 12047
- full only original: 14
- full only adjbreak: 25
- nonfull visible both: 65
- visible-token fraction delta mean/p95/max: 0.000359 / 0.000000 / 0.659574

## Training-order digest

A future adjbreak 100M training file should use the same row-index order formula as the row-holdout materializer: `random.Random(82914124 + 7000 + 1000 + pass_i).shuffle(range(n_rows))`. Digest for 64740 rows and 10 passes: `be18034d83ccd44b1204abe2fa667a3351de6d5fd1a948f192be0c2a36379479`.

## Interpretation

The control is now better supported as a source-own-rewrite correspondence intervention: it preserves marginals and row geometry, keeps aggregate token exposure matched, and the donor rewrite usually has much lower content overlap with the assigned source than the true rewrite. It remains a mechanism control to run only after the already-running full and seed endpoint results justify spending GPU time.

JSON: `experiments/archive/representation_and_objectives/data/compact_reinvest_adjbreak_control/compact_reinvest_adjbreak_semantic_and_visibility_audit.json`
