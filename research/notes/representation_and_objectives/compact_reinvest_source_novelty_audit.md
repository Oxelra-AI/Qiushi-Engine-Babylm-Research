# compact core joint visibility audit compact_view_reinvest source-novelty audit

JSON: `experiments/archive/representation_and_objectives/data/compact_reinvest_source_novelty_audit/compact_reinvest_source_novelty_audit.json`

## Added-source counts

- Core pairs: 10094; added pairs: 2061; reinvest equals core union added: True.
- Core/added source-key overlap: 0; exact normalized added-source duplicates in core: 1.
- Added pairs whose document ID also appears in core: 1779 across 1779 overlapping docs; added doc count: 2061.

## Nearest-core lexical similarity for added sources

- Best trigram Jaccard median/p95/max: 0.0204 / 0.0952 / 1.0000.
- Best bigram Jaccard median/p95/max: 0.0682 / 0.1379 / 1.0000.
- Best content-token Jaccard median/p95/max: 0.0952 / 0.2143 / 1.0000.
- Counts above similarity thresholds: trigram>=0.8 1, trigram>=0.5 1, content>=0.8 1, content>=0.6 1.

## Scientific reading

The added 2,061 compact pairs appear to be mostly additional lexical/source material rather than exact or obvious near-duplicate repetitions of the 10,094 core sources. This supports, but does not prove semantically, the fixed-budget breadth part of the compact_view_reinvest interpretation. Downstream value remains dependent on the running full and independent-seed evaluations.
