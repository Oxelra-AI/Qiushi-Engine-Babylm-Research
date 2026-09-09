# relation filtered pair pools — active relation pair pool semantic audit

CPU-only, model-free audit of whether the muon50 gp and relation pair synthesis active-pair reservoir is likely to be a hard conditional-alternative pool or mostly arbitrary sentence-specific target recovery.

Pairs: **18651**

## Overall indicators

Generic target rate: either=0.111, both=0.013; antonym-hint rate=0.0001
Target char similarity: {'n': 18651, 'mean': 0.21886893751716485, 'median': 0.2, 'p05': 0.0, 'p95': 0.5, 'min': 0.0, 'max': 1.0}; normalized edit distance: {'n': 18651, 'mean': 0.8371155031468546, 'median': 0.8571428571428571, 'p05': 0.5555555555555556, 'p95': 1.0, 'min': 0.125, 'max': 1.0}
Context word Jaccard: {'n': 18651, 'mean': 0.10369209829332665, 'median': 0.10112359550561797, 'p05': 0.0582010582010582, 'p95': 0.15862068965517243, 'min': 0.019736842105263157, 'max': 0.2485207100591716}; local target-window Jaccard: {'n': 18651, 'mean': 0.04899920061106449, 'median': 0.043478260869565216, 'p05': 0.0, 'p95': 0.13636363636363635, 'min': 0.0, 'max': 0.3}

## By family
| family | n | either generic | antonym hint | mean target char J | mean context J | mean local-window J | top targets |
|---|---:|---:|---:|---:|---:|---:|---|
| causal_connector | 5340 | 0.214 | 0.0002 | 0.221 | 0.112 | 0.054 | give:32, would:32, may:32, that's:32, they're:32, there's:32 |
| comparative | 77 | 0.156 | 0.0000 | 0.219 | 0.107 | 0.053 | rather:9, known:4, ask:4, stronger:3, much:3, invites:3 |
| negation | 3538 | 0.115 | 0.0003 | 0.210 | 0.114 | 0.043 | worry:31, forget:29, belong:29, yet:29, hurt:27, nice:27 |
| physical_change | 1285 | 0.119 | 0.0000 | 0.203 | 0.096 | 0.037 | alone:24, small:21, around:21, eyes:21, door:19, room:13 |
| spatial | 4585 | 0.057 | 0.0000 | 0.214 | 0.093 | 0.051 | wall:32, pears:32, bottles:31, alley:31, apple:31, weeks:31 |
| temporal | 3826 | 0.025 | 0.0000 | 0.235 | 0.097 | 0.049 | arrived:29, reached:28, period:26, became:25, week:23, morning:23 |

Interpretation aid:
- A useful training reservoir for the current problem should not merely show large own-context target recovery. It should contain cross-targets that are competitive enough for four-cell margins to differ across checkpoints in the same direction as EWoK/GlobalPIQA relation movement.
- Low antonym/semantic-alternative hints, low context overlap, and generic high-frequency targets would support the suspicion that the reservoir is dominated by ordinary attested-coherence rather than the hard conditional-choice structure.

JSON: `experiments/archive/representation_and_objectives/data/relation_pair_pool_semantic_audit/relation_pair_pool_semantic_audit.json`
CSV: `experiments/archive/representation_and_objectives/data/relation_pair_pool_semantic_audit/relation_pair_pool_semantic_audit_rows.csv`
