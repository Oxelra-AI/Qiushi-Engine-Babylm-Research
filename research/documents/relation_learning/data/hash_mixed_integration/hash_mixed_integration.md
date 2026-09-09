# causal lm objective generality plan hash-mixed exact/rewrite integration

The hash-mixed seed43022 arm preserves the selected sources and 100M-word budget, but assigns same-window companions by deterministic hash: 16,560 exact repeat companions and 16,731 compact rewrite companions (repeat fraction 0.4974). It tests whether mixed relation practice behaves like proportional averaging, identity-dominant capture, restatement robustness, or context-selected dual competence.

## Compact nonoverlap rewrite and natural-copy readouts

| readout | HM−C | HM−R | HM−V | R−C | V−C | split references |
|---|---:|---:|---:|---:|---:|---|
| compact nonoverlap rewrite gain | +0.2422 ± 0.0439 | +0.9940 | -0.4414 | -0.7517 | +0.6837 | RS−C -0.0313; VS−C +0.0298 |
| natural-copy gain | +0.4837 ± 0.0405 | -0.0159 | +0.1566 | +0.4996 | +0.3271 | RS−C -0.2005; VS−C +0.0997 |

For compact nonoverlap rewrite gain, the half-weighted endpoint prediction is -0.0303; observed HM−C is +0.2422. Thus the mixed arm is not captured by exact recurrence despite half of its relation rows being exact. It lands near the proportional cancellation point and near CLEAN, while remaining much better than REPEAT (+0.9940) and much worse than VIEW (-0.4414).

For natural copy, HM−C is positive (+0.4837) but below original REPEAT (+0.4996) and close to or somewhat above VIEW (+0.3271). This is blended relation practice rather than a pure relation-selected double competence.

## T/U component placement

On compact nonoverlap rewrites, HM−C true-source NLL is -0.7313 and unrelated-source NLL is -0.4890; the gain contrast is U−T. The mixture therefore cancels source-conditioned computations rather than simply moving the whole arm uniformly above or below CLEAN.

## Entity cue-ablation placement

On the probe results interpretation stale-gold Entity cue-ablation readout, rel≥3 margin_full HM−C is -0.1675; HM−R is +0.1568; HM−V is -0.8480. This is not official Entity scoring and should remain secondary. It is useful only as another indication that HM is intermediate rather than an endpoint clone.

## Scientific reading

Hash-mixed same-window training strengthens the sequence-composition principle: when a finite learner practices incompatible local relations on the same family of content, the installed source-use computation can partially cancel. The result does not support a strong near-duplicate-contamination threshold at 50%; it shows that exact and restatement relation practice blend under this dose. A dose curve would be needed to state when minority exact recurrence dominates. For the current paper, hash-mix belongs as a boundary/composability result, not as part of the core causal proof.

## Files

- `experiments/archive/relation_learning/data/hash_mixed_integration/hash_mixed_rewrite_contrasts.csv`
- `experiments/archive/relation_learning/data/hash_mixed_integration/hash_mixed_copy_contrasts.csv`
- `experiments/archive/relation_learning/data/hash_mixed_integration/hash_mixed_entity_ablation_contrasts.csv`
