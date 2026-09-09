# causal lm objective generality plan hash-mixed exact/rewrite integration

The hash-mixed seed43022 arm preserves the selected sources and 100M-word budget, but assigns same-window companions by deterministic hash: 16,560 exact repeat companions and 16,731 compact rewrite companions (repeat fraction 0.4974). It tests whether mixed relation practice behaves like proportional averaging, exact-recurrence capture, restatement robustness, or coexistence of multiple relation-conditioned readouts. It does not by itself separate a half-dose restatement curve from active competition between exact and restatement relations.

## Compact nonoverlap rewrite and natural-copy readouts

| readout | HM−C | HM−R | HM−V | R−C | V−C | split references |
|---|---:|---:|---:|---:|---:|---|
| compact nonoverlap rewrite gain | +0.2422 ± 0.0439 | +0.9940 | -0.4414 | -0.7517 | +0.6837 | RS−C -0.0313; VS−C +0.0298 |
| natural-copy gain | +0.4837 ± 0.0405 | -0.0159 | +0.1566 | +0.4996 | +0.3271 | RS−C -0.2005; VS−C +0.0997 |

For compact nonoverlap rewrite gain, the half-weighted endpoint prediction is -0.0303, while observed HM−C is +0.2422. For natural copy, HM−C is +0.4837, nearly original REPEAT (+0.4996) and above VIEW (+0.3271). Thus a half-exact/half-rewrite window diet is not captured by exact recurrence: it retains near-full exact-repeat copy behavior and a reduced but real source-conditioned restatement readout. The restatement readout is substantially weaker than VIEW (HM−V -0.4414), but this single arm cannot decide whether the shortfall is half-dose restatement learning or active suppression by exact adjacency.

## T/U component placement

On compact nonoverlap rewrites, HM−C true-source NLL is -0.7313 and unrelated-source NLL is -0.4890. Relative to VIEW, HM has almost the same unrelated-source term (HM−V U +0.0061) but a worse true-source term (HM−V T +0.4475). The exact-recurrence half therefore selectively reduces the additional benefit from the true related source while leaving broad compact target fit close to the restatement endpoint. Relative to REPEAT, HM is much better on true-source use (HM−R T -1.1792) and modestly better on the unrelated term (HM−R U -0.1853).

## Entity cue-ablation placement

On the probe results interpretation stale-gold Entity cue-ablation readout, rel≥3 margin_full HM−C is -0.1675; HM−R is +0.1568; HM−V is -0.8480. This is not official Entity scoring and should remain secondary. It only indicates that HM is intermediate rather than an endpoint clone.

## Scientific reading

Hash-mixed same-window training adds an auxiliary composition result to the relation-typed sequence-composition principle. Under a 50/50 local diet, near-full exact-repeat copy behavior and a positive compact restatement readout are both visible. The shortfall relative to VIEW is concentrated in true-source use while the unrelated-source compact term is close to VIEW, making active competition plausible, but the present design also changed the VIEW relation dose. The result therefore supports coexistence of relation-conditioned behavior and motivates the HALF_VIEW control; it should not be used alone as proof of interference. A recipe-matched half-dose single-relation control is needed to distinguish dose response from active exact/restatement competition.

## Files

- `experiments/archive/relation_learning/data/hash_mixed_integration/hash_mixed_rewrite_contrasts.csv`
- `experiments/archive/relation_learning/data/hash_mixed_integration/hash_mixed_copy_contrasts.csv`
- `experiments/archive/relation_learning/data/hash_mixed_integration/hash_mixed_entity_ablation_contrasts.csv`
