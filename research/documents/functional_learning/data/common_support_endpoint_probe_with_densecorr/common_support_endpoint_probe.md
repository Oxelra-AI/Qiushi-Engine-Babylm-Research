# earlier analysis common-support endpoint probe

Created: `2026-09-08T08:01:33Z`

This probe reuses the deterministic common token positions from the earlier analysis rendering comparison. It scores ground-truth CE/rank and parent KL for existing endpoints under ordinary-common and dense-common renderings. The positions are familiar training-row positions but were not sparse acquisition labels; their improvement is not held-out transfer evidence.

## Target support

- Macro indices: `[0, 20, 40, 60]`
- Frozen target record JSONL: `experiments/archive/functional_learning/data/common_support_endpoint_probe_with_densecorr/frozen_common_support_targets.jsonl`

| branch | examples | target tokens |
|---|---:|---:|
| ordinary_common | 169 | 1003 |
| dense_common | 169 | 1003 |

## Endpoint scores

| endpoint | branch | KL(parent||endpoint) | parent CE | endpoint CE | CE gain vs parent | parent rank | endpoint rank | rank gain vs parent |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| coherent86_parent | ordinary_common | 2.31204e-09 | 2.73 | 2.73 | 0 | 174.445 | 174.445 | 0 |
| coherent86_parent | dense_common | 1.83821e-09 | 4.04548 | 4.04548 | 0 | 381.316 | 381.316 | 0 |
| ordinary_inherited_wwm_seed62064 | ordinary_common | 0.000714716 | 2.73 | 2.7221 | 0.00790477 | 174.445 | 172.388 | 2.05683 |
| ordinary_inherited_wwm_seed62064 | dense_common | 0.00130078 | 4.04548 | 4.05042 | -0.00494313 | 381.316 | 380.614 | 0.701894 |
| exact_ms_seed62064 | ordinary_common | 0.0288476 | 2.73 | 2.70832 | 0.0216833 | 174.445 | 180.167 | -5.72183 |
| exact_ms_seed62064 | dense_common | 0.502657 | 4.04548 | 3.28843 | 0.757052 | 381.316 | 248.468 | 132.848 |
| clean_ms_kl_seed62064 | ordinary_common | 0.0112702 | 2.73 | 2.71126 | 0.0187424 | 174.445 | 177.174 | -2.72981 |
| clean_ms_kl_seed62064 | dense_common | 0.352518 | 4.04548 | 3.36331 | 0.682168 | 381.316 | 252.972 | 128.344 |
| clean_ms_kl_seed62065 | ordinary_common | 0.0113457 | 2.73 | 2.70978 | 0.020216 | 174.445 | 176.71 | -2.2652 |
| clean_ms_kl_seed62065 | dense_common | 0.345477 | 4.04548 | 3.36611 | 0.679364 | 381.316 | 253.312 | 128.004 |
| densecorr_ms_kl_seed62064 | ordinary_common | 0.00169977 | 2.73 | 2.70971 | 0.0202908 | 174.445 | 173.622 | 0.822532 |
| densecorr_ms_kl_seed62064 | dense_common | 0.00935649 | 4.04548 | 3.88678 | 0.158699 | 381.316 | 346.492 | 34.8245 |

## Dense-minus-ordinary common contrast

| endpoint | Δ endpoint CE | Δ CE gain vs parent | Δ endpoint rank | Δ rank gain vs parent | Δ KL(parent||endpoint) |
|---|---:|---:|---:|---:|---:|
| coherent86_parent | 1.31548 | 0 | 206.871 | 0 | -4.7383e-10 |
| ordinary_inherited_wwm_seed62064 | 1.32833 | -0.0128479 | 208.226 | -1.35494 | 0.000586067 |
| exact_ms_seed62064 | 0.580109 | 0.735369 | 68.3011 | 138.57 | 0.47381 |
| clean_ms_kl_seed62064 | 0.652052 | 0.663426 | 75.7976 | 131.074 | 0.341248 |
| clean_ms_kl_seed62065 | 0.65633 | 0.659148 | 76.6022 | 130.269 | 0.334131 |
| densecorr_ms_kl_seed62064 | 1.17707 | 0.138409 | 172.869 | 34.002 | 0.00765672 |

## Relative to exact `(M,S)` seed62064

Negative CE/rank deltas mean the endpoint predicts the ground-truth token better than exact `(M,S)` on the same rendering and positions.

| endpoint | branch | endpoint CE − exact CE | CE gain diff | endpoint rank − exact rank | rank gain diff | KL diff |
|---|---|---:|---:|---:|---:|---:|
| coherent86_parent | ordinary_common | 0.0216833 | -0.0216833 | -5.72183 | 5.72183 | -0.0288476 |
| coherent86_parent | dense_common | 0.757052 | -0.757052 | 132.848 | -132.848 | -0.502657 |
| ordinary_inherited_wwm_seed62064 | ordinary_common | 0.0137785 | -0.0137785 | -7.77866 | 7.77866 | -0.0281328 |
| ordinary_inherited_wwm_seed62064 | dense_common | 0.761995 | -0.761995 | 132.147 | -132.147 | -0.501356 |
| clean_ms_kl_seed62064 | ordinary_common | 0.0029409 | -0.0029409 | -2.99202 | 2.99202 | -0.0175773 |
| clean_ms_kl_seed62064 | dense_common | 0.0748842 | -0.0748842 | 4.50449 | -4.50449 | -0.150139 |
| clean_ms_kl_seed62065 | ordinary_common | 0.00146734 | -0.00146734 | -3.45663 | 3.45663 | -0.0175019 |
| clean_ms_kl_seed62065 | dense_common | 0.0776886 | -0.0776886 | 4.84447 | -4.84447 | -0.157181 |
| densecorr_ms_kl_seed62064 | ordinary_common | 0.00139247 | -0.00139247 | -6.54437 | 6.54437 | -0.0271478 |
| densecorr_ms_kl_seed62064 | dense_common | 0.598353 | -0.598353 | 98.0239 | -98.0239 | -0.493301 |

## Scientific reading

At dense-common positions, exact `(M,S)` improves ground-truth CE by 0.757052 and mean rank by 132.848 relative to coherent86, while its parent KL is 0.502657. This is the key signal: dense-state parent anchoring can constrain predictions where acquisition has made the ground-truth token easier, not only states where the student has drifted harmfully. On ordinary-common rendering at the same positions, exact `(M,S)` changes CE by only 0.021683 and rank by -5.722; the dense rendering is therefore where the acquired fit is concentrated. clean_ms_kl_seed62064 has dense-common CE gain 0.682168, rank gain 128.344, and parent KL 0.352518. Compare this triple with exact `(M,S)` rather than reading reduced KL alone as better preservation. clean_ms_kl_seed62065 has dense-common CE gain 0.679364, rank gain 128.004, and parent KL 0.345477. Compare this triple with exact `(M,S)` rather than reading reduced KL alone as better preservation. ordinary_inherited_wwm_seed62064 has dense-common CE gain -0.004943, rank gain 0.702, and parent KL 0.001301. Compare this triple with exact `(M,S)` rather than reading reduced KL alone as better preservation. densecorr_ms_kl_seed62064 has dense-common CE gain 0.158699, rank gain 34.825, and parent KL 0.009356. Compare this triple with exact `(M,S)` rather than reading reduced KL alone as better preservation.
