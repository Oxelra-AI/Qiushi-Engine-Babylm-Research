# successor route comparison while minfreq runs source-view representation alignment probe

CPU-only measurement over already trained checkpoints. It does not train, evaluate official tasks, change data, or select the next H100 run.

## Purpose

Measure whether source spans and their compact rewrite spans in the frozen changed block are already close in the model representation, using shuffled and same-row controls. This informs the possible leverage of a future low-weight shared-subspace consistency loss if the active minfreq50 screen is weak.

## Sample
- changed examples available: `3005`
- changed examples sampled: `160`
- pair span records in sampled rows: `635`
- train SHA: `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`
- tokenizer SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`

## Last-layer centered geometry
- tokenmean_20M: pairs=635, centered actual cos mean=0.7799, shuffled=-0.0150, margin=0.7949, same-row-other=0.2676, centered retrieval top1=0.9827, top5=0.9937, elapsed=10.015s
- tokenmean_80M: pairs=635, centered actual cos mean=0.8232, shuffled=-0.0242, margin=0.8474, same-row-other=0.0318, centered retrieval top1=1.0000, top5=1.0000, elapsed=9.266s
- tokenmean_100M: pairs=635, centered actual cos mean=0.8251, shuffled=-0.0239, margin=0.8490, same-row-other=0.0323, centered retrieval top1=1.0000, top5=1.0000, elapsed=9.249s
- wordmean_80M: pairs=635, centered actual cos mean=0.8089, shuffled=-0.0209, margin=0.8298, same-row-other=0.0527, centered retrieval top1=0.9969, top5=1.0000, elapsed=9.442s

## Interpretation
- Token-mean 80M has a measurable paired-view geometry in the last layer: centered actual-minus-shuffled cosine margin 0.8474, retrieval top1 1.0000, and same-row-other mean 0.0318. This means the training stream already induces some source/rewrite abstraction; a future consistency loss should be low-weight and subspace-limited rather than full-vector forcing.
- Word-mean 80M changes last-layer paired geometry relative to token-mean by margin delta -0.0176 and retrieval-top1 delta -0.0031. Because word-mean harmed broad language scores, any increase here would not by itself endorse stronger alignment; any decrease would suggest that its GlobalPIQA/COMPS signal is not source-view abstraction.
- This CPU result is not behavior evidence for a source-view objective. It only tells the next route choice whether the validated compact-view mechanism still has representational headroom and what failure mode to avoid: forcing all source/rewrite information together instead of extracting a small invariant component while preserving residual details.

Full JSON: `experiments/archive/frontier_consolidation/data/pair_alignment_probe/pair_alignment_probe.json`
