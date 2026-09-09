# earlier analysis dense seed item-overlap analysis

Common fast-screen items across parent/seed62064/seed62065: `108219`.

## Overall item overlap
- Accuracies: parent `54.05243071918979`, seed62064 `54.107873848400004`, seed62065 `54.10232953547898`.
- Seed agreement fraction: `0.9974311350132602`; disagreement fraction `0.0025688649867398514`.
- Gains vs parent: seed62064 `3819`, seed62065 `3879`, shared `3777`, gain Jaccard `0.96327467482785`.
- Losses vs parent: seed62064 `3759`, seed62065 `3825`, shared `3725`, loss Jaccard `0.9652759782327027`.
- Items both dense seeds gain over parent: `3777`; items both dense seeds lose vs parent: `3725`; shared net item delta `52`.

## Column summaries
- BLiMP: n `13400`, seed agreement `0.9985820895522388`, shared gains/losses `172/245`, gain/loss Jaccard `0.9608938547486033`/`0.953307392996109`, seed disagreements `19`.
- COMPS: n `91028`, seed agreement `0.997242606670475`, shared gains/losses `3493/3379`, gain/loss Jaccard `0.9635862068965517`/`0.965980560320183`, seed disagreements `251`.
- EWoK: n `1100`, seed agreement `0.9945454545454545`, shared gains/losses `30/28`, gain/loss Jaccard `0.8823529411764706`/`0.9333333333333333`, seed disagreements `6`.
- Entity: n `2238`, seed agreement `0.9991063449508489`, shared gains/losses `76/69`, gain/loss Jaccard `0.987012987012987`/`0.9857142857142858`, seed disagreements `2`.
- GlobalPIQA_nonparallel: n `100`, seed agreement `1.0`, shared gains/losses `2/0`, gain/loss Jaccard `1.0`/`None`, seed disagreements `0`.
- GlobalPIQA_parallel: n `103`, seed agreement `1.0`, shared gains/losses `2/1`, gain/loss Jaccard `1.0`/`1.0`, seed disagreements `0`.
- Supplement: n `250`, seed agreement `1.0`, shared gains/losses `2/3`, gain/loss Jaccard `1.0`/`1.0`, seed disagreements `0`.

## Interpretation
The two dense seeds are nearly identical at the item level on the fast screen, not merely in aggregate scores. The remaining fast-screen movement relative to coherent86 is therefore a stable consequence of the dense policy under these two seeds, while official full-eval and SuperGLUE can still overturn practical promotion.
