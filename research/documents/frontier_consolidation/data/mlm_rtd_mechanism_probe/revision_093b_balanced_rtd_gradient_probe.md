# Step093b: Balanced RTD Calibration and Trained-Head Gradient Geometry

Balanced RTD head: loss 0.6949 → 0.5739 (80 steps)
Hard held-out: balanced_acc=0.6844, AUROC=0.7584, replaced_recall=0.7196, original_recall=0.6493
Random held-out: balanced_acc=0.7543, AUROC=0.9495, replaced_recall=0.9666, original_recall=0.5419
Random-hard gap: balanced_acc=0.0698, AUROC=0.1911

## Trained-head gradient geometry
embedding        cosine=+0.0273  RTD/MLM norm=0.094
embeddings_other cosine=+0.0293  RTD/MLM norm=0.165
layer_0          cosine=+0.0521  RTD/MLM norm=0.189
layer_1          cosine=+0.0661  RTD/MLM norm=0.205
layer_2          cosine=+0.1353  RTD/MLM norm=0.227
layer_3          cosine=+0.0291  RTD/MLM norm=0.213
layer_4          cosine=+0.0351  RTD/MLM norm=0.233
layer_5          cosine=+0.0118  RTD/MLM norm=0.254
layer_6          cosine=+0.0071  RTD/MLM norm=0.264
layer_7          cosine=+0.0067  RTD/MLM norm=0.343
rel_embeddings   cosine=-0.1876  RTD/MLM norm=0.268

Mean trunk cosine: +0.0429
Mean trunk RTD/MLM norm ratio: 0.241

## Interpretation
- context_requirement: Random corruptions are easier by 0.070 balanced accuracy and 0.191 AUROC; hard model-sampled corruption discrimination therefore requires substantially more contextual knowledge than token-identity anomaly detection.
- hard_signal: Held-out hard RTD balanced accuracy=0.684, AUROC=0.758, replaced recall=0.720; the signal is learnable but not saturated.
- gradient: With a trained class-balanced head, mean trunk MLM-vs-RTD cosine=+0.0429 and raw norm ratio=0.241; this sets the local RTD loss-scale needed for a joint trainer rather than importing lambda=50.
- gdes: Embedding cosine=+0.0273, norm ratio=0.094; block RTD gradients from the shared MLM embedding/decoder path.

Elapsed: 227.9s
