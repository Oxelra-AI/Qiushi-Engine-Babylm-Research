# Recorded Training

## Shared Pretraining

- DeBERTa-v2: 8 layers, hidden size 480, 8 heads, feed-forward size 1,920.
- Byte-level BPE: 16,384 entries; sequence length 256.
- Fixed 15% whole-word masking; AdamW; peak learning rate 0.001.
- Cosine decay, 6% linear warmup, weight decay 0.01, effective batch 256.
- Model configuration seed 43; initialization seed 43022; training RNG seed 43023.
- The first residual adapter has bottleneck 128 and scale 1.75.

At 82,012,495 counted words, a second zero-output adapter path is attached.
The Stage I model has 86,005,295 counted word exposures and second-adapter
scale 0.75. Stage III continues this exact parent for 80 updates using
3,162,742 acquisition words plus 517,332 preservation presentations, totaling
89,685,369 words. Only the second adapter is updated in Stage III.

This model's cumulative exposure: **86,005,295 words**.

## Checkpoints and Evaluation

The required intermediate targets reached before these endpoints are 1M through
10M in 1M increments, then 20M through 80M in 10M increments: 17 shared checkpoints.
Actual save boundaries are recorded separately in `CHECKPOINTS.json`.
Neither release substitutes a later 90M/100M branch for its own training history.
AoA adds the exact final endpoint to these 17 shared checkpoints, giving 18 points.

Final scores use the full prediction files, complete adapter-aware SuperGLUE
fine-tuning with seed 42, and measured AoA. Intermediate Fast metrics are attached
separately and do not replace the full final evaluation.

The reference evaluation implementation is
[`babylm-org/babylm-eval`](https://github.com/babylm-org/babylm-eval),
with the AoA estimator traced to commit `6f825c291e2c4c78ad33b1935fd64d45f52642dc`.
The official early-stop instructions require matching checkpoint lists to the
actual training endpoint. See the [submission instructions](https://github.com/babylm-org/babylm-eval/blob/main/strict/README.md#submission-requirements).
