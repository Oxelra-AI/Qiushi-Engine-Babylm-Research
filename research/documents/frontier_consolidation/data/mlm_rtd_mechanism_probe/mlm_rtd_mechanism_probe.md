# earlier analysis: MLM+RTD Gradient-Separated Mechanism Probe

Checkpoint: chck_80M (34,467,424 params)
Probe data: 5184 examples, 799882 words
Batches: 15 train + 6 eval, batch 256×seq256

## Phase A: Gradient Geometry (untrained RTD head)
MLM loss mean: 2.4947
RTD hard loss mean: 0.6528
RTD random loss mean: 0.6605

### Per-layer gradient cosine (MLM vs RTD-hard) and norm ratio
  embedding     cos=-0.0015  norm_ratio=0.1026
  layer_0       cos=+0.0020  norm_ratio=0.5435
  layer_1       cos=-0.0034  norm_ratio=0.6007
  layer_2       cos=-0.0161  norm_ratio=0.7807
  layer_3       cos=-0.0132  norm_ratio=0.7170
  layer_4       cos=-0.0122  norm_ratio=0.8071
  layer_5       cos=-0.0102  norm_ratio=0.8914
  layer_6       cos=-0.0011  norm_ratio=1.0905
  layer_7       cos=+0.0010  norm_ratio=1.3746
  embeddings_other  cos=+0.0010  norm_ratio=0.2774
  mlm_head      cos=—  norm_ratio=—
  rel_embeddings  cos=+0.1835  norm_ratio=0.1341

Trunk cosine mean (hard): -0.0067
Trunk cosine mean (rand): -0.0079
Embedding RTD/MLM norm ratio: 0.10264259846973177

## Corruption Statistics
Generator accuracy (T=1.0): 44.6%
Hard replacement rate: 55.4%

## Phase B: Calibrated RTD Head ({step} steps frozen-encoder training)
RTD head loss: 0.6901 → 0.2434
Hard:   acc=0.9234  repl_acc=0.0382  auroc=0.7476
Random: acc=0.9234  repl_acc=0.5195  auroc=0.9449
Shortcut gap (random - hard): 0.0000

## Interpretation
  shortcut: WEAK: Shortcut gap = 0.000. Model-sampled corruptions may be nearly as detectable as random ones.
  gradient_compatibility: ORTHOGONAL: Trunk gradient cosine = -0.0067 (6/8 negative). RTD provides independent signal, no strong conflict.
  gdes_necessity: RECOMMENDED: RTD embedding gradient = 0.103x MLM. GDES blocking is advisable.
  generator_quality: At temperature 1.0, the 80M model correctly predicts 44.6% of masked tokens, leaving 55.4% as genuine replacements for RTD.
  overall: GOOD: Calibrated hard RTD accuracy = 0.923. The task is learnable but not trivial.

Elapsed: 160.6s
