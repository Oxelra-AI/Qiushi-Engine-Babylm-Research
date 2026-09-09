# causal attention verification Mixed-Objective Construction Design Note

## Hypothesis

Mixing bidirectional reconstruction (MLM) with causal next-token prediction
throughout training improves how limited experience is consolidated into broader,
more transferable knowledge—compared to pure MLM on the same aligned corpus.

## Mechanism

The core intervention is a single bit flip per batch: with deterministic
low-discrepancy scheduling, some batches train with standard whole-word masking
(bidirectional attention, predict masked tokens), while other batches train with
causal attention (lower-triangular mask, predict next token from left context).

The same model weights, prediction head, data order, and word exposure are shared.
The only change is the learning signal:
- MLM forces the model to integrate both left and right context simultaneously.
- Causal prediction forces efficient use of incremental left context, which may
  improve sequential reasoning, entity tracking through text, and reading
  comprehension—areas where the model needs to understand information flow.

## Architecture Match

No architecture changes from clean-Qwen baseline:
- DeBERTa-v2: 8 layers, hidden=480, 8 heads, FFN=1920, 34,467,424 parameters
- Baseline 16k tokenizer (shared)
- Disentangled attention: content-to-position + position-to-content
- Relative position embeddings (256 buckets)

For causal batches, the same disentangled attention operates with a triangular
mask—each position attends only to itself and earlier positions. The prediction
head is shared: the same linear projection that maps hidden states to vocab
logits serves both MLM (predict masked) and causal (predict next) objectives.

## Experimental Arms

| Arm | causal_fraction | MLM batches | Causal batches | GPU |
|-----|----------------|-------------|----------------|-----|
| causal50 | 0.50 | ~1,258 | ~1,257 | 0 |
| causal15 | 0.15 | ~2,138 | ~377 | 1 |
| Control (existing) | 0.0 | ~2,515 | 0 | — |

All share: extra_init_seed=43022, train_rng_seed=43023, batch_size=256,
seq_length=256, learning_rate=1e-3, warmup_fraction=0.06, weight_decay=0.01,
mask_prob=0.15, AdamW betas=(0.9, 0.98), cosine LR over full planned_steps.

## Prediction target density

MLM predicts about 15% of non-padding tokens (expanded to whole-word groups),
whereas the causal batch predicts every valid next token. Thus a causal batch has
roughly 6–7× more supervised positions than an MLM batch at the same batch and
sequence geometry. This asymmetry is inherent to the objectives, while each
optimizer update still averages one objective loss over that objective's valid
targets. The realized target counts are recorded separately in
`scientific_metrics.json` rather than estimated from padded sequence length.

## Control and comparison

- The existing clean-Qwen pure-MLM run (Overall 41.3443, equal7 43.1129) serves
  as the matched control (same data, init, word exposure, architecture).
- Initialization is verified identical (same embedding SHA256 hash).
- Word counting is objective-independent: each consumed row's whitespace words
  are counted once regardless of whether that batch used MLM or causal loss.
- The masking RNG (mask_gen) is consumed only on MLM batches, so later MLM batches
  in the mixed run see slightly different masking patterns than the pure-MLM run.
  This is acceptable—the intervention IS the objective mixing, not masking identity.

## Success criterion

- If either arm's best equal7 exceeds clean-Qwen's 43.1129, the mechanism has
  positive no-AoA evidence.
- If best equal7 exceeds ~43.5 with coherent profile, proceed to full 9-column
  evaluation (SuperGLUE + AoA) to test whether the mixed objective avoids the
  AoA collapse seen in the tail-restart family.
- The key test: does this from-scratch mixed-objective training preserve AoA≥0
  (unlike the 80M→continuation family) while improving broad no-AoA columns?

## Relation to prior work

- GPT-BERT (BabyLM baseline) uses alternating MLM/causal modes. Our test is
  architecture-matched on the proven DeBERTa-v2 backbone rather than bundling
  architecture changes.
- The tail-restart family showed that non-uniform masking can improve no-AoA
  columns but damages AoA acquisition when applied as late continuation. A
  from-scratch mixed objective tests whether the same kind of signal enrichment
  preserves acquisition dynamics when present throughout training.
- The visible leader uses WWM→token switching (different masking granularity over
  time). Our mixed objective tests a different dimension: prediction direction
  (bidirectional vs causal) rather than prediction granularity.

## Files

- Trainer: `scripts/mixed_objective_trainer.py`
- Launch: `scripts/launch_mixed_objective_training.sh`
- Evaluation: `scripts/eval_mixed_objective_noaoa.sh`
- Training outputs: `training/runs/mixed_causal{50,15}_qwen_seed43022/`
- Evaluation results: `data/mixed_objective_eval/`
