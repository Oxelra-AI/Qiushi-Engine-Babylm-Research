# causal intervention: Context-Credit Replay — Design and Scientific Motivation

## Scientific question

Can the private adapter learn more transferable competence from the legal data
tail by focusing credit on tokens the frozen carrier cannot already predict?

## Connection to synthetic mechanism

The orbit-binding synthetic line (orbit binding design) established that:

1. **Relation-aligned answer credit** preserves a low-dimensional layer-1 selection
   interface that generalizes to held entities
2. **Bag-independent credit** at the same weight/schedule collapses this interface
3. The key difference is not the weight but *what* the credit is aligned with:
   query-conditioned relational targets versus locally predictable bag targets
4. **Interface reach** — the set of symbols/contexts that can instantiate the
   causal relation interface — narrows under broader prediction pressure,
   even when familiar-symbol competence recovers

## Translation to BabyLM

The inherited 42.12 recipe uses a frozen-carrier/plastic-adapter architecture:
- **Carrier** (frozen): scale1.75 chck_82M adapter DeBERTa-v2, 35.46M params
- **Private adapter** (plastic): bottleneck-128, 995K params, trained on ~4M-word tail
- **Inference**: private_adapter_scale = 0.75

Currently, standard WWM MLM treats all masked tokens equally. The private adapter
spends credit on tokens the carrier already predicts well (local collocations,
common patterns) alongside genuinely new competence (relational/contextual patterns
the carrier misses).

### Carrier-residual credit

For each masked position i:
- w_i = 1 - p_carrier(correct_token_i)
- Normalized so mean(w) = 1 over all masked positions

This means:
- Tokens the carrier predicts with high probability → low weight → private adapter
  focuses less on redundant patterns
- Tokens the carrier cannot predict → high weight → private adapter focuses on
  what the carrier genuinely doesn't know

### What this is NOT

This is **not** simple confidence filtering . The weights
come from the frozen carrier, not from the model being trained. We are asking
"what does the existing 82M model NOT already know?" rather than "what is the
current model uncertain about?"

The concern about suppressing useful uncertain predictions applies to self-teachers
and student-teacher distillation where the model's own uncertainty determines what
it sees. Here, the carrier's certainty determines the credit allocation, and the
private adapter is free to learn from any token — just with differential emphasis.

### Falsification criterion

If carrier_residual credit leads to WORSE equal7 or Overall than the standard
control, this would suggest that:
- The carrier-error signal is noise rather than useful structure
- The private adapter benefits from uniform exposure including redundant patterns
- The synthetic mechanism doesn't transfer to the BabyLM setting

If carrier_residual is COMPARABLE to standard, the credit reallocation doesn't
help or hurt — the ~4M-word tail may be too short for credit allocation to matter.

If carrier_residual is BETTER, the directed credit hypothesis has empirical support.

## Experimental design

### Arms

| Arm | Credit mode | Description |
|-----|------------|-------------|
| standard | Equal weight | Exact earlier analysis reproduction |
| carrier_residual | 1 - p_carrier | Upweight carrier-error tokens |

### Shared parameters (matched exactly)

- Endpoint: chck_82M (scale1.75, 35.46M params)
- Data: compact-view-reinvest 100M stream, rows 530,944+
- Tail budget: 3,992,918 words (82M → ~86M)
- Private adapter: bottleneck=128, training scale=1.0, inference alpha=0.75
- Optimizer: AdamW, lr=0.001, betas=(0.9, 0.98), eps=1e-6, wd=0.01
- Schedule: cosine with 6% warmup, 455 total steps
- Masking: WWM at 15%
- Neutral KL: lambda=1.0, subsample=32/accumulation
- Seed: 43023
- Micro-batch: 32, accumulation: 8 (effective batch: 256)
- Gradient checkpointing: enabled
- Gradient clipping: max_norm=1.0

### Differences

Only the per-token loss weight differs. Standard mode uses equal weight;
carrier_residual computes weight from the frozen carrier's softmax probability
of the correct token, normalized to unit mean.

### Smoke test observations

From the smoke test (2-step, 60K words):
- Standard: main_loss ~2.5, neutral_loss ~0.003
- Carrier_residual: main_loss ~4.2 (weighted CE emphasizes harder tokens),
  neutral_loss ~0.005 (private adapter diverges more on hard tokens)
- ~37-44% of masked tokens have carrier_prob > 0.5 ("easy" for carrier)
- Normalized weight range: [0.0, ~2.2], mean=1.0

### Evaluation

Fast official-compatible screen: BLiMP, Supplement, EWoK, Entity, COMPS,
GlobalPIQA, Reading → equal7 (equal-weight average of seven columns).
Compare against known coherent86 alpha0.75 baseline (Overall ~42.12).

### Complementary interventions

Complementary interventions were also in progress; they were not additional variants of confidence weighting.

## Files

- Trainer: `scripts/context_credit_trainer.py`
- Evaluator: `scripts/fast_eval.py`
- Runs: `training/runs/step025_{standard,carrier_residual}_seed43023/`
- This note: `notes/context_credit_design.md`
