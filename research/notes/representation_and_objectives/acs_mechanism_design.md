# acs mechanism design: Auxiliary Contrastive Softmax (ACS) Mechanism Design

## Scientific motivation

The central unresolved failure across all tested architectures, corpora, objectives, and
optimizer paths is **context-conditioned alternative binding**: models learn local pivot
compatibility (word-context associations) but fail when the same plausible alternatives
must bind differently under competing contexts (EWoK conditional reversals, GlobalPIQA
parallel deep-rank errors).

Standard MLM distributes the masked-token gradient across the entire vocabulary V:
  ∂L_CE/∂z_k = P(k|context)   for k ≠ y (correct)
  ∂L_CE/∂z_y = P(y|context) - 1

For a 40K vocabulary, each wrong token receives gradient proportional to its small probability.
The top-K competitors that the model actually confuses with the correct token receive only
modestly more gradient than irrelevant vocabulary items. This dilutes the signal for
context-dependent discrimination.

## ACS mechanism

For each masked position, compute an auxiliary contrastive loss over the correct token
and the K strongest wrong tokens (hard negatives selected by detached top-K):

  L_ACS = -log( exp(z_y) / (exp(z_y) + Σ_{k∈topK\{y}} exp(z_k)) )

This is a (K+1)-way classification that concentrates gradient on the specific alternatives
the model currently confuses with the correct token. The gradient amplification for each
top-K competitor is approximately V/(K+1) ≈ 4,444× relative to standard CE.

Total loss: L = L_CE + α · L_ACS

Key properties:
- **Dense:** applies to every masked position in every batch
- **Dynamic:** negatives change as the model learns
- **Context-dependent:** the hard negatives depend on each specific context
- **No additional data or exposure:** uses the existing legal corpus
- **Clean ablation:** α = 0 recovers standard MLM

## Screen design

Paired 80M→100M tail continuations from the legal40k DeBERTa-v2 8×480 baseline
(legal40k accum training completion chck_80M), sharing:
  - Same initial weights
  - Same data stream (80M→100M tail, 129,256 examples, 19,965,632 words, 505 steps)
  - Same tokenizer (legal40k, 40,000 vocab)
  - Same masking seed (train_rng_seed=53023) → identical WWM masks
  - Same dropout seed → identical dropout patterns
  - Same LR schedule (reheat cosine, peak 3e-4, warmup 0.06 fraction)
  - Same optimizer (fresh AdamW moments)

Only intended difference: treatment has ACS loss (α=0.5, K=8); control has standard MLM only.

## Parameters

- α = 0.5: ACS contributes ~14% of CE loss (estimated ACS mean ~0.72 vs CE ~2.5)
- K = 8: captures the 8 strongest competitors; small enough for focused gradient, large
  enough to cover multiple plausible alternatives
- Temperature = 1.0: no temperature scaling

## Success criteria (predeclared)

The treatment arm must show, relative to the matched control:
1. **EWoK:** reduced stable conditional reversals AND improved four-cell interaction
2. **GlobalPIQA hard52:** more correct rows OR lower mean top-minus-correct margin
3. **Without damage:** training CE loss similar to control (±5%); no catastrophic broad-score loss

If criteria 1-2 are met: design a from-beginning recipe with ACS integrated from babylm2026 live surface.
If only one is met: investigate dosage (lower α, fewer K) or application schedule.
If neither is met: ACS is closed as a single-mechanism repair; the gradient concentration
hypothesis does not suffice, and the next route must change the training distribution
or architecture, not just the loss function.

## Relation to closed routes

ACS is genuinely different from all previously tested mechanisms:
- Route B (packets): changed the DATA with synthetic role-switch text → ACS changes the LOSS
- PVDM: changed the MASKING distribution → ACS keeps standard WWM
- Adapters: changed the ARCHITECTURE → ACS keeps the base model
- Bridge/mature averaging: changed WEIGHTS post-training → ACS changes training dynamics
- Natural miners: tried to find special ROWS → ACS applies to every position
- Late readout: tried to protect INFERENCE → ACS changes REPRESENTATION formation

## Implementation files

- Trainer: `scripts/acs_mechanism_screen.py`
- Readout: `scripts/acs_readout.py`
- Treatment run: `training/runs/acs_treatment_alpha05_topk8/`
- Control run: `training/runs/acs_control_standard/`
- This note: `notes/acs_mechanism_design.md`
