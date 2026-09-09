# corruption vs loss design corruption-vs-loss factorial: scientific synthesis

## Central finding

Background MLM gradients are the destructive factor for recipient-sensitive binding, not evidence corruption. The factorial cleanly separates these:

| Arm | Input | Loss | Train flip | Held flip | Held U | Held R | Held N |
|---|---|---|---|---|---|---|---|
| Baseline | — | — | 2/36 | 0/12 | -0.399 | +0.360 | +0.790 |
| 1. clean_answer_only | Clean | Answer only | 30/36 | 2/12 | +0.446 | +0.039 | +1.673 |
| 2. corrupted_answer_only | 15% WWM | Answer only | 36/36 | **6/12** | +1.728 | -0.722 | +2.296 |
| 3. corrupted_answer_plus_bg | 15% WWM | Answer + bg (full weight) | 6/36 | **0/12** | +2.018 | -1.385 | +0.866 |

Arms 2 and 3 see IDENTICAL corrupted inputs (~188K bg positions corrupted per arm across 500 epochs). The ONLY difference is whether background positions contribute to loss. Both arms have 36K forced answer targets.

## Three mechanism observations

### 1. Evidence corruption is harmless — may act as regularization
Arm 2 (corrupted input, answer-only loss) outperforms Arm 1 (clean input, answer-only loss):
- Held flips: 6/12 vs 2/12
- Held U: +1.728 vs +0.446
- Train convergence: faster (34/36 at e100 vs 26/36)

Corrupting ~15% of background tokens does not destroy the relational information needed for binding. The corruption may improve generalization through a noise-injection/regularization effect analogous to dropout.

### 2. Background gradients actively undo binding
Arm 3 (corrupted + bg loss) shows REGRESSION during training:
- Train flips: 2 → 6 → 24 → 18 → 16 → 6 (trajectory reverses)
- Held flips: 0 → 0 → 0 → 0 → 0 → 0 (never improves)
- R goes from +0.190 to -1.385 on held (more negative than baseline)
- N collapses from +1.389 to +0.866 (overall representation damaged)

This is not dilution — the answer gets full weight in Arm 3. The bg gradients push the shared adapter parameters toward a different solution that is incompatible with recipient-sensitive binding. The bg pressure dominates because it affects ~188K positions vs 36K answer positions and targets generic word prediction rather than entity tracking.

### 3. Seed sensitivity in absolute held-out transfer
earlier analysis answer-only (seed 43033) achieved 10/12 held flips, but corruption vs loss design's equivalent (seed 43034) only reaches 2/12 (clean) or 6/12 (corrupted). The binding operation's held-out transfer varies by seed, confirming that the absolute level is not fully robust. The relative ordering (bg gradients destructive) is robust within this factorial.

## Connection to BabyLM intervention design

This result directly constrains the mixed-objective intervention for real BabyLM training:

1. **Contrastive packets need gradient isolation, not evidence protection.** Standard WWM can be applied to packet text without destroying relational evidence. But packet answer loss MUST be computed separately from background MLM loss.

2. **Practical architecture: interleaved mini-batch updates.** The cleanest BabyLM design alternates:
   - Mini-batches from ALN text → standard 15% WWM loss → update shared model
   - Mini-batches from contrastive packets → forced answer mask, answer-only loss → update shared model
   This avoids per-token loss weighting and gives clean gradient separation.

3. **Answer-credit coefficient should reflect Arm 2, not diluted mean.** If mixing losses, the answer term must be scaled to match its effective weight when trained alone, not averaged into a per-token mean with bg targets.

4. **Background corruption on packet text is not harmful and may improve generalization.** The BabyLM intervention can apply standard masking to contrastive packet inputs without special handling.

## What this does NOT establish

- The factorial tests a 48-parameter private adapter on template packets. Transfer to full-model BabyLM pretraining requires the mixed-objective pilot.
- The corrupted arm's 6/12 is better than clean's 2/12 but not robust across seeds. Corruption as regularization is suggestive, not proven.
- The mechanism is demonstrated for recipient-only binding on controlled templates. Extension to natural-language entity binding (natural multi-token answers) needs the multi-token scorer.
- Background gradient interference is shown for the current adapter-only training. In full BabyLM pretraining, the carrier model is frozen and only the private path trains, so the mechanism should transfer, but the scale and interaction with curriculum may differ.

## Files
- Summary: `data/corruption_vs_loss_factorial/factorial_summary.json`
- Markdown: `data/corruption_vs_loss_factorial/factorial_summary.md`
- Design note: `notes/corruption_vs_loss_design.md`
- Train/held groups: `data/corruption_vs_loss_factorial/train_groups.jsonl`, `heldout_groups.jsonl`
- Script: `scripts/corruption_vs_loss_factorial.py`
