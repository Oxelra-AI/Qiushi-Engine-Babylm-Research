# event binding attribution panel — RTD+MLM implementation fix and shortcut diagnosis

## Fixes applied
- Removed duplicate-parameter defect: the GDES-shared discriminator embedding is now held
  as a non-registered reference (`object.__setattr__(self, '_disc_embed', ...)`), so it is
  owned only by the discriminator. AdamW no longer sees it twice. No duplicate-param warning
  in the re-run.
- Discriminator param count confirmed = 34,467,424 (matches protected backbone exactly).

## Smoke result (20k words, 16 steps, batch 8) — DIAGNOSTIC ONLY

| step | gen_loss | disc_mlm | rtd_loss | rtd_acc | pred_original_rate | label_original_rate |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 9.89 | 9.86 | 0.699 | 0.499 | 0.527 | 0.844 |
| 4 | 9.76 | 9.88 | 1.456 | 0.153 | 0.000 | 0.847 |
| 8 | 9.45 | 8.77 | 0.606 | 0.859 | **1.000** | 0.859 |
| 12 | 9.34 | 7.93 | 0.433 | 0.847 | **1.000** | 0.847 |
| 16 | 8.99 | 7.36 | 0.416 | 0.854 | **1.000** | 0.854 |

## Key finding: RTD collapses to the all-original majority-class shortcut

By v0 1m multitask profile and comparison plan+, `rtd_pred_original_rate = 1.0` and `rtd_acc` exactly equals `rtd_label_original_rate`
(~0.85). The RTD head is predicting "original" at EVERY position and getting ~85% accuracy for
free, because only ~15% of tokens are replaced. This is precisely the degenerate solution the
control was intended to exclude.

## Why this happens (mechanistic, not a bug)
The RTD signal is only useful if the generator produces **plausible-but-wrong** replacements
(hard negatives). At 20k words / 16 steps, the generator is essentially untrained (gen_loss ~9,
near log(vocab)=log(16640)≈9.72). Its samples are near-random, but the discriminator here does
not exploit them — it takes the majority-class shortcut because:
1. Class imbalance: 85% original vs 15% replaced → constant "original" gives 85% acc.
2. Weak generator → replaced tokens are either trivially detectable (so RTD saturates and stops
   contributing gradient) OR the head minimizes loss fastest via majority class.

## Design consequences for the real experiment
The full-scale run must be *engineered against the shortcut* and *measured against the baseline*:

1. **Metric: RTD accuracy ABOVE majority-class.** Report `rtd_acc - max(orig_rate, 1-orig_rate)`.
   Success requires RTD to beat the constant-original predictor (~0.85), not merely reach it.
   Also report RTD accuracy split by original vs replaced positions (recall on the replaced class
   is the real signal; majority-class shortcut has replaced-recall ≈ 0).

2. **Generator must be strong enough to make hard negatives.** With adequate exposure (1M+), the
   generator's MLM loss drops and its samples become plausible, forcing the discriminator to use
   context. Must verify replaced-class recall rises above 0 during real training.

3. **Class-balancing option.** Consider upweighting the replaced class in RTD BCE, or focal loss,
   to prevent majority-class collapse. But do NOT hand-tune to fabricate signal — the honest test
   is whether RTD contributes real contextual discrimination.

4. **Context-ablation probe (required control).** On non-evaluation text, for a fixed later
   token, delete or shuffle EARLIER sentences and measure the change in that token's RTD logit and
   MLM logit. If RTD/MLM genuinely uses cross-sentence context, earlier-context ablation must
   change later-token predictions specifically (more than a matched local-only control). This is
   the decisive evidence that the objective alters cross-sentence credit assignment — separate from
   whether RTD merely adds local signal density.

## Decision
Do NOT launch 100M. Next: run a matched 1M WWM vs WWM+RTD comparison with (a) correct
lr_total_steps (no LR collapse), (b) replaced-class recall reporting, (c) the context-ablation
probe. Only if RTD shows real above-majority, above-zero-replaced-recall contextual discrimination
AND context-ablation specificity should scaling be considered.
