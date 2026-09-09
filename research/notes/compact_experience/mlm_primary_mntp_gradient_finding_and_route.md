# mlm primary mntp gradient finding and route — MLM-primary same-corruption MNTP: gradient-complementarity finding and route

## Why this step exists
full eval repair and measurement state fully measured the causal attention verification mixed-objective package (whole causal batches replacing whole MLM batches):
- `causal15` true-100M Overall **41.20913** (delta vs clean-Qwen `-0.13516`), equal7 `42.9786`, SuperGLUE `70.0322`, AoA 0.0.
- `causal50` true-100M Overall **40.09992** (delta `-1.24437`), equal7 `41.9243`, SuperGLUE `67.4293`, AoA 0.0.
Both are AoA-safe but below clean-Qwen `41.34429`. `causal15` traded Supplement (`-2.94`) and EWoK (`-1.62`) for GlobalPIQA (`+2.485`) and Reading (`+0.965`); `causal50` broadly damaged the profile.

The experimental constraint: do not run a lower-fraction sweep. Keep architecture fixed. Test whether directional prediction is a **complementary credit-assignment signal** that can be added while retaining the full legally counted MLM stream, with any weighting/conflict handling derived only from training-gradient statistics. If no stable gradient complementarity appears, pivot to representation/architecture rather than spending another 100M-word run on dose interpolation.

## What was built
`scripts/mlm_mntp_gradient_probe.py` — a pre-training probe (no training, no eval/AoA/CDI/leaderboard signal). On the same WWM-corrupted clean-Qwen training batch it computes, per parameter group, the gradient relationship between:
- `mlm` loss (logits at masked position j predict x_j), and
- `mntp` auxiliary loss on the SAME corruption:
  - `token_shift`: predict masked x_j from position j-1;
  - `word_start`: same but only first subtoken of each selected word.

Metrics per group: cosine(grad_mlm, grad_aux), negative_dot fraction, aux/mlm gradient-norm ratio. Deterministic mask sequence reused across all checkpoints. Verified in CPU smoke (fixed a graph-retention bug: two auxiliaries from one forward needed retain_graph on the first).

Artifacts:
- `data/gradient_complementarity/smoke_init_cpu.json`
- `data/gradient_complementarity/clean_qwen_init_10M_50M_100M_grad_probe.json` (corpus head, 4×4)
- `data/gradient_complementarity/clean_qwen_broad_stride4096_10M_50M_100M_grad_probe.json` (8×8, stride 4096 across corpus)

## Findings (clean-Qwen backbone; init, 10M, 50M, 100M)
1. **No destructive conflict.** Full-parameter (`all`) mean cosine is positive at every trained checkpoint: broad probe `token_shift` ~0.060 (10M), 0.098 (50M), 0.092 (100M); `word_start` ~0.051 / 0.081 / 0.077. `full_gradient_conflict=false` everywhere. This is qualitatively different from a signal that removes MLM pressure.
2. **Layer structure.** Embeddings, lm_head, and mid/upper encoder layers are consistently positively aligned. Earliest layers (00/01) show mild negative cosine at 10M (negative_fraction up to 0.625) but recover to positive by 50M–100M. The `other` group (LayerNorm/bias scalars, negligible parameter mass) turns negative late.
3. **Gradient-norm mismatch is the real design issue, not conflict.** Aux/MLM gradient-norm ratio is ~2–4× at trained checkpoints (aux loss ~12 vs MLM ~2.6, because the pure-MLM model was never trained to predict at shifted positions). A naive equal-weight auxiliary would be dominated by the auxiliary. Any auxiliary must be **norm-calibrated** (e.g. scale aux gradient to a target fraction of MLM gradient norm), which satisfies the "weighting from training-gradient statistics" constraint.
4. `token_shift` is slightly more MLM-aligned than `word_start`; `word_start` is closer to orthogonal and has ~2/3 the targets.

## Route judgment
The gradient evidence supports the sharper hypothesis: an **MLM-primary, same-corruption MNTP auxiliary** is gradient-compatible with clean-Qwen's reconstruction objective and can be added without deleting MLM updates — unlike causal attention verification's whole-batch substitution. This is worth one carefully controlled construction, NOT another causal-fraction sweep.

## Proposed implementation
Build a matched trainer variant on the exact clean-Qwen recipe (same arch/tokenizer/init/seed/LR/exposure/checkpoint policy) that, on EACH batch:
1. computes the ordinary WWM-MLM loss and its gradient (full legal MLM stream retained, every word counted once);
2. computes the same-corruption MNTP auxiliary (`token_shift`) on the same masked positions, shifted to j-1;
3. adds the auxiliary with a coefficient `lambda_t` set so the auxiliary gradient norm is a fixed small fraction (candidate 0.10–0.25) of the MLM gradient norm, using running gradient-norm statistics only (no eval/AoA). Consider PCGrad-style projection only for the early-layer mild-negative regime, decided from probe statistics, not eval.

Decisive comparison: one true-100M run vs clean-Qwen `chck_100M` on the complete nine columns (AoA aggregate-only terminal readout). Promote to SuperGLUE+AoA only if a coherent column profile improves Overall; otherwise the same-stack auxiliary route is closed and the research pivots to representation/architecture (attention gating / GEGLU / layer weighting, source-grounded in GPT-BERT ablation) or tokenization, still preserving the same-window Qwen second-view corpus.

## Guardrails
- Do not reopen closed tail/mask, geometry, cluster, agreement, SWA, ordering, or contaminated mix25 routes.
- Trusted admissible best remains clean-Qwen seed43022 `chck_100M` Overall **41.34429066479573**, below visible leader 41.8.
- AoA aggregate-only; no CDI/AoA words, curves, predictions, or scores as training or selection signals.
