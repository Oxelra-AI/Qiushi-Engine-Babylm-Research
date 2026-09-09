# initialization parity and interaction interpretation — Initialization parity of the DeBERTa positional-ablation coordinate

## Scientific Motivation

The earlier analysis experiment (running on H100) estimates the
four-cell interaction

    I = (compact - repeat)_full_DeBERTa  -  (compact - repeat)_no_disentangle_abs

where full = `pos_att_type=[p2c,c2p]` and no_disentangle_abs = `relative_attention=True, pos_att_type=[]`
(both keep absolute position embeddings and encoder `rel_embeddings`).

Before the terminal results arrive, the load-bearing question is: does the `no_disentangle_abs`
architecture equal "full DeBERTa minus only the c2p/p2c score terms" at initialization, or is it a
differently-initialized model? If the latter, the interaction is still valid but its causal claim is
narrower.

## Evidence (CPU-only, reproducible)

Script `scripts/initialization_parity_probe.py` instantiates each variant through the
exact compliant tokenizer retrain status/COMPACT_EXPERIENCE trainer `build_model` under the exact seed path (seed 43, extra_init_seed 43022,
train_rng_seed 43023) and compares all same-named tensors and initial logits.

Result (`data/initialization_parity_probe/initialization_parity_probe.json`):

- `full` vs `no_disentangle_abs`, same seeds: **140 common tensors, 53 nonexact**. The divergence is
  NOT confined to the removed modules; it propagates through attention QKV (24/48 mismatched),
  intermediate (8/16), attention output (8/32), layer output (8/32), embeddings (2/4), MLM head (2/7),
  and encoder rel_embeddings (1/1). Cause: omitting the optional `pos_key_proj`/`pos_query_proj`
  modules shifts the torch construction-time RNG stream for every subsequently built tensor.
- Initial-logit difference on compact-stream texts: mean_abs **0.494**, rms **0.620**, max **3.45**.
- `c2p_only` vs `p2c_only`, same seeds: **140/140 bit-identical** (equal module count → same RNG
  stream), equal parameter count 32,620,384. This is a clean contrast.
- Common-copy repair: copying all 140 common tensors from `full` init into `no_disentangle_abs` yields
  140/140 exact, and the resulting logits match `full` within mean_abs **0.0024**, max **0.018** — the
  only residual difference is the removed positional-score path itself, exactly as expected.

## Scientific interpretation (independent_review-verified)

.

1. Within each architecture, both data arms start from the identical initialization (same seeds, same
   architecture per arm), so any *additive* initialization-level effect cancels inside each
   compact-minus-repeat difference. The four-cell interaction therefore remains a **valid
   data-treatment interaction across two realized architecture-plus-initialization coordinates**. The
   0.494 initial-logit gap is not an additive bias in I.

2. What survives: because training is nonlinear, `Δ(W_A) = S(W_A,C)-S(W_A,R)` need not equal
   `Δ(W_B)`, i.e. an initialization×data interaction. Also bundled: capacity/parameter-count change
   (34,467,424 vs 30,773,344, −3,694,080), attention-score composition, gradient paths, optimization
   basin/scale. So the running result **cannot claim** "c2p/p2c score terms are necessary" as a pure
   information channel — attenuation would implicate the whole removed package plus init/capacity.

3. Correct disambiguating follow-up (only if the delivered result is important but ambiguous): retrain
   `no_disentangle` compact/repeat initialized by copying all common tensors from the SAME full
   DeBERTa init while omitting positional projection tensors, so B is exactly A-minus-score-terms.
   This removes the RNG-stream confound; capacity/parameterization differences still remain.

4. `c2p_only` vs `p2c_only` is the clean lens for the narrower directional question (does the retained
   disentangled channel direction modulate the compact effect?), but it cannot replace
   full-vs-no-disentangle for the "score-term package needed?" question, since both single-term arms
   keep one channel.

## Reading the delivered earlier analysis result

- Survival = compact-minus-repeat comparable in B, I near zero with a narrow interval on stable
  families (`cheap6_no_GlobalPIQA`, `cheap5_no_GlobalPIQA_Reading`, `EWoK_plus_Entity_sum`,
  Supplement, Entity, COMPS) at both 80M and 100M. Same sign alone is insufficient; magnitude must
  exclude material attenuation. Conditional on this init; common-copy replication strengthens it.
- Attenuation/collapse = report as bundled architecture-coordinate effect, not score-term necessity.
- Loss, GlobalPIQA-only movement, and single best-checkpoint stories are not sufficient support.

## Boundary

CPU-only fresh-initialization probe. No training launched, no selected official-compatible
evaluation, no SuperGLUE, no AoA, no upload, no leaderboard submission. The two earlier analysis H100 tasks
were not polled or disturbed.
