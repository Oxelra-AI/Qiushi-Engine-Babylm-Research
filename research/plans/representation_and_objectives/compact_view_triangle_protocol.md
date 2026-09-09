# earlier analysis compact-view triangle protocol

## Purpose

The current post-SOTA scientific question is why compact-view reinvestment improves data-efficient pretraining. The matched triangle is designed to separate three mechanisms on the trusted DeBERTa-v2 8x480 / baseline16k / WWM fixed recipe:

1. **Semantic density plus reinvested breadth**: compact rewrites improve learning because they shorten redundant text and let the 10M word budget include more distinct sources. If this dominates, a corpus that preserves the same source and rewrite marginals but breaks source-own-rewrite adjacency should stay near the compact-view arm.
2. **Source-own-rewrite local consolidation**: presenting an original source beside its own faithful compact rewrite supplies a short-window alignment signal that helps the model consolidate meaning across surface forms. If this dominates, the adjacency-broken arm should move toward the repeat arm while the compact-view arm stays high.
3. **Repetition or generic breadth**: if repeat-reinvest approaches compact-view, the compact rewrite transformation itself is not the main effect; reinvested additional source exposure or endpoint/seed dynamics carry most of the movement.

The triangle is not a leaderboard search. It is a mechanism experiment whose answer should predict a different source or model family.

## Arms and fixed recipe

All runs use the same COMPACT_EXPERIENCE masking-curriculum trainer, DeBERTa-v2 masked LM, hidden size 480, 8 layers, 8 heads, ffn_mult 4, batch 256, seq 256, AdamW LR 0.001, warmup 0.06, weight decay 0.01, WWM mask probability 0.15, checkpoint every 1M words, max exposure 100M, base seed 43, extra init seed 43022, train RNG seed 43023 unless a tie-breaking seed is explicitly queued.

Known reference arm:
- `compact_view_reinvest`: trained in companion analysis at `experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022`, full Overall 42.086785719138156 in compact core joint visibility audit / note 21; official-coordinate old-tokenizer counterpart also has seed43022 42.0331347900748 and seed43122 41.24823958912208.

Missing triangle arms to train in companion analysis:
- `compact_repeat_reinvest`: exact repeat counterpart from `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_repeat_compact_reinvest_100M.jsonl`.
- `adjbreak_reinvest`: source and compact rewrite multisets preserved, source-own-rewrite adjacency broken within close domain/length strata. Existing 10M control preserves the row word sequence and has full pair visibility 0.993; the 100M stream must be written and hashed before training.

## Seed-spread calibrated effect bands

The documented same-recipe inherited-tokenizer compact-view reinvest spread is large: seed43022 Overall 42.0331347900748 versus seed43122 Overall 41.24823958912208, spread **0.7848952009527181 Overall**. The same comparison over the seven non-SuperGLUE/AoA columns has official full-seven spread **0.8468886294750527**, while the earlier fast equal7 spread was **1.355714285714285**. Therefore:

- A view-adjbreak gap below **0.785 Overall** cannot settle the mechanism by itself; it queues a second seed for the closer or tie-bearing arm rather than being read as evidence that rewrite marginals alone suffice.
- If only fast seven-column scores are available, a gap below **0.85** is clearly too small, and a gap below **1.36** is not strong enough for a final mechanism interpretation without either full scoring or a second seed.
- A view-adjbreak gap above the calibrated band can support source-own-rewrite consolidation only if the broad-competence profile below is intact.

Tie-breaking rule: if `adjbreak_reinvest` lands close to `compact_view_reinvest`, repeat `adjbreak_reinvest` with extra_init_seed 43122 / train_rng_seed 43123 before deciding that source-own adjacency is not load-bearing. If `compact_repeat_reinvest` lands close to `compact_view_reinvest`, repeat `compact_repeat_reinvest` first. If both controls are far below compact-view but close to one another and broad competence is intact, source-own compact second-view consolidation is the leading explanation and the next work should test it in a different model/data coordinate rather than running more same-coordinate seeds immediately.

## Broad-competence profile for the adjacency-broken control

The spanbreak failure showed that a broken-control corpus can collapse because it is generically incoherent, not because it selectively removes the intended mechanism. Therefore an adjbreak drop localizes to lost source-own-rewrite consolidation only if broad competence is not broadly damaged.

Read the adjbreak profile in this order:

1. **Training dynamics**: loss curve should remain near the repeat and compact-view recipe scale; checkpoint count and word exposure must match.
2. **Broad language columns**: BLiMP, Supplement, COMPS, Reading, and the nonparallel half of GlobalPIQA should not all move downward together by a large amount relative to repeat-reinvest.
3. **Targeted columns**: if EWoK, Entity, and relation/state-sensitive subsets lose while broad columns remain similar, the signal is compatible with lost cross-view consolidation.
4. **Generic incoherence outcome**: if adjbreak is much worse across nearly every column, especially BLiMP/Supplement/COMPS/Reading, the control is too destructive to identify source-own-rewrite adjacency. Then redesign a softer distance-broken control rather than interpreting collapse as mechanism proof.

## Decision-changing minimum work

Before the H100 runs:
- Freeze protected endpoint/data references.
- Verify `compact_repeat_reinvest` has no trained seed43022 arm.
- Materialize and hash the `adjbreak_reinvest` 100M stream.
- Dry-run the two training commands with exact hashes and empty output directories.
- Post to companion analysis: companion analysis owns the matched DeBERTa triangle; companion analysis should test one leading explanation in a different data source or model family, preferably a compact-view versus repeat contrast on a GPT-like or different-scale model at 20M/40M first.

After the two trainings finish:
- Run the same cheap official-compatible seven-column readouts, plus training-dynamics extraction.
- Use full official-compatible SuperGLUE/AoA only for a promising or mechanism-settling arm.
- Compare against `compact_view_reinvest` seed43022 with the calibrated bands above, not as a one-seed verdict.
