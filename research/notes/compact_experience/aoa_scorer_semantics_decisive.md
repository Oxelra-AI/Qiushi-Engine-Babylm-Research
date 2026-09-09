# aoa safety audit and route — What AoA=0.0 actually means (decisive artifact read)

## The 0.0 is a valid statistical result, not a crash or fallback

Read the official scorer `evaluation_pipeline/utils.py::AoAEvaluator.compute_curve_fitness`
and `compute_model_aoa`, plus the per-target AoA artifacts under
`data/full_eval/aoa_outputs/*/aoa_local_ckpts.json`.

Confirmed facts:
- Every clean-Qwen and control AoA run **fully completed**: 19 checkpoints
  (`chck_1M..chck_10M`, then `chck_20M..chck_100M`), 124,640 rows = 6,560 target
  contexts × 19 steps, valid monotone-ish mean surprisal curves, ~1000s runtime.
- `compute_model_aoa` fits a bounded sigmoid to each word's negative-surprisal curve
  over `log10(step_words+1)` and returns the log-step where surprisal crosses a
  threshold set at 50% between the uniform-chance ceiling
  (`n_subword_tokens * ln(vocab)`) and the word's own minimum surprisal.
- `compute_curve_fitness` = Pearson r between per-word `model_aoa` and child `child_aoa`,
  **gated by `p_value > 0.1 → return 0.0`**.

Therefore:
- **official_lengthmatched AoA = −0.157** is a *significant negative* correlation:
  the model's word-acquisition order is anti-correlated with the child CDI order.
- **qwen_clean_aligned AoA = 0.0** means the model→child AoA correlation was **not
  significant at p<0.1** — a genuine "no reliable developmental-order match", not a
  missing checkpoint, NaN, or fallback. The `aoa_score.json` is 16 bytes (`{"aoa":0.0}`)
  vs 31–33 bytes for nonzero arms only because the numeric value is shorter.

This resolves the measurement-status ambiguity: the measurement process is valid; 0.0
is the real value.

## Why this makes the developmental-order route well-founded and legal

The AoA column rewards a model whose per-word acquisition *order* matches the child
curve. The mechanism that moves it:
- present material so that words children acquire early cross the surprisal threshold
  at earlier checkpoints, and later-acquired words cross later, producing a **positive**
  model→child AoA rank relationship.

The aoa safety audit and route developmental first-pass (child-directed → simple → dialogue → dense prose,
with corpus-frequency/short-form-first within source) directly shapes which words are
well-modeled at early checkpoints. It uses only:
- official source labels, corpus-internal word frequency, row lexical statistics.

It never reads the CDI human curve, AoA target words, AoA predictions, or AoA scores.
So it is an AoA-safe intervention on the *training-side cause* of the acquisition-order
curve, not an evaluation leak.

## Expected signatures to check when aoa safety audit and route evaluates

1. AoA: does model→child correlation become significant and positive? Even a small
   positive r has large Overall leverage (ΔOverall = 100·Δr / 9).
2. Early-checkpoint mean surprisal by source: developmental order should lower early
   surprisal on high-frequency/child-directed words relative to baseline first pass.
3. NLP columns: the same-window pair signal (Supplement/Entity/SuperGLUE) must not
   collapse because Qwen pair rows are delayed to ~3.8M in the first pass. Compare
   same-seed vs clean qwen compliance and validity/clean qwen experiment state baselines.

## Follow-up controls the verifier correctly flagged (after aoa safety audit and route result)

The current order is source-blocked (source_rank dominates). To separate "developmental
source schedule" from "lexical difficulty":
- source-balanced easy→hard (each 1M bin keeps all sources; only within-bin difficulty tilts),
- reverse (hard→easy) control,
- source-block-only vs lexical-only,
- stratified random first-pass permutation as a null band.

These are the right next arms **if** aoa safety audit and route shows a positive AoA move without NLP loss.
