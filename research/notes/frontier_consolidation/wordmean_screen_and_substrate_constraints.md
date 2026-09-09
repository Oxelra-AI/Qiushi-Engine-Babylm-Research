# wordmean screen and substrate constraints — Word-mean MLM screen launched, substrate-constrained, minfreq50 fallback prepared

## Prior Experimental Evidence

The mature within-legal-tokenizer clean-vs-reinvest vector (delivered earlier analysis) is decisive
and positive: compact-view reinvestment survives the spatial repair route status legal tokenizer coordinate and
becomes strongly beneficial at maturity (70M Δmean7 +1.2921, 80M +1.3464), with mature EWoK
relation AND property both positive. So the compact-view reinvestment data mechanism is a
validated scientific core; the remaining gap to 41.8 is a general legal representation/
optimization deficit, not the data mechanism. legal40k both seeds also miss 41.8
(mean 40.780), a clean support/segmentation trade-off. Distinct candidate factors are word-mean MLM + support-floored minfreq50, and leader-shape depth + optional minfreq25.

## Word-Mean MLM Screen

The earlier analysis objective changes only MLM loss normalization: selected token cross-entropies
are averaged within each selected WWM group, then over selected groups (tokenizer-invariant
whole-word credit). The 80M seed43022 screen is running. Its dependent
70M/80M cheap-column evaluation is pending (waits for checkpoints then evaluates BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading).

### CPU-Only Substrate Analyses

1. **Credit-scale is not pure relative reweighting** —
   `data/wordmean_credit_scale_analysis/`. On matched selected WWM batches,
   per-token weight ratio word/token is mean_k/k: 1-piece 1.459, 2-piece 0.730, 3-piece
   0.486, k>=4 down to 0.05-0.37. Selected mean tokens/group 1.4594. Uncorrelated
   logit-gradient weight RMS scale ~1.098 (p10 1.095, p90 1.102). So an unchanged learning
   rate does not isolate relative credit from update-scale effects.

2. **Actual-gradient smoke** — `data/wordmean_gradient_scale_smoke/` (CPU 4-row,
   two batches; GPU version cancelled to protect the running screen). Same init, same
   masks/dropout seed for token-mean and word-mean backward: parameter-grad norm ratio
   word/token ~1.063, cosine only ~0.910. This confirms word-mean shifts both the scale and
   direction of the update, not only relative token weight.

3. **Reweighting profile** — `data/wordmean_reweighting_profile/` (79 sampled
   batches, 453,739 selected groups). Word-mean shifts credit share toward the highest-
   frequency 1-piece function words (the/and/a/to/of; +21.8% share) and away from
   multi-piece content: has_upper -7.9% share, has_digit -2.2%, bpe_len4 -2.7%, dates/years
   (2017/2022) and names/CHILDES speaker codes strongly down (bpe_len 4-8+ ratio 0.14-0.37).
   cue::mental_social_dialogue +1.9%, spatial +1.1%, causal +0.6% (these are function-word
   heavy). **The legal deficit (binding/filler-gap syntax, physical/material dynamics,
   QA-congruence) is largely carried by content words that word-mean DEmphasizes.** So the
   objective is not obviously aligned to the deficit and could amplify function-word credit.

### Decision rule for the pending word-mean vector

Continue word-mean to a 100M/full nine-task endpoint ONLY if 70/80M cheap columns beat the
existing spatial repair route status token-mean reinvest trajectory (70M mean7 42.6086, 80M 42.9486) broadly and
plausibly enough to recover ≥~4.9 summed official column points after SuperGLUE/AoA.
Given the substrate analyses, a scale-aware or LR-adjusted follow-up may be needed before
attributing any gain to credit allocation. If flat or worse, STOP the objective — do not
retune — and switch to the prepared support-floored representation fallback. Do NOT erode
the validated compact-view reinvestment mechanism.

## Prepared dormant fallback: support-floored minfreq50 tokenizer screen

Prepared the legal support-floored minfreq50 tokenizer and preflighted a one-seed
80M screen that changes ONLY the tokenizer relative to the spatial repair route status legal 16k reinvest recipe:
- Tokenizer: `data/supportfloor_tokenizers/legal_byte_bpe_40k_minfreq50`, vocab
  19,609, tokenizer.json SHA `9900f42b392fb69dd55c9c9fd7f539da09b3a33487e4e3542f9b953f48310922`,
  special IDs unk0/bos1/eos2/pad3/mask4, trained only on the exact 10M pool.
- Launcher: `scripts/train_minfreq50_supportfloor.py` (preflight OK, hashes matched:
  train 3dd19f09, pool 215944, tok 9900f42b). Full-batch trainer (no accum confound).
- Scientific factor: minfreq50 keeps ~25% of 40k's segmentation savings with only a 3.1%
  low-support tail and +1.5M params (clean control trained and representation frontier frontier). It tests whether a support floor
  repairs the 16k-vs-40k trade-off (Supplement/EWoK gain without GlobalPIQA/Entity collapse)
  while preserving compact-view reinvestment. This is genuinely distinct from minfreq25
  and from leader-shape depth.

This fallback is NOT launched. It should only run after the word-mean cheap vector is read,
and only as the next single expensive experiment if word-mean does not deserve continuation.

## Analysis Assets
- `scripts/wordmean_credit_scale_analysis.py`
- `scripts/wordmean_gradient_scale_smoke.py`
- `scripts/wordmean_reweighting_profile.py`
- `scripts/compare_wordmean_trajectory.py` (route surface; not-ready until eval lands)
- `scripts/wordmean_subtask_interpreter.py` (UID layer; not-ready until eval lands)
- `scripts/wait_and_eval_wordmean.py` (compares completed results)
- `scripts/train_minfreq50_supportfloor.py` (dormant fallback launcher)

## Discipline
Best fully legal endpoint remains 41.2578 (< 41.8). Old 42.033 non-submittable. No new 100M
endpoint, corpus/tokenizer change, or packaging until the word-mean cheap vector selects
continue/stop, one intervention at a time.
