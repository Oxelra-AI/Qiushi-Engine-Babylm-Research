# active endpoint consumption and u256 interpretation — active endpoint consumption, U256 mechanism interpretation, and post-eval tools

This note records CPU-only work done while the three managed GPU tasks continued. It does not add training or model evaluation.

## Pending Training and Evaluation

- exact deterministic scale1.75 adapter128 100M endpoint ladder training, output `training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder`.
- hardened full official-compatible evaluation for scale1.75 100M, output `data/scale1p75_100M_full_eval_hardened/`.
- U256 faithful-visibility 100M training, output `training/runs/eu_U256_legal16k_seed43022_100M/`.

The final comparison requires the completed endpoint evaluation and its authoritative summary files; incomplete outputs do not establish a score.

## Related Results and Pending Comparisons

The role-switch packet route was closed. Full SuperGLUE and AoA evaluation of the scale1.75 80M checkpoint remained pending. The conditional projection used cheap7=43.8121 and estimated that SuperGLUE >= 69.5 would cross Overall 41.8 with AoA=0. The 100M scale1.75 training was still in progress; the U256 legal16k visibility profile was available.

## U256 visibility profile completed

Script: `scripts/u256_visibility_profile.py`.

Outputs:

- JSON: `data/u256_visibility_profile/u256_visibility_profile.json`
- Markdown: `notes/u256_visibility_profile.md`
- CSVs: `data/u256_visibility_profile/recovered_suffix_by_source_legal16k.csv`, `top_recovered_suffix_rows_legal16k.csv`, `update_mass_summary_legal16k.csv`, `row_token_length_buckets_legal16k.csv`

Key facts:

- Exact legal substrate checked: base 10M SHA `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`, stream SHA `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`, tokenizer JSON SHA `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`.
- Row256 stream reconstruction matched spatial repair route status training-log batch words with 0 mismatches over 2,529 updates.
- Per 10M epoch, legal16k raw tokens are 14,664,519; spatial repair route status row256 exposes 14,294,893 active tokens; U256 exposes all 14,664,519.
- U256 recovers 369,626 tokens per 10M epoch, active-token ratio 1.025857, and 197,806 fully hidden charged words (1.9781% of words), plus 10,461 boundary-hidden token pieces.
- First 20M comparison: U256/row256 active-token ratio 1.025411, realized masked-target ratio 1.026450, masked-per-active ratio 1.001013. The 20M +0.9179 cheap7 movement is much larger than the token-count increase.
- Recovered source-tail mass is concentrated: CHILDES 223,846 recovered tokens (60.56% of all; 5.25% of CHILDES raw tokens), SimpleWiki 74,970 (20.28%; 4.43%), OpenSubtitles 38,835 (10.51%), Gutenberg 29,993 (8.11%). The compact FineWeb pair block contributes only 636 recovered tokens and qwen_pair_packed only 1,279, so U256 mostly modifies the official-source long-row tails, not the protected compact-pair mechanism.

Interpretation: U256 is a small, legal, faithful visibility repair, not new text and not a new objective. If its endpoint helps, the likely mechanism is altered credit flow from long-row tails rather than the protected compact-pair block. If it hurts or rotates competence, check EWoK material/social/quantitative and Reading movement against these recovered source tails.

## U256 recovered-tail quality profile completed

Script: `scripts/u256_tail_quality.py`.

Outputs:

- JSON: `data/u256_tail_quality/u256_tail_quality.json`
- Markdown: `notes/u256_tail_quality.md`
- CSVs: `data/u256_tail_quality/tail_quality_by_source.csv`, `top_recovered_tail_noise_rows.csv`

Key facts:

- 15,117 rows have recovered row-tail text, with 369,626 recovered tokens and 197,806 recovered full words.
- Heuristic cleanish rows: 5,639 / 15,117 (37.30%). Encoding-noise rows: 91 (0.60%). All-caps/glued rows: 1,115 (7.38%). CHILDES-like rows: 10,430 (69.00%); subtitle-like rows: 1,155 (7.64%); list/TOC-like rows: 856 (5.66%).
- By source: CHILDES tails are dominant but only 20.24% cleanish under the heuristic because transcripts are marked and speaker-coded; SimpleWiki tails are 77.16% cleanish; OpenSubtitles and Gutenberg contain more all-caps/glued/list/encoding artifacts.

Interpretation: the endpoint should not be read as a uniform +2.59% token exposure. U256 exposes concentrated, heterogeneous tails: a lot of CHILDES transcript continuation, substantial clean SimpleWiki prose, and smaller but noisier subtitle/Gutenberg tails. This can plausibly help Supplement/GlobalPIQA/BLiMP early, but it can also fund EWoK/Reading tradeoffs if noisy/list/transcript tails dominate credit.

## Post-evaluation analyzers prepared

Scale1.75 analyzer:

- Script: `scripts/scale1p75_post_eval_analyzer.py`
- Intended input after endpoint evaluation completes: `data/scale1p75_100M_full_eval_hardened/summary/scale1p75_100M_full_eval_hardened_summary.json`
- Output: `data/scale1p75_100m_post_eval_analysis/`
- Function: verify nine-column arithmetic, endpoint exposure/first-loss/ladder checks, collator validation, patch candidate staged payload if `official_overall` was absent, and run the validated saved-prediction item-flip comparator versus spatial repair route status 100M.

U256 analyzer:

- Script: `scripts/u256_post_eval_analyzer.py`
- Intended input after the prepared U256 full evaluator is eventually run: `data/u256_100M_full_eval_hardened/summary/u256_100M_full_eval_hardened_summary.json`
- Output: `data/u256_100m_post_eval_analysis/`
- Function: same as above, but expects U256 first loss 9.826857208144903 and links interpretation to `notes/u256_visibility_profile.md`.

All three active endpoint consumption and u256 interpretation scripts parse by AST. The post-evaluation analyzers were not run because their required full-evaluation summaries are not present yet.

## Immediate next use

1. After endpoint evaluation completes, read `data/scale1p75_100M_full_eval_hardened/summary/scale1p75_100M_full_eval_hardened_summary.{json,md}` and the pristine collate summary. If present and valid, run:

```bash
python -B experiments/archive/frontier_consolidation/scripts/scale1p75_post_eval_analyzer.py
```

This is CPU-only and will produce endpoint item-family analysis. If Overall is above 41.8, route to independent verification/reproducibility/submission-readiness work rather than claiming completion from a single artifact.

2. After U256 training completes, verify U256 scientific metrics first. If the 100M endpoint is complete, launch the already prepared full evaluator:

```bash
python -B experiments/archive/frontier_consolidation/scripts/full_eval_u256_100M_hardened.py
```

After it finishes, run:

```bash
python -B experiments/archive/frontier_consolidation/scripts/u256_post_eval_analyzer.py
```

3. Do not launch another expensive training route until scale1.75 and/or U256 endpoint evidence has actually changed the route judgment or freed resources. If extra CPU work is needed, focus on non-inference source/mechanism analyses that sharpen interpretation of delivered endpoint scores.
