# official aggregation and aoa convention resolved — Official aggregation + AoA convention resolved from the live leaderboard source

Purpose: resolve like-for-like comparability from the **current server-side source**, so a recomputed 8005-row reinvest AoA can be placed against the visible 41.8 leader without mixing conventions. All content below is read directly from the live leaderboard Space `BabyLM-community/BabyLM-Leaderboard-2026` (sha `c84d8c10100149acc04cbaf65ee1f20a7102a618`, lastModified 2026-07-09) at `resolve/main/`, and cross-checked against the local eval repo commit `6f825c291e2c4c78ad33b1935fd64d45f52642dc`.

## 1. Official Overall aggregation (read from `src/leaderboard/read_evals.py`, `to_dict`)

For strict / strict-small (non-multilingual):

```
NLP_BENCHMARK_KEYS   = ("blimp","blimp_supplement","ewok","entity_tracking","comps","glue")
global_piqa          = macro-avg of global_piqa_parallel & global_piqa_nonparallel  (one column)
HUMAN_LIKE_KEYS      = ("reading","aoa")

nlp_vals     = [blimp, blimp_supplement, ewok, entity_tracking, comps, glue] + [global_piqa]   # 7 values
human_vals   = [reading, aoa]                                                                    # 2 values
overall_vals = nlp_vals + human_vals                                                             # 9 values

Overall Average   = sum(overall_vals)/9
NLP Average       = sum(nlp_vals)/7
Human-like Average= sum(human_vals)/2
```

=> Overall = unweighted mean of the SAME nine columns we have been using. Our local nine-column arithmetic (BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, SuperGLUE, Reading, AoA) matches the official Overall formula. `Reading` is itself the mean of spr+rt; `GlobalPIQA` is the mean of parallel+nonparallel; both already folded before the 9-way mean. CONFIRMED.

## 2. AoA column value (read from `read_evals.py` `_get_benchmark_score` + `eval_submission.py`)

- `eval_submission.evaluate_submission` stores `processed_results_dict["aoa"] = {"aoa": <float>}` where `<float>` is the single AoA score computed by the pipeline (the p>0.1 -> 0.0 gated correlation, in raw correlation units).
- `read_evals._get_benchmark_score("aoa")` = `mean(values in the aoa dict) * 100` = `aoa_float * 100`.

=> The displayed **AoA leaderboard column = aoa_float × 100**. This exactly matches how our local nine-column arithmetic treated AoA (e.g. compact_core raw -0.126873 significant -> -12.687; reinvest / clean-Qwen p>0.1 -> 0.0). CONFIRMED like-for-like, including the gate.

Leaderboard evidence consistent with ×100: strict-small rows show AoA values such as -15.1, +22.9, -12.23, +24.69, -20.47 — i.e. raw correlations ~-0.15 / +0.23 scaled by 100.

## 3. AoA row-count = 8005 is the HARD official requirement (read from `check_validity.py`)

- `src/display/utils.py`: `AOA_SIZE = 8005`.
- `check_validity.is_valid_full_predictions` line ~218: `if len(revision_results) != AOA_SIZE: Error("... contains N datapoints. Was expecting 8005")` for every AoA checkpoint (`NUM_CHECKPOINTS_10M = 19` for strict-small).
- Direct count of `cdi_childes.json`: min_context=0 -> 8005 contexts (504 words); min_context=20 -> 6560 contexts (328 words).

=> Our historical local AoA helper hardcoded `min_context=20` -> 6560 rows. That value would be **rejected by the official validator** and is NOT the server convention. The endpoint's previously-reported AoA=0.0 (and clean-Qwen 41.3443's AoA=0.0) were computed under the wrong (6560-row) convention. They must be recomputed at min_context=0 / 8005 rows. The pending recomputation uses this convention for seed43022 and seed43122.

## 4. Two additional official gates that our submission must satisfy (read from read_evals.py)

- `aoa_surprisals`: AoA counts only if raw surprisals are uploaded under key `aoa_surprisals`; otherwise `results["aoa"]=0.0`. (Our AoA=0 endpoints are unaffected in value, but a real submission must upload the 8005-row surprisals.)
- `entity_tracking_filtered`: Entity counts only if scored from the "nothing"-filtered predictions (key `entity_tracking_filtered`); legacy Entity is zeroed. Our full evaluator already used filtered Entity (reinvest Entity 27.75 is nonzero and credible), but the submission JSON must carry the `entity_tracking_filtered` marker.

## 5. Comparability Verdict

- 8005-row / min_context=0 IS the live server convention (hard-enforced by the validator). The visible 41.8 leader `wwm_curriculum_simplification_40k` was scored server-side under this same pipeline, so once reinvest AoA is recomputed at 8005 rows, the +0.2868 comparison is like-for-like.
- The whole +0.2868 lead still rests on reinvest AoA staying non-significant (p>0.1 -> 0.0) under the 8005-row count. If the 8005-row fit turns reinvest AoA significantly negative, Overall drops sharply (reinvest endpoint review sensitivity: raw -0.05 -> ~41.53; -0.10 -> ~40.98). So the recomputed AoA for seed43022 AND seed43122 remain the decisive evidence.

## 6. compact_repeat_core SuperGLUE completed — substrate-vs-view attribution CLOSED

Full SuperGLUE means (finetune):
- clean-Qwen        : 70.3086
- compact_repeat_core: **70.6244** (BoolQ 67.339, MultiRC 67.492, RTE 68.345, WSC 67.308, MRPC 85.294, QQP 78.466, MNLI 60.126)
- compact_view_core : 68.9012

Interpretation:
- The FineWeb **repetition substrate** does NOT lose SuperGLUE (+0.32 vs clean-Qwen). So the compact_view_core SuperGLUE loss (-1.41 vs clean, -1.72 vs repeat) is caused by the **compact-view semantic transformation** (documented role/modality/relational loss), not by displacing official source words.
- This mirrors the Supplement anatomy in the opposite direction: Supplement loss was mostly substrate (repeat 59.96 << clean 62.84), SuperGLUE loss is mostly view.
- Reinvest recovered SuperGLUE to 71.3811 (+1.07 vs clean): reinvested source diversity more than offsets the compact-view SuperGLUE cost. This is a genuine, now causally-separated, benefit of the reinvestment factor, not an artifact.
