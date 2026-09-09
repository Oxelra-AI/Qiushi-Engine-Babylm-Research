# official coordinate and seed43122 — Official-coordinate endpoint protection and seed43122 robustness

This analysis addresses two endpoint-protection questions for compact_view_reinvest seed43022 (local Overall 42.0868) and reads the completed seed43122 replication.

## 1. Pristine official coordinate reconstruction

Built a fresh, unmodified upstream evaluation coordinate to test whether the
earlier analysis/035 EWoK size mismatch is a local evaluator defect or an upstream inconsistency.

- Fresh code clone at the exact pinned commit `6f825c291e2c4c78ad33b1935fd64d45f52642dc`
  (matches `git ls-remote ... main`); scoped code diff/status against
  `strict/evaluation_pipeline`, `strict/scripts`, `strict/README.md` are all empty →
  official code is clean/unmodified.
- Official Strict eval HF dataset revision `8d52da9424a9ff30b9e8266c4f751aba9c504233`
  (BabyLM-community/BabyLM-2026-Strict-Evals, lastModified 2026-04-22); EWoK source
  `ewok-core/ewok-core-1.0` revision `34d912a608066c92e2990a0328ffc3bd9a716042`
  (parquet sha256 `4dbc0812bc21173c75f372c9c39078b88b77a91508df8e6c82f6c6853a4e1ac1`).
- Regenerated `ewok_filtered` from the pristine checkout's own `dl_and_filter` logic
  (vocab + nltk punkt/punkt_tab downloaded into isolated local storage), then compared counts.

### EWoK count decision (`ewok_regen_and_compare.json`)

| domain | collator const | pristine gen | LOCAL (INITIAL_MODEL_STUDIES) | raw×2 | filtered×2 |
|---|---:|---:|---:|---:|---:|
| agent-properties | 2210 | 2210 | 1846 | 2240 | 2210 |
| material-dynamics | 770 | 770 | 770 | 1560 | 770 |
| material-properties | 170 | 170 | 130 | 248 | 170 |
| physical-dynamics | 120 | 120 | 100 | 150 | 120 |
| physical-interactions | 556 | 556 | 436 | 560 | 556 |
| physical-relations | 818 | 818 | 818 | 830 | 818 |
| quantitative-properties | 314 | 314 | 276 | 380 | 314 |
| social-interactions | 294 | 294 | 294 | 350 | 294 |
| social-properties | 328 | 328 | 308 | 370 | 328 |
| social-relations | 1548 | 1548 | 1548 | 1570 | 1548 |
| spatial-relations | 490 | 490 | 140 | 490 | 490 |
| **total** | **7618** | **7618** | **6666** | — | **7618** |

**Decision:** pristine-generated EWoK counts exactly equal the collator constants and
the freshly recomputed vocab-filtered source counts (total 7618). The LOCAL INITIAL_MODEL_STUDIES
`ewok_filtered` is undercounted on 7 domains (total 6666), most severely
`spatial-relations` 140 vs 490. So:

- The EWoK mismatch is a **stale/corrupt local data copy in initial_model_studies**, NOT an upstream
  code/data inconsistency and NOT a defect in the endpoint model.
- The pristine official coordinate is internally consistent (code + data + collator).
- **The compact reinvest full eval summary EWoK score 53.67 was computed on the undercounted local 6666-row data**
  (its recorded `data_path` = `evaluation_data/full_eval/ewok_filtered`), so the EWoK
  column of the 42.0868 Overall must be re-scored on the official 7618-row data.

## 2. AoA min_context discrepancy (carried from aoa mincontext discrepancy audit, confirmed)

- Official AoA: `min_context=0` → 504 CDI words, 8005 contexts (= collator AOA_SIZE).
- Inherited helper hardcoded `min_context=20` → 328 words, 6560 contexts (= compact reinvest full eval summary).
- compact reinvest full eval summary AoA=0.0 was gated by p>0.1 over only 203 valid words on the non-official
  6560-context subset; the official 8005-context AoA is being recomputed on the trained
  seed43022 ladder (managed task `s35_t26_tool1`, running, no retraining).
- The unmodified official collator sets `aoa_surprisals` to null for the 6560-row file,
  confirming the previous submit_ready_aoa flag is invalid under the official path.

## 3. Two decisive re-evaluations launched (minimum cost, reuse trained ladder)

- `s35_t26_tool1` (running): official min_context=0 AoA over the 19-checkpoint seed43022
  ladder → corrected AoA leaderboard score.
- `s36_t41_tool1` (launched): re-score EWoK on the pristine 7618-row `ewok_filtered`
  reusing seed43022 chck_100M with the identical compact reinvest full eval summary official invocation
  (`sentence_zero_shot.run --task ewok --batch_size 64 --min_temperature 1.0
  --save_predictions`). Output → `official_ewok_reeval/official_ewok_reeval.json`.

Neither retrains or regenerates the corpus; both only correct the two non-official
columns of the same endpoint on the pristine coordinate.

## 4. seed43122 replication (delivered)

Independent-seed reinvest training (seed43122, 100M exposure, rc=0) + fast no-AoA screen:

| target | BLiMP | Supp | EWoK | Entity | Ent_full | COMPS | GPIQA | Reading | equal7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reinvest seed43122 | 65.75 | 63.20 | 49.36 | 26.68 | 26.29 | 51.54 | 35.135 | 8.865 | **42.933** |
| reinvest seed43022 (companion analysis) | 66.63 | 66.40 | 53.09 | 28.07 | 27.75 | 51.97 | 35.62 | 8.24 | **44.289** |
| core seed43022 (companion analysis) | 66.93 | 65.60 | 51.55 | 27.30 | 27.85 | 52.18 | 35.135 | 8.25 | 43.849 |

**seed43122 − seed43022 reinvest**: equal7 −1.356, EWoK −3.73, Supplement −3.20,
Entity −1.39, BLiMP −0.88, COMPS −0.43, GlobalPIQA −0.485, Reading +0.625.

**Robustness reading:** the +0.29 margin over the visible leader was carried on a single
seed. On the fast no-AoA surface the replication seed is materially weaker (equal7
−1.356), and is even below the core seed43022 fast surface. seed43122 relative to
COMPACT_EXPERIENCE clean-Qwen full is roughly flat/negative on the fast columns (BLiMP −1.09,
EWoK −0.83, COMPS −0.24, GlobalPIQA −1.485; Entity +0.92, Supplement +0.36,
Reading +1.105). This indicates the seed43022 endpoint is at the favorable end of
initialization variance rather than a stable, seed-robust SOTA gain.

## 5. Provisional corrected-Overall envelope for seed43022

The compact reinvest full eval summary nine-column Overall 42.0868 uses local-EWoK 53.67 and non-official AoA 0.0.
Two corrections are pending:

- EWoK: official 7618-row score replaces 53.67 (direction/magnitude unknown until
  `s36_t41_tool1` returns; the undercount removed ~952 rows concentrated in
  spatial-relations, physical-interactions, agent-properties — could move EWoK up or down).
- AoA: official min_context=0 score replaces 0.0 (could be +, −, or 0; each column is
  1/9 of Overall, so a swing of ±5 AoA points moves Overall by ±0.56).

Because both corrections and the seed gap are individually large enough to erase the
+0.29 margin, **42.0868 must be treated as provisional** until the pristine-coordinate
EWoK + official AoA re-evaluations return and are recombined into a single corrected
nine-column Overall. If the corrected seed43022 Overall drops below the visible leader,
this route is not a confirmed SOTA and the decision returns to Lead/direction rather
than a new local repair branch.

## Artifacts

- `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/pristine_official_coordinate_audit.json`
- `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/ewok_regen_and_compare.json`
- `experiments/archive/representation_and_objectives/data/official_ewok_reeval` (pending `s36_t41_tool1`)
- `experiments/archive/representation_and_objectives/data/aoa_discrepancy_audit/aoa_mincontext_discrepancy_audit.json`
- seed43122: `experiments/archive/representation_and_objectives/data/compact_reinvest_seed43122_fast/compact_view_reinvest_seed43122_fast_summary.json`
