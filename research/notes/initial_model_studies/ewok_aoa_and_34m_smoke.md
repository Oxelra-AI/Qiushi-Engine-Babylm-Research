# ewok aoa and 34m smoke — EWoK resolution, AoA=0 diagnosis, 34M smoke pass

## research EWoK recovery (verified)

research loop (exec) succeeded on the EWoK source proposal. Report: `analysis/Research_Report.md`; artifacts under `analysis/`.

Findings:

- Official full EWoK is genuinely gated. The public `BabyLM-community/BabyLM-2026-Strict-Evals` bundle (revision `8d52da9424a9ff30b9e8266c4f751aba9c504233`, 175 files) contains only `evaluation_data/fast_eval/ewok_fast.zip`; no full `ewok_filtered` JSONL, Parquet, or LFS pointer.
- Upstream `ewok-core/ewok-core-1.0` (revision `34d912a608066c92e2990a0328ffc3bd9a716042`, test object `data/test/ewok-core-1.0.parquet`, 110,797 bytes) returned HTTP 401 for anonymous file, pinned-file, and dataset-server access. Metadata/README are public.
- Expected full shape: 3,809 filtered originals → 7,618 JSONL rows across 11 domains (per-domain counts recorded in `work/source_probe.json`).
- A protected `ewok-paper` release reproduces the 3,809 total but with per-domain deltas (physical-relations +10, social-interactions −8, social-properties −2), so it is NOT a validated substitute for the gated Parquet.

Interim (non-full, verified): official **fast** EWoK on `chck_100M` = **46.18%** macro accuracy, independently reproduced by `evaluation_pipeline.calculate_results_from_pred._calculate_ewok_results`. Provenance in `analysis/fast_eval_summary.json`. This must be labeled `ewok_fast_interim`, never entered in the official full-EWoK column.

Completing the official column requires authorized Hugging Face access with the EWoK terms accepted, then `python -m evaluation_pipeline.ewok.dl_and_filter`, then the standard full EWoK zero-shot run.

## AoA = 0.0 diagnosis (mechanical, not a true acquisition score)

Log `notes/wwm100m_aoa_eval.log` and evaluator `evaluation_pipeline/utils.py::compute_curve_fitness`:

- The AoA runner completed all 19 checkpoints and saved `surprisal.json` (997,130 lines) — surprisals are real, nonzero.
- CDI data loaded 504 words across ages 16-30 months. The probed word set was 328 words (176 filtered by min_context 20).
- `compute_curve_fitness` returned `Only 0 valid words for correlation`, so it returned `curve_fitness = 0.0` by the `len(model_aoas) < 3` guard.

Interpretation: AoA 0.0 is a degenerate output of the correlation stage, not a measured human-likeness score. Either (a) fewer than 3 probed words survived the model-AoA fitting/CDI join, or (b) a track/step-naming or word-overlap mismatch left the valid-word set empty. The official leaderboard treats AoA missing/degenerate as 0 in Overall aggregation, so 0.0 could still enter a strict Overall, but it should be reported as `aoa_degenerate_zero` (valid words 0) rather than a genuine capability value. The mechanism to investigate next is why `compute_curve_fitness` finds 0 valid words: sigmoid AoA fitting per word may fail for the MLM surprisal trajectories over only 19 checkpoints, or the CDI word-string join may not match the tokenizer word forms.

This is consistent with the visible leaderboard top row also reporting AoA 0, suggesting AoA 0 may be common for these MLM submissions rather than unique to this baseline. It should still be verified, not assumed.

## 34M A/B smoke: PASS

`scripts/run_34m_smoke.py` repaired (`lr_total_steps 5 → 32`). Both arms trained, saved, and reloaded:

| arm | model | params | non-embedding | note |
|---|---|---:|---:|---|
| A capacity | BertForMaskedLM | 34,151,936 | 25,763,328 | 8×512, 8 heads |
| B DeBERTa-v2 | DebertaV2ForMaskedLM | 34,467,424 | 26,603,104 | 8×480, 8 heads, p2c,c2p relative |

Summary: `data/34m_smoke_summary.json`, `notes/34m_smoke_summary.md`. Both load via `AutoModelForMaskedLM`/`AutoTokenizer`. The full-cycle trainer, epoch cycling, and checkpoint lifecycle work for both architectures at 34M. Ready for full 100M A/B training.

## Coordinate state after ewok aoa and 34m smoke

WWM100M protected baseline `babylm_fullcycle_wwm_seed42_100M`, `chck_100M`, backend `mlm`:

| column | value | status |
|---|---:|---|
| BLiMP | 55.92 | full eval |
| BLiMP Supplement | 52.03 | full eval |
| EWoK | — (fast interim 46.18) | official full gated/unavailable; fast interim only |
| Entity Tracking | 16.76 | full eval |
| COMPS | 51.66 | full eval |
| (Super)GLUE | 63.08 | official finetune, validation-accuracy mean |
| GlobalPIQA | 32.74 (par 17.48 / nonpar 48.0) | full eval |
| Reading | eye 11.68 / self-paced 3.82 | full eval subcomponents |
| AoA | 0.0 (degenerate, 0 valid words) | official runner ran; correlation stage empty |

No true official nine-column Overall yet: EWoK full is gated, and AoA 0.0 is a degenerate correlation output whose validity should be checked. If both were accepted as-is under leaderboard missing/zero conventions, the coordinate would still be a lower bound, not a competitive result.

## Next work

1. Launch full 100M A (BERT 8×512) and B (DeBERTa-v2 8×480) WWM training, matched to the protected baseline recipe, ideally concurrently on the two H100s.
2. Investigate AoA valid-word emptiness: inspect how `compute_curve_fitness` maps probed words to CDI words and whether MLM surprisal-based sigmoid AoA fitting yields <3 fitted words; decide whether AoA 0 is genuine-for-this-eval or a fixable mapping issue.
3. Keep EWoK official column null pending authorized access; keep `ewok_fast_interim=46.18` as labeled non-full evidence.
