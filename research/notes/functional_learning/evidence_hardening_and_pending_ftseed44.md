# Evidence hardening with matched ftseed44 evaluations pending

This historical note records source-level verification and interpretation corrections for the v5 candidate while matched ftseed44 evaluations were still pending. No results were inferred from incomplete evaluations.

## 1. Source-level v5 evidence verifier

Script: `experiments/archive/functional_learning/scripts/v5_export_evidence_verifier.py`  
Output: `experiments/archive/functional_learning/data/v5_export_evidence_verifier`

The verifier reads the actual all-six same-coordinate table, official zero/Reading payloads, repaired-AutoModel SuperGLUE payloads, measured AoA manifests, and AutoModel validation records. It verifies task return codes, prediction/log/report paths, primary-metric SuperGLUE recomputation, measured AoA shape, source/table agreement, and the scientific account three levels seed62064 weight hash.

Result: `all_source_level_valid=true`, `n_errors=0`.

Recomputed Overall values from source payloads:

| model | Overall | delta vs coherent86 |
|---|---:|---:|
| coherent86 | 42.023967991315104 | 0.0 |
| dense_seed62064 `(M,M)` | 42.14909111936738 | +0.1251231280522731 |
| dense_seed62065 `(M,M)` | 42.168422702303516 | +0.14445471098841267 |
| clean_pres_seed62064 `(M,S)+KL` | 42.246412332209445 | +0.2224443408943415 |
| clean_pres_seed62065 `(M,S)+KL` | 42.23173113265801 | +0.20776314134290885 |
| exact acquisition-only `(M,S)` seed62064 | 42.20253795433653 | +0.17856996302142392 |

Direct clean64 minus exact `(M,S)` remains `+0.04387437787291759` Overall, with component deltas BLiMP `+0.14`, Supplement `+0.20`, EWoK `-0.13`, Entity `+0.01`, COMPS `+0.01`, SuperGLUE `+0.15986940085623758`, GlobalPIQA `0`, Reading `+0.005`, AoA `0`.

The verifier also repaired a small schema display issue: the evaluation-repair summary's `interface_validated` and `all_repaired_valid` are nested under `summary`; the verifier output now shows `interface_validated=true`, `all_repaired_valid=true`, and `all_negatives_stock=true`, with coherent86/dense repaired AutoModel hidden-state exact matches. The earlier validation records remain all valid for clean64, clean65, and exact `(M,S)`.

## 2. Repaired direct zero/Reading profile

Patched script: `experiments/archive/functional_learning/scripts/zero_reading_ms_direct_profile.py`  
Repaired output: `experiments/archive/functional_learning/data/zero_reading_ms_direct_profile_repaired`

The earlier zero/Reading profile had a minor null Reading-comparison subsection because it asked the zero/Reading and SuperGLUE item-evidence parser for nonexistent `pred_values`/`prev_pred_values` keys. The profile was corrected to use the actual `pred` and `prev_pred` arrays returned by `parse_reading`, filtering finite pairs.

The repaired Reading comparisons now show:

- clean64 vs exact `(M,S)`: Reading score delta `+0.005`, prediction-vector Pearson `0.9998382563300703`, mean absolute prediction difference `0.07536332613527545`, previous-prediction Pearson `0.9997350586197238`, previous mean absolute difference `0.06141692705604427`.
- clean65 vs exact `(M,S)`: Reading score delta `0.0`, prediction-vector Pearson `0.9998306476554961`, mean absolute prediction difference `0.0764168072852784`, previous-prediction Pearson `0.9997247343645003`, previous mean absolute difference `0.06218570024315063`.
- clean64 vs clean65: Reading score delta `+0.005`, prediction-vector Pearson `0.9999949901599272`, mean absolute prediction difference `0.01187593425854518`, previous-prediction Pearson `0.9999907847007538`, previous mean absolute difference `0.011319091025221302`.

The official Reading component scores and all zero/Reading component sums are unchanged; this only repairs the interpretability subsection.

## 3. Reworded paired uncertainty note

Patched script: `experiments/archive/functional_learning/scripts/direct_increment_paired_uncertainty.py`  
Reworded output: `experiments/archive/functional_learning/data/direct_increment_paired_uncertainty_reworded`

The earlier paired uncertainty numbers were correct, but the generated prose said the completed policy improvement over coherent86 was "not in doubt from this analysis." The corrected interpretation is that the fixed-coordinate higher score is factual, but broad superiority beyond the fixed item pools is not established, and clean64-vs-coherent86's conditional paired-item interval also crosses zero.

The analysis was regenerated at the same resampling scale (`n_boot_zero=600`, `n_boot_sg=600`, `n_boot_reading=200`, seed `99099`) with corrected wording. Quantitative summaries are unchanged:

- clean64 vs exact `(M,S)`: payload Overall delta `+0.04387437787291551`; bootstrap median `+0.04470931943580151`; interval `[-0.03454097016577249, +0.12907298844466833]`; positive fraction `0.865`.
- clean65 vs exact `(M,S)` context: payload Overall delta `+0.029193178321485897`; interval `[-0.05568143728161748, +0.12246470204949428]`; positive fraction `0.75`.
- clean64 vs coherent86 fixed-policy score: payload Overall delta `+0.22244434089434076`; conditional item interval `[-0.047047218699175865, +0.5370297028575998]`; positive fraction `0.925`.

The corrected scientific reading now says: completed files establish a higher fixed-coordinate score for clean64, but the paired resampling does not establish broad superiority beyond the fixed validation pools. This matches the active three-level account in `research/notes/functional_learning/scientific_account_three_levels.md`.

## 4. Matched ftseed44 evaluations pending at the time

The following evaluations were in progress when this note was written:

- coherent86 SuperGLUE ftseed44 matched comparison; planned output: `experiments/archive/functional_learning/data/matched_ftseed44_superglue/coherent86_alpha075_ftseed44/faithful_superglue_seeded_summary.json`.
- exact `(M,S)` SuperGLUE ftseed44 comparison; planned output: `experiments/archive/functional_learning/data/ms_ftseed44_superglue/densemask_sparselabel_seed62064_ftseed44/faithful_superglue_seeded_summary.json`.
- clean64 SuperGLUE ftseed44 comparison; planned output directory: `experiments/archive/relation_learning/data/clean_pres62064_ftseed44_superglue`.

No ftseed44 result was used in this note. The planned comparison script was `experiments/archive/functional_learning/scripts/ftseed_comparison.py`, with the following seed42 reference values:

- seed42 clean64 minus exact `(M,S)` SuperGLUE: `+0.15986940085623758` component points (`+0.01776326676180485` Overall units).
- seed42 clean64 minus coherent86 SuperGLUE: `+0.10199906804906078` component points.
- seed42 exact `(M,S)` minus coherent86 SuperGLUE: `-0.05787033280716969` component points.

A weaker or reversed preservation-specific SuperGLUE increment at seed44 would narrow the downstream-transfer interpretation. It would not erase the fixed-coordinate endpoint ordering or the source-level provenance.
