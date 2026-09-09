# Corrected-tokenizer interpretation toolkit

## State at step close
- Both earlier analysis corrected-tokenizer retrains are complete and verified as compliant training artifacts (100M exposure, 100 checkpoints, complete AoA ladder, tokenizer SHA `4a95a2a2ead21813a409c5ad7c2c008fbf418dcf3e0dc17bd970bd9ce25ea738` at chck_1M and chck_100M, stream SHA `3dd19f09...`, example-order seed 43, exit 0). Final losses (health only): seed43022 2.571291208267212, seed43122 2.6958017349243164. Record: `data/strictsmalltok_training_completion/strictsmalltok_training_completion_summary.json`.
- Two full official evaluations are running on patched pristine-code wrappers: `s51_t41_tool1` (seed43022, GPU0) and `s51_t42_tool1` (seed43122, GPU1). Old EWoK full margin atlas `s49_t23_tool1` also running. No corrected endpoint score exists yet.
- Public Strict-Small leader unchanged: `wwm_curriculum_simplification_40k` Overall 41.8, 122 entries (refreshed `2026-08-30T00:43:20Z`). Threshold in comparison scripts is current.

## What to do the moment corrected collations arrive
1. Collect `s51_t41_tool1` and `s51_t42_tool1`. Read controller summaries, pristine collate summaries, EWoK prediction counts (expect 7,618), AoA `row_count_values` [8005], collated SHAs, and any failed-command logs. If a wrapper column failed, repair only that column without changing model/data/tokenizer/seeds/recipe/coordinate.
2. Run `python3 -B scripts/compare_corrected_tokenizer_two_seed_results.py` → corrected two-seed vectors, margins over 41.8, per-column seed deltas, corrected-vs-inherited movement.
3. Run `python3 -B scripts/corrected_subtask_localization.py` → column + SuperGLUE/Entity/GlobalPIQA/Reading subtask deltas vs inherited.
4. Run `python3 -B scripts/corrected_endpoint_interpretation.py` → scientific reading + next_action_class, integrating leaderboard, tokenizer surface, training completion, relation dynamics.

## Interpretation assets already established (CPU-only, no GPU used)
- Exact tokenizer surface: `data/tokenizer_surface_contingency/`. Legal tokenizer is shorter on the allowed 10M pool (ALL new/old ratio 0.9919) but lengthens EWoK (1.0322, old-only occ 3.92%; social-relations 1.0621/7.40%), COMPS (1.0107), GlobalPIQA (1.0071/1.0103), shortens SuperGLUE (0.9959). Concrete-word fragmentation handles: screwdriver, hates, landlord, crocodile/octopus/hippo, cupboard/backpack/baking. Overall eval-text ratio ~1.0000.
- MLM loss does NOT select the winning seed. Old-tokenizer focused EWoK relation advantage forms late (66.18 vs 30.56 at 100M); corrected-tokenizer focused EWoK is far less polarized: 50M 40.14 vs 56.60, 70M 51.54 vs 45.39, 80M 51.90 vs 46.47, 100M 48.46 vs 45.93; corrected 100M seed-delta correlation with old endpoint ≈ -0.067. `data/relation_phase_dynamics_summary/relation_phase_dynamics_summary.json`.
- Item-level tokenizer-length features have ~zero correlation (max |r| = 0.041) with corrected 100M EWoK focus margins: `data/ewok_focus_tokenization_margin_link/`. So official EWoK movement should be read as learned relation dynamics / vocabulary-target geometry, not crude fragmentation.

## Decision policy (from strategist + companion analysis)
- Give BOTH completed corrected seeds the same pristine full official coordinate; do not select a favorable draw. Surface-first partials are scheduling aids only; missing AoA stays missing until officially measured (not zero).
- If both corrected seeds clear 41.8 → compact_view_reinvest survives compliant representation as a strong endpoint family; protect and prepare submission after final checks.
- If one clears → protect winning seed, decide third seed or mechanism repair from column movement.
- If neither clears → old 42.033 inherited-tokenizer result stays mechanism evidence; route to scientific rebuild using score movement + tokenizer surface, not fragmentation-only explanations.
- Do not import old-tokenizer seed superiority or MLM loss as endpoint predictions.

## Peer state
- companion analysis message 54: their compliant tokenizer (SHA 91b775...) has 84.3% vocab overlap, mild occurrence-mass shifts; clean-Qwen with reinvest tokenizer is a fixed-tokenizer control (union 10,423,520 words), not a submission.
- companion analysis posted message 55 with the exact companion analysis-tokenizer surface (SHA 4a95...).
