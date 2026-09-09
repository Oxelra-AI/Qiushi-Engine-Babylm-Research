# clean qwen experiment state — Clean-Qwen first full eval and control wave decision

## Repaired full evaluation
The first `launch_full_eval.sh` run failed only at AoA because the inherited AoA helper used relative cache paths under the strict eval repo cwd, causing a read-only path error. All zero-shot, Reading, and SuperGLUE tasks had completed successfully. I created `scripts/aoa_local_ckpts_for_model.py` with absolute local cache paths and patched `scripts/full_overall_eval_runner.py` to use it. Rerunning only `--columns AoA` for both targets succeeded and `scripts/summarize_full_eval.py` produced a complete summary.

Canonical summary: `data/full_eval/full_eval_summary.json`  
Note: `notes/clean_qwen_full_eval.md`

## Scores (superseded by aoa unit correction AoA unit correction)

The table and `+0.676392` delta originally written here used raw AoA correlation as the AoA leaderboard column. That was corrected in aoa unit correction. Use the canonical corrected files instead:

- `data/full_eval/full_eval_summary.json`
- `data/control_eval_summary.json`
- `notes/aoa_unit_correction.md`

Corrected first-seed values: `official_lengthmatched` Overall **38.9404** with AoA leaderboard **-15.7045** (raw -0.157045); `qwen_clean_aligned` Overall **41.3443** with AoA leaderboard **0.0**. Corrected delta `qwen_clean_aligned - official_lengthmatched` is **+2.403887** Overall, decomposing into NLP-average +0.874354, Reading -0.19, and AoA leaderboard +15.7045.

## Interpretation right now
This is a real internally matched positive result for the cleaned 16.568%-pair filtered-Qwen pipeline. It is **not** yet SOTA and **not** yet a mechanism result:
- Absolute `qwen_clean_aligned` Overall 41.3443 is below the visible 2026 Strict-Small leader 41.8.
- The result is higher than the inherited 40.7028 coordinate by about +0.6415, but still below the old contaminated mix25 local 41.4803.
- The positive profile is concentrated in Supplement, Entity, and SuperGLUE, not broad across all columns.
- The treatment still confounds paired semantic adjacency, within-row redundancy, generated-register shift, source reweighting, and tokenizer truncation asymmetry.

## Immediate control wave
Since the +0.25 gate was passed, launch causal/replication controls without mechanism claims:
1. Shuffled-pair control: same selected originals and same Qwen rewrite multiset, rewrites shuffled within source/length bins to break correspondence. This tests whether original--rewrite adjacency matters beyond generated text/style/source.
2. Second-seed replication: start seed 43122/43123 arm(s) to determine whether the sign survives initialization/training RNG.
3. Source-matched and original-duplication controls are already prepared and should be run/evaluated after the first control wave or in parallel when GPU slots are available.

The next full evaluation for controls should use `scripts/full_eval_runner.py --target <target>`; it now includes `qwen_shuffled_control`, `official_sourcematched`, and `official_original_dup` target definitions.
