# clean qwen control interpretation and next mechanism — Clean-Qwen controls: scientific interpretation and next mechanism work

## What the complete clean qwen control eval summary controls establish

Authoritative summary: `experiments/archive/compact_experience/data/control_eval_summary.json`.

The clean-Qwen paired corpus is no longer a one-seed positive trace. Full local official-style nine-column evaluation gives:

- Seed 43022 matched pair: `qwen_clean_aligned` Overall **41.3443** vs `official_lengthmatched` **40.6679**, Δ **+0.6764**.
- Seed 43122 matched pair: `qwen_clean_aligned_seed43122` **40.6501** vs `official_lengthmatched_seed43122` **40.2318**, Δ **+0.4182**.
- Same seed 43022, correspondence-breaking shuffled rewrite control: aligned − shuffled Δ **+0.7135**.
- Same seed 43022, official-only source/domain matched control: aligned − source-matched Δ **+1.6605**.
- Same seed 43022, earlier official original-duplication control: aligned − original-dup Δ **+2.0530**.

The positive component is concentrated most stably in **Entity** and **SuperGLUE**, with Supplement positive in both seeds and AoA helped by the Qwen checkpoint ladder. The most consistent losses are **Reading** and **COMPS**; BLiMP, EWoK, and GlobalPIQA are less stable across seeds.

The best clean endpoint, **41.3443**, is still below the visible 41.8 leader. This evidence therefore supports continued mechanism and score work, not any final SOTA-facing statement.

## What remains uncertain

The remaining confounds are:

1. **Selected-original exposure/quality** remains the largest causal alternative. The Qwen pipeline selects 37,594 clean, rewriteable official sentences. The paired rows may function partly as a high-quality selected-original curriculum, not only a generated second-view signal.
2. **Same-window adjacency vs coexistence** remains unresolved. The shuffled control proves that a wrong adjacent rewrite is worse than the correct adjacent rewrite, but does not tell whether the correct rewrite must be in the same attention window or merely present elsewhere in the corpus.
3. **Source-matched control is informative but harsh.** It matches source totals but flattens and rechunks official streams, so the very large +1.66 contrast should not be interpreted as a pure source-mixture proof.
4. **The earlier original-dup control was useful but imperfect.** It used 36,687 selected duplicate pairs rather than all 37,594 and changed source/length details. It strongly suggests exact repetition is not enough, but a stricter all-selected duplicate control is needed.
5. **Training-time pair-side clipping is negligible.** `data/token_exposure_audit/pair_side_truncation_audit.json` shows only 106/12,236 pair rows exceed seq256; original visible fraction 0.999827 and rewrite visible fraction 0.998759. This does not explain the +0.4–0.7 effects.
6. **Two seeds are a sign/profile check, not a variance model.** The effect survives the stated continuation threshold in two seeds, but task-level instability remains substantial.

## New clean qwen control interpretation and next mechanism mechanism assets built in this step

Two materializers were written and CPU-audited:

1. `scripts/materialize_selected_original_dup_all_control.py`
   - Output: `data/selected_original_dup_all_control/`.
   - Keeps all 37,594 selected official originals, duplicates each original once, uses no Qwen words, preserves pair boundaries, matches clean-Qwen effective source totals.
   - Materialized duplicate-pair words: 1,698,026 (fraction 0.169803), close to the Qwen pair-word fraction 0.16568 but with all selected originals retained.
   - Purpose: test selected-original exposure/redundancy vs generated second-view value.

2. `scripts/materialize_qwen_separated_pair_control.py`
   - Output: `data/qwen_separated_pair_control/`.
   - Uses the same 37,594 selected originals and same Qwen rewrite multiset as the aligned treatment, but packs originals and rewrites into separate rows/windows; source totals matched to clean-Qwen effective source mix.
   - Pair words exactly match clean treatment: 1,656,800 (fraction 0.16568).
   - Purpose: test same-window original--rewrite adjacency vs mere coexistence/augmentation.

`data/selected_exposure_audit.json` verifies a useful invariant: for clean aligned, shuffled, separated, selected-original-dup-all, old original-dup, and official full-pool reference, `explicit_pair_plus_selected_filler_words` equals the unique selected example-ID row budget **4,077,760** per 10M pool. Thus these arms do not simply add selected original-row exposure above what a full official pool would contain; they substitute pair/duplicate material for the selected-row material plus filler. This strengthens but does not completely remove the selected-quality concern, because the paired material changes what part of the selected official row is emphasized.

Training launcher prepared and submitted:

- `scripts/launch_mechanism_control_training.sh`
- Runs to be produced:
  - `training/runs/selected_original_dup_all_16k_seed43022`
  - `training/runs/qwen_separated_pair_16k_seed43022`

Evaluation scripts prepared for after training:

- `scripts/full_eval_runner.py`
- `scripts/launch_mechanism_eval.sh`
- `scripts/summarize_mechanism_eval.py`

## How to interpret the upcoming clean qwen control interpretation and next mechanism controls

Let `A = qwen_clean_aligned`, `D = selected_original_dup_all`, `S = qwen_separated_pair`.

- If `A - D` remains clearly positive, the generated rewrite second view contributes beyond simply duplicating every selected original.
- If `A - D` collapses, the clean qwen control eval summary effect is mostly selected-original/redundancy and the route should pivot toward better official-only selected-exposure/data-quality recipes.
- If `A - S` remains clearly positive, same-window original--rewrite adjacency/correspondence is an active component.
- If `A - S` collapses, the benefit is likely coexistence/augmentation rather than attention-visible correspondence; next work should optimize rewrite selection/dose rather than adjacency.
- If `S - shuffled` is positive but `A - S` is small, correct pair identity matters somewhere in corpus exposure but not necessarily within one self-attention window.

## Score-oriented next work after mechanism controls

The clean route needs roughly +0.46 Overall over the best clean endpoint to reach the visible 41.8 leader. Mechanism and score work should therefore focus on the observed tradeoff:

- Preserve stable gains in Entity/SuperGLUE/Supplement.
- Recover Reading and COMPS losses.
- Use endpoint/trajectory checks, because prior mix25 peaked at 90M rather than 100M.

Prepared but not launched in this step:

- `scripts/launch_clean_qwen_trajectory.sh` for full zero-shot + Reading trajectories over `chck_10M..chck_100M` for both clean-Qwen seeds and matched official controls.
- `scripts/summarize_clean_qwen_trajectory.py` for trajectory comparison.

Proposed follow-up experiments:

1. Paired-fraction dose response around the current 16.568% (e.g. 8%, 16.6%, 24%).
2. Pair order/balanced packing: 50% original→rewrite and 50% rewrite→original, with tokenizer-aware no-clipping packing.
3. Same data but official consolidation tail: paired exposure early, final pass(es) official-only continuous rows to recover Reading/COMPS.
4. Source-conditional pairs: test whether conversational-source rewrites cause Reading/grammar losses; preserve SimpleWiki/Gutenberg pairs first.
5. Mid-overlap/high-utility pair reselection from existing Qwen pairs to reduce near-copy waste while keeping semantic relatedness.

None of these is a final deliverable; all are research execution toward a clean SOTA-capable principle.
