# chck84 item movement synthesis `chck_84M` endpoint branch: item-level movement synthesis

CPU/file-only analysis of already-produced official-compatible predictions on the scale1.75 seed43022 reference trajectory. No SuperGLUE evaluation or training was performed. The chck_84M SuperGLUE evaluation and DeBERTa common-grid resume remained pending.

Script: `scripts/chck84_item_movement_analyzer.py`
Output: `data/chck84_item_movement_analysis/`

## Reproduction check

Column scores recomputed from the actual official prediction/data files match the reconstructed selected-grid cheap columns exactly:

- `chck_82M`: BLiMP 68.4913, Supplement 62.9378, EWoK 50.0555, Entity 28.3140, COMPS 52.1912, GlobalPIQA 37.5777, Reading 8.15 → cheap7 43.95857142857143.
- `chck_84M`: BLiMP 68.2512, Supplement 63.4837, EWoK 50.0735, Entity 28.5751, COMPS 52.2056, GlobalPIQA 38.1214, Reading 8.155 → cheap7 44.12357142857143.
- `chck_100M`: BLiMP 68.6318, Supplement 62.8952, EWoK 49.0801, Entity 27.4645, COMPS 52.3039, GlobalPIQA 36.1068, Reading 8.32 → cheap7 43.54214.

170,722 classification items compared; 14,393 flip rows saved.

## `chck_84M` vs `chck_82M` (Δ84−82)

- BLiMP −0.2401, Supplement +0.5459, EWoK +0.0180, Entity +0.2611, COMPS +0.0144, GlobalPIQA +0.5437.
- 5/6 classification columns positive; only BLiMP negative.
- cheap6 without GlobalPIQA is still up (Supplement + Entity gains outweigh the BLiMP loss), so the 84M lift is **not** a GlobalPIQA/Reading accident. It is a genuine broad-but-shallow classification improvement.

## The 84M peak coincides with where the 100M decline concentrates

Δ100−84 by column: BLiMP +0.3806, Supplement −0.5885, EWoK −0.9934, Entity −1.1106, COMPS +0.0983, GlobalPIQA −2.0146.

The families that 84M gains (Supplement, EWoK, Entity, GlobalPIQA) are exactly the families that decline sharply by 100M, while BLiMP/COMPS recover. This is the same vector-valued competence reallocation identified earlier for 82M→100M, now resolved one 2M step later: **84M is the true interior late-peak on this trajectory, not 82M.** The protected/submitted public endpoint `chck_82M` is one 2M step below its own trajectory's cheap7 peak.

## Churn shows allocation, not accumulation

Raw item churn is large relative to net movement even where the weighted column score improves:

- BLiMP: 1015 gains vs 1154 losses (net −139), despite tiny net column move.
- COMPS: 3001 gains vs 2919 losses (net +82) on ~91k items.
- Supplement 51/61, EWoK 237/240, Entity 134/123, GlobalPIQA 5/4.

So the 84M advantage is a reweighting of which items are answered correctly, consistent with the residual-adapter trajectory redistributing competence rather than monotonically adding it.

## Subtask signatures (largest 82→84 movers)

- BLiMP up: existential_there_quantifiers_2 (+7.57), left_branch_island_echo_question (+3.80); down: npi_present_2 (−5.69), only_npi_licensor_present (−5.10), npi_present_1 (−4.18). NPI licensing is the main BLiMP loss.
- Supplement up: qa_congruence_easy (+1.56), qa_congruence_tricky (+1.21), turn_taking (+0.71).
- EWoK up: material-dynamics (+1.69); down: physical-dynamics (−1.67), agent-properties (−1.04).
- Entity up: move_contents_* subsets broadly (+1.1 to +1.5).

## Endpoint-branch decision, kept separate from mechanism

`chck_84M` is a real, legal, same-trajectory endpoint opportunity. It is exposure 84,028,405 words (8.4028405 epochs), model SHA256 `2217917c687faf4de26ef6f381be3048d0bd66b2025c06382683c24d78e8d8c9`, static files matching `chck_82M` except weights, trusted-code loadable as `AdapterDebertaV2ForMaskedLM` (35,463,008 params).

Whether it beats protected `chck_82M` on Overall depends on SuperGLUE (pending). Thresholds (AoA=0): SuperGLUE ≥ ~68.620 beats `chck_82M`; ≥ ~69.135 reaches Overall 42.0; ≥ ~70.224 matches coherent86 α=0.75.

this remains an endpoint branch. It does **not** organize the transferable residual-capacity story; that requires the pending seed43122 grid to test whether the late peak and its family profile survive a different init + mask stream. A single-seed 84M peak is one trajectory, not a law.
