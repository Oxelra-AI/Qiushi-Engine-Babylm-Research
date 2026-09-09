# seed43122 route decision: seed43122 delivered (timed out, reconstructed) — route decision

## Job state and recovery

The selected-grid scorer (scale1.75 seed43122 dense common 70M–100M 2M grid) reached the 10,800s wall-time limit and returned `timed_out` with exit_code 1 (wall-time limit exceeded). This is the same failure mode as the causal transfer result synthesis grids. It nevertheless wrote per-target payloads for 70M–92M before the cap.

Reconstructed with `reconstruct_selected_mlm_trajectory.py --write`:

- **11 complete endpoints**: `chck_70M` … `chck_90M` (all 8 cheap tasks, cheap7 recomputes exactly).
- `chck_92M` incomplete (BLiMP/Supplement/EWoK/Entity/COMPS present; GlobalPIQA/Reading missing).
- `chck_94M`…`chck_100M` missing.

Integrity: `data/seed43122_partial_integrity/` — 70–90M OK; strict-complete fails only on the missing tail rows (expected).

**Why the recovered range is sufficient for the decision.** The reference late competence-allocation phase is the window `chck_82M → chck_84M → chck_86M`. seed43122's own cheap7 peak among complete endpoints is `chck_88M` (43.909286), and its fixed `82M→84M→86M` window is fully inside the recovered range. Both windows needed to test recurrence of the reference phase are present. The missing 92–100M rows would only describe a later decline; they cannot change whether the 82–88M late phase's signed structure recurs.

## Recovered seed43122 selected cheap7 (70–90M)

70M 43.384 · 72M 43.330 · 74M 43.471 · 76M 43.849 · 78M 43.724 · 80M 43.626 · 82M 43.584 · 84M 43.726 · **86M 43.628 · 88M 43.909 (recovered peak)** · 90M 43.664.

Note: seed43122 `chck_82M` cheap7 43.584 and `chck_84M` cheap7 43.726 are both **below** the reference seed43022 `chck_82M` 43.959 / `chck_84M` 44.124. The seed43122 trajectory is broadly weaker on the stable columns at the reference coordinate.

## Three-analyzer readout (all CPU/file-only, on recovered grid)

### 1. Score-level peak-vector structure (lead cross seed decision framework)
`data/cross_seed_common_grid_analysis_all3_partial/`

- Column-peak-vector **Pearson 0.6528**, Spearman 0.7182 vs reference (scale1.25 was −0.048). So the coarse family-peak *schedule* is partly related: BLiMP late (92 vs ref 94), Supplement early (70 vs 76), EWoK mid, Entity mid, COMPS mid, GlobalPIQA late (88 vs 84).
- Aggregate recovered peak `chck_88M`, shift +4M.
- Mean Δ vs reference across common endpoints: **cheap7 +0.082**, but **cheap6-no-GlobalPIQA −0.349**, **cheap5-no-GP/Reading −0.423**, ΔEWoK/Entity +0.595. The nominally positive cheap7 rows (7 of 11) and 2 "broad positive" rows are carried by GlobalPIQA/Reading and by seed43122's stronger EWoK, not by broad stable-column gains.

### 2. Own-peak signed transitions (signed transition readout ready)
`data/signed_transition_signature_all3_partial/`

seed43122 own-peak window `86M→88M→90M` vs reference `82M→84M→86M`:

- **Item-level signed-transition overlap ≈ 0**: sign agreement on reference-changed items 0.0189 (rise) / 0.0162 (fall); signed-transition cosine −0.0053 / +0.0010; gain/loss Jaccard 0.008–0.013.
- **Subtask-level rise+fall pattern**: Pearson −0.109, weighted −0.101, cosine −0.109. **Anti-aligned to zero, not positive.**
- Coarse `column_official_subtask_mean` fall similarity is high (0.92 cosine), but this is the generic post-peak decline shared by all trajectories (scale1.25 also shows 0.93). It is not evidence of a shared rise into the phase.
- The reference top-moving BLiMP NPI subtasks (npi_present_1/2, only_npi_licensor_present) move in the **wrong** direction in seed43122 rise (pattern same? = False for most).

### 3. Fixed-window vs internal background (transition background for seed43122)
`data/transition_window_background_all3_partial/`

- seed43122's own-peak and fixed `82→84→86` windows do **not** stand out against its own late-window background at subtask/item level: own-peak column_subtask rise+fall cosine −0.109 ranks 7/9; item-pattern cosine −0.002 ranks 8/9; fixed-window item-pattern cosine 0.0022 ranks 3/9 with tiny absolute value.
- The best-matching windows in seed43122 are elsewhere (74→76→78 for coarse column, 84→86→88 for subtask), and even those are modest (subtask cosine ≤ 0.25).

## Decision

This is the **"structural but stochastic"** outcome from the predeclared framework (notes 195/197/198):

- The coarse family-peak *schedule* partly resembles the reference (peak-vector Pearson 0.65), showing the DeBERTa residual-adapter coordinate has a repeatable tendency to reallocate competence across families late in training.
- **BUT** the specific signed item/subtask transitions that produce the reference `chck_84M` late peak do **not** recur under a different init + mask stream, and no seed43122 window stands out against its own background. seed43122 is also broadly weaker at the reference coordinate.

Therefore the reference seed43022 `chck_84M` late competence-allocation peak is **a favorable stochastic alignment on one seed/mask trajectory, not a reproducible learning law**. It remains a valid **endpoint asset** (validated Overall(AoA0) 42.0189, HF revision `040284de9ac49eee6dc1b30cf65aea97ec17d86e`), but it must not be treated as a transferable data-efficient learning principle, and no more work should tune the seed43022 peak.

### What this closes
- Lower adapter amplitude (scale1.25): already closed (broadly negative).
- The narrow 82–84M peak as a robust residual-capacity law: closed by seed43122 non-recurrence.
- Do not launch further seed/scale trajectory scans to chase this peak.

### What this points to (next scientific target)
The residual-capacity coordinate reallocates broad competence **stochastically**. The next high-value question is **stabilizing broad competence across stochastic trajectories** rather than harvesting one lucky checkpoint. This is a distinct optimization-dynamics object (informed by, but distinct from, the closed prediction-vote and naive two-point-average routes). Any stabilization probe (e.g. same-trajectory weight averaging within a seed, or cross-seed competence-covariance-informed selection) must be judged by official-compatible selected scoring and must improve cheap7, cheap6-no-GlobalPIQA, cheap5-no-GP/Reading, and be nonnegative on EWoK/Entity — a GlobalPIQA/Reading-carried bump repeats the scale1.25/alpha0.75 failure pattern.

## Tool repair
`cross_seed_common_grid_analyzer.py` `contiguous_band` crashed on rows with `cheap7=None` (partial grid). Fixed the `max(...)` key to treat missing/None cheap7 as −1e99 so partial trajectories are analyzable.

## Protected assets unchanged
- Public `chck_82M` displayed 41.94 (challenge-counted fallback).
- `chck_84M` endpoint asset: Overall(AoA0) 42.0189, HF rev `040284de…`, native carrier SHA `b55e1997…`.
- coherent86 α=0.75: projected Overall(AoA0) 42.1210 (redistributive, submission candidate only).

## directional-fork loop

No official-compatible compact directional-fork result was available at the seed43122 route decision. Compact-order H100 work remained conditional on integrating that evidence.
