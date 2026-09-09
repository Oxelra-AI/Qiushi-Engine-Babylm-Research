# changed block neardup scan route decision: resolve seed43122 official AoA BEFORE completing its full surface

## Situation (grounded, changed block neardup scan)

- Active best complete result: `compact_view_reinvest` seed43022, Overall **42.0868** (nine-column
  official mean), +0.7425 vs clean-Qwen 41.3443, +0.2868 vs the visible 41.8 leader.
- The sensitivity analysis established the single load-bearing risk with an exact sensitivity table:
  the entire +0.2868 lead rests on AoA staying **non-significant** (mapped to 0.0). If a second seed
  makes AoA significant:
  - AoA raw −0.05 → Overall 41.53 (still > leader)
  - AoA raw −0.10 → Overall 40.98 (below leader)
  - AoA raw −0.1269 (compact_core's value) → Overall 40.68 (below clean-Qwen)
  So seed stability of the AoA gate is decisive, not the non-AoA columns.

## What is actually running (do NOT duplicate)

- The replication is training a NEW seed43122 reinvest model and then running only a
  **fast no-AoA screen** (`fast_eval_density_seed.py`: BLiMP, Supplement, EWoK, Entity,
  Entity_full, COMPS — no SuperGLUE finetune, no AoA). Run dir:
  `experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122/hf_model`.
  The `per_target/compact_view_reinvest_seed43122.json` file exists (9258 bytes) but the directory is
  still being written; contents must not be inferred.
- Official AoA evaluation is running on the seed43022 ladders
  (near_repeat, near_view, compact_repeat_core, compact_view_reinvest).
- The remaining evaluation resumes compact_repeat_core SuperGLUE subtasks.
- GPU state at decision time: GPU0 util 100% / 70GB free, GPU1 util 97% / 79GB free. No isolated
  H100 free now; no additional GPU work can start at this point.

## Unresolved Gap

The seed43122 fast screen cannot measure official AoA — its screen is fast/no-AoA by design. But the
seed43122 training uses the identical recipe (`train_density_arm_seed.py`,
`checkpoint_words=1000000`, `max_word_exposure=100000000`), so it **saves the full 19-step
checkpoint ladder** (chck_1M…chck_100M) in its `hf_model` dir. That ladder is exactly what the
official AoA helper `experiments/archive/compact_experience/scripts/aoa_local_ckpts_for_model.py`
consumes (via `run_aoa_localization_ladders.py`).

Therefore the correct, lowest-cost, load-bearing next GPU action is:

**Run official AoA on the seed43122 checkpoint ladder first.**

- Cost: one AoA pass over a 19-step ladder (~single-GPU, comparable to the ~4497s AoA localization
  already running), far cheaper than a full 9-column eval or a SuperGLUE finetune sweep.
- Decision value:
  - If seed43122 AoA is **non-significant** (p > 0.1 → mapped 0.0): the AoA gate reproduces, the lead
    is robust to the seed on its most fragile column, and it is then worth completing the remaining
    official surface (SuperGLUE finetune + any missing zero-shot) to confirm the full second-seed
    Overall.
  - If seed43122 AoA is **significant negative**: the +0.2868 lead is seed-fragile exactly where the
    sensitivity table predicts; the endpoint is not a stable SOTA and the mechanism needs another
    scientific attack (e.g., an AoA-neutral variant or a source-diversity route that does not push the
    developmental-timing correlation across the significance boundary). Do NOT complete an expensive
    full second-seed surface in that case.

AoA-first evaluation makes the full second-seed surface conditional on the decisive column, limiting expensive evaluation without treating the unfinished surface as evidence.

## Preconditions before launching the seed43122 AoA measurement

AoA evaluation on the existing seed43122 ladder remains conditional on training completion and all 19 checkpoints being readable. Preflight must confirm `model.safetensors` at every checkpoint from `chck_1M` through `chck_100M` in `.../repl_compact_view_reinvest_seed43122`. This is a prerequisite for the planned evaluation, not evidence that it completed.

## Conditional Comparison Plan

- Add a `density_compact_view_reinvest_seed43122` target to a small dedicated launcher that points
  `model_root` at `experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122/hf_model`
  and reuses the exact official AoA helper. Run preflight first (no GPU), then the AoA pass on a free GPU.
- Compare seed43122 fitted correlation + p-value against seed43022 (r=0.0 mapped, non-significant).
  Read the seed43122 fast no-AoA columns (once lock releases) only as corroboration that the non-AoA
  multi-column gain reproduces — never as a substitute for the AoA measurement.

## Keep frozen

- The seed43022 endpoint is frozen. No new 100M training on the reinvest substrate at this stage.
- The GlobalPIQA/practical repair blueprint stays conditional — only after seed stability + contamination
  closure, and only if it does not disturb the EWoK/Entity/SuperGLUE/Supplement/AoA balance that made
  the lead.
