# architecture interaction readout repair architecture-interaction readout repair

## Scientific object

The active expensive experiment remains the earlier analysis DeBERTa architecture interaction:

\[
(\text{compact}-\text{repeat})_{\text{full DeBERTa}}
\quad \text{versus} \quad
(\text{compact}-\text{repeat})_{\text{no\_disentangle\_abs}}.
\]

The full compact cell is the existing legal compliant tokenizer retrain status compact run. Training remained pending or in progress for the three missing cells:

- Full-repeat followed by no-disentangle-repeat training.
- No-disentangle-compact training.

No conclusion about c2p/p2c necessity can be drawn until these terminal results are complete, pass integrity checks, and the selected readout is run.

Interpretation convention:

- Survival of compact-minus-repeat without c2p/p2c attention-score tensors means those score terms are not necessary for the compact semantic second-view effect in this coordinate.
- Attenuation/collapse implicates the whole removed score-term package plus parameterization and score-composition changes, not c2p versus p2c individually.
- Stable families remain primary: `cheap6_no_GlobalPIQA`, `cheap5_no_GlobalPIQA_Reading`, `EWoK_plus_Entity_sum`, Supplement, Entity, COMPS. Loss, GlobalPIQA-only movement, or single best-checkpoint stories are insufficient.

## Repairs completed

### 1. Integrity reader replaced and hardened

Replaced `experiments/archive/frontier_consolidation/scripts/architecture_interaction_integrity_reader.py` with a hardened file-only reader.

New capabilities:

- Verifies all four arms with explicit expected data stream, tokenizer, recipe, parameter count, checkpoint grid, config, and state-dict signatures.
- For new earlier analysis arms, treats `train_command.json` as mandatory preflight record and checks:
  - `status = DEBERTA_POSITIONAL_ABLATION_PREFLIGHT_OK`
  - `variant`, `data_arm`
  - legal stream SHA
  - legal tokenizer SHA `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`
  - exact `word_info`: 647,400 rows, 100,000,000 words, 30,050 changed rows, 3,005 changed IDs, no bad repeat counts
  - model-variant preflight parameter/vocab/pos-projection indicators
  - mandatory recipe scalars including seed 43, extra_init_seed 43022, train_rng_seed 43023, batch256/seq256, AdamW lr 0.001, warmup 0.06, WWM p=0.15, checkpoint_words 10M, max exposure 100M, lr_total_steps 2529.
- For existing full compact, allows the known older compliant tokenizer retrain status format (checkpoint_words 1M and no explicit `--deberta_pos_att_type` flag) while checking the legal compact stream SHA, legal tokenizer SHA, compliant tokenizer retrain status recipe, and all required checkpoints.
- Directly inspects checkpoint tensor keys at `chck_80M` and `chck_100M`:
  - full variants must contain 16 `pos_key_proj` tensors and 16 `pos_query_proj` tensors (weight+bias for 8 layers), plus absolute position embeddings and encoder relative embeddings.
  - no-disentangle variants must contain zero `pos_key_proj` and zero `pos_query_proj` tensors while retaining absolute position embeddings and encoder relative embeddings.

Plan output:

- `experiments/archive/frontier_consolidation/data/architecture_interaction_integrity_plan/architecture_interaction_integrity_plan_8ac99ae7.json`

Existing full-compact self-test:

- `experiments/archive/frontier_consolidation/data/architecture_interaction_integrity_fullcompact_selftest/architecture_interaction_integrity.json`
- Result: `full_compact_existing` passes, including:
  - 100M exposure, 2529 steps, 34,467,424 params
  - legal tokenizer/stream SHA
  - required `chck_80M` and `chck_100M`
  - 100 saved checkpoints as expected for compliant tokenizer retrain status
  - tensor signatures: both `chck_80M` and `chck_100M` have 16 `pos_key_proj`, 16 `pos_query_proj`, absolute position embeddings, and encoder relative embeddings.

### 2. Tensor-key expectations validated on real smoke checkpoints

Output:

- `experiments/archive/frontier_consolidation/data/architecture_interaction_tensor_signature_probe/tensor_signature_probe.json`

Results:

- `full_smoke` (`smoke_full_repeat_999918w_seed43022/chck_1M`): ok, 16 `pos_key_proj`, 16 `pos_query_proj`, absolute position embeddings present, encoder relative embeddings present, 170 total keys.
- `nodis_smoke` (`smoke_nodis_compact_999918w_seed43022/chck_1M`): ok, 0 `pos_key_proj`, 0 `pos_query_proj`, absolute position embeddings present, encoder relative embeddings present, 138 total keys.
- existing `full_compact_100M`: ok, 16/16 positional projection tensors, absolute and relative embeddings present.

This validates that the direct state-dict signature distinguishes the intended architecture change in the actual checkpoint format.

### 3. Selected-panel validator path bug repaired

The earlier analysis selected panel had a false rejection for the reused full-compact 100M payload: it compared a root-relative recorded `run_dir` with an absolute expected path by raw string form, producing an apparent mismatch even when the printed paths named the same run.

Patched in `experiments/archive/frontier_consolidation/scripts/architecture_interaction_selected_panel.py`:

- `validate_selected_per_target` now resolves recorded paths against their recorded root and compares normalized absolute paths for both `run_dir` and `model_path`.

Correct no-force self-test:

- `experiments/archive/frontier_consolidation/data/architecture_interaction_selected_fullcompact_selftest_v3/architecture_interaction_selected_panel_summary.json`
- Result: both reused full-compact rows validate:
  - 80M cheap7 42.948571, cheap6_no_GlobalPIQA 44.176667, `payload_validation_ok=true`
  - 100M cheap7 43.005714, cheap6_no_GlobalPIQA 44.1625, `payload_validation_ok=true`
- Remaining notices are only expected one-arm interaction absence at 80M and 100M.

An earlier v2 self-test used `--force`, bypassing cached-payload reuse and starting a selected evaluation on GPU0. It timed out at 120s and produced no scientific result. Cached full-compact reuse tests must omit `--force`.

### 4. Four-cell bootstrap analyzer revalidated

Re-ran the synthetic-zero four-cell bootstrap after earlier analysis/245 hardening:

- `experiments/archive/frontier_consolidation/data/architecture_interaction_bootstrap_synthetic_zero_run/architecture_interaction_four_cell_bootstrap_5796e4c3.json`
- Markdown summary: `.../architecture_interaction_four_cell_bootstrap.md`

Result: using the same existing full-compact 80M per-target file for all four arms, the direct item-level interaction is exactly zero:

- stable_five item and cluster interaction point 0.0, interval [0.0, 0.0]
- EWoK_plus_Entity item and cluster interaction point 0.0, interval [0.0, 0.0]

This checks sign convention, common-item handling, and bootstrap algebra in the null case.

## Role-exchange probe evidence

The commoncopy and paired world design v2 records 42 structurally usable role-exchange families, cross-teacher agreement 0.7976, both-teachers-expected 0.7560, strict source-bridge retention 11/42 families and full-context retention 7/42. This collection is useful only as probes/design seeds and is too sparse for student or BabyLM training. It does not justify a new sparse role-substrate training route before the architecture-interaction comparison is resolved.

## Validation after all training arms complete

1. Obtain completed training results for all four arms.
2. Run the repaired integrity reader on all four arms:

```bash
python3 experiments/archive/frontier_consolidation/scripts/architecture_interaction_integrity_reader.py \
  --out-dir experiments/archive/frontier_consolidation/data/architecture_interaction_integrity_final
```

3. If a new arm fails from engineering, rerun only that same missing cell with the same recipe.
4. If all four cells pass, run the hardened selected panel at 80M and 100M:

```bash
python3 experiments/archive/frontier_consolidation/scripts/architecture_interaction_selected_panel.py \
  --out-dir experiments/archive/frontier_consolidation/data/architecture_interaction_selected_panel_final \
  --checkpoints chck_80M chck_100M
```

5. Then run:
   - `architecture_interaction_interval_driver.py` for within-architecture compact-minus-repeat paired intervals.
   - `architecture_interaction_bootstrap_analyzer.py` for the direct four-cell item interaction.

Do not run SuperGLUE, AoA, upload, or leaderboard submission from this mechanism readout.
