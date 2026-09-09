# accum trainer equivalence and confound consolidated evidence: legal-40k trainer equivalence and representation interface

Purpose: while the two legal-40k compact_view_reinvest 100M trainings run asynchronously
(`s61_t27_tool1` seed43022, `s61_t27_tool2` seed43122), make the eventual full-official
40k-vs-16k comparison interpretable by (a) bounding the accumulated-trainer implementation
confound and (b) mapping the intended representation-interface change. All work here is
CPU-only and did not poll or interrupt the GPU tasks.

## 1. The 16k-vs-40k comparison has one trainer-implementation confound

- Legal-16k baseline (earlier analysis/51: seed43022 40.704, seed43122 41.024) used the BASE trainer
  `experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py` at full batch 256.
- Legal-40k uses the legal40k accum training completion accumulated trainer (effective 256 via 4x64 microbatches),
  required because direct batch-256 40k OOMed (~75.43 GiB allocated, ~2 GiB free).
- So the comparison changes (i) the intended tokenizer representation package AND
  (ii) full-batch-256 vs microbatched-256 forward. The gradient objective weighting is
  algebraically equivalent (masked-token-weighted accumulation = full-batch token mean),
  but realized gradients differ through dropout realization and floating-point/kernel order.
  This is objective-weight equivalence, not realized-gradient identity.

## 2. Structural pre-forward identity is proven (CPU)

`accum_data_mask_identity_check.py` ->
`data/accum_data_mask_identity_check/accum_data_mask_identity_check.json`
- For legal16k and legal40k, recombining 4x64 microbatches reproduces the direct 256-row
  input tensors exactly and identical pre-forward WWM masked_inputs/labels. all_pass=true.

`accum_sequential_and_overflow.py` ->
`data/accum_sequential_and_overflow/accum_sequential_and_overflow.json`
- Stronger continuous-RNG test: two equally seeded generators kept alive across many
  consecutive effective batches. legal16k long run covered 100 full 256-row effective
  batches plus a 37-row final partial effective batch. all_sequential_pass=true, including
  final_generator_state_equal=true for legal16k short/long and legal40k short.
- Conclusion: the 4x64 recombination preserves the entire masking RNG stream and the final
  RNG state; the only trainer-level difference is the stochastic forward execution.

## 3. Residual confound is bounded by a prepared short control

`accum16k_equivalence_control.py` (preflight PASS,
`data/accum16k_equivalence_control/accum16k_equivalence_preflight.json`):
- When a GPU frees, run the accumulated trainer with the EXACT legal-16k tokenizer and
  seed43022, stopping at 4,030,900 words (base-16k chck_4M row boundary, fw comparison mechanical audit), with
  `--lr_total_steps 2529` so the LR schedule matches the full 100M run.
- Compare per-checkpoint/step loss to the base-16k log via cumulative target-weighted loss;
  bound the trainer effect against ordinary between-seed dispersion. This is a bound, not a
  score, and transferring it to 40k remains an explicit assumption because direct 40k
  batch-256 cannot run.

## 4. The intended representation-interface change (corrected map)

`wwm_target_burden_map.py` ->
`data/wwm_target_burden_map/` (JSON + summary + CSVs). Separates raw row length /
overflow from visible seq256 target surface.
- Global 10M pool: raw tokens/word 1.4669 (16k) -> 1.3943 (40k); visible target-token
  reduction 4.14%. Visible WWM groups/word rise slightly 0.9802 -> 0.9871.
- Changed compact-view block: visible target-token reduction 9.98%; raw tokens/word
  1.4362 -> 1.2916.
- Under fixed WWM, word/group selection probability is unchanged; 40k reduces subword
  targets per selected word and exposes marginally more visible groups/context. The score
  vector must be attributed to the whole package (segmentation, visible context, per-epoch
  target volume, and the 34.5M->45.8M parameter increase), not fragmentation alone.

## 5. Overflow transition matrix (unambiguous)

From `accum_sequential_and_overflow.json` on the exact 10M pool (64,740 rows):
- neither overflow: 49,597
- only legal16k overflow: 3,736
- only legal40k overflow: 0
- both overflow: 11,407
- Net overflow reduction 16k-40k = 3,736, and it is strictly one-directional: 40k never
  introduces new seq256 truncation anywhere on the pool. This replaces the earlier ambiguous
  "rescued rows" net count with an exact partition.

## 6. Interpretation boundary for the eventual 40k result

- If legal 40k recovers Supplement and EWoK without losing GlobalPIQA/Entity/COMPS, the
  representation-interface hypothesis (less fragmentation, more visible context, fewer but
  cleaner subword targets) is supported.
- If it loses GlobalPIQA/Entity/COMPS, the larger vocabulary likely weakened useful rare-
  token/subword regularization under a 10M-word budget.
- Either way, the accumulated trainer does not invalidate the experiment; the trainer effect
  is bounded (structurally proven identical pre-forward; short control prepared), while the
  intended package effect is mapped above.
- Decision criterion: if legal 40k does not clear the 41.8 leader under full pristine
  official evaluation, do not spend another full-run pair on intermediate vocabulary sizes;
  move to a genuinely different curriculum or architecture route.
