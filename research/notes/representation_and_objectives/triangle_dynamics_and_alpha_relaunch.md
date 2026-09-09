# triangle dynamics and alpha relaunch triangle dynamics and alpha0.75 replay relaunch

## Compact-view mechanism triangle: CPU training-dynamics readout

Completed a CPU-only readout of the three available training logs while the official-compatible no-AoA/posthoc readout remains managed work.

Files:
- Script: `experiments/archive/representation_and_objectives/scripts/triangle_training_dynamics.py`
- JSON: `experiments/archive/representation_and_objectives/data/triangle_training_dynamics/triangle_training_dynamics_mechanism.json`
- Markdown summary: `research/documents/representation_and_objectives/data/triangle_training_dynamics/triangle_training_dynamics_mechanism.md`

Run endpoints and local objective traces:
- `compact_view_reinvest_historical`: 2529 steps, 100,000,000 words, final training loss 2.565617084503174, `hf_model/chck_100M` exists.
- `compact_repeat_reinvest_gc`: 2529 steps, 100,000,000 words, final training loss 2.577287197113037, `hf_model/chck_100M` exists.
- `adjbreak_reinvest_gc`: 2529 steps, 100,000,000 words, final training loss 2.6621243953704834, `hf_model/chck_100M` exists.

Pairwise loss deltas are `b - a` on the same 2529-step LR schedule and identical word-exposure grid:
- `repeat - view`: mean -0.008632, median -0.008364, final +0.011670, last100 -0.008763, fraction higher 0.313.
- `adjbreak - view`: mean +0.072744, median +0.081514, final +0.096507, last100 +0.099683, fraction higher 0.899.
- `adjbreak - repeat`: mean +0.081376, median +0.088388, final +0.084837, last100 +0.108446, fraction higher 0.945.

Scientific reading: breaking source-own rewrite adjacency makes the training stream persistently harder for the MLM objective after early training, even when the rewrite marginal is preserved. This supports adjacency/coherent consolidation as a real local-learning factor, but it is not yet a generalization result. A lower training loss can mean easier local prediction or repetition. Mechanism interpretation still needs the official-compatible no-AoA endpoint table, broad-competence profile, and item-level retained/new/lost pattern from the managed readout under the earlier analysis seed-spread bands.

The implementation-equivalence requirement is satisfied by earlier analysis repaired evidence: `experiments/archive/representation_and_objectives/data/gc_reference_equivalence_repair/gc_reference_equivalence_repair.json` reports exact 26-update trajectory equivalence and `allows_historical_reference_for_causal_triangle: true`. The triangle dynamics and alpha relaunch dynamics script was patched and rerun so its JSON exposes this true value.

## Managed triangle readout still active

Triangle evaluation and post-hoc analysis remained pending. Before interpreting their outputs, verify that they use the repaired equivalence summary rather than the superseded lr26 comparison or compact-only stale table.

## coherent86 alpha0.75 continuation branch

earlier analysis's first managed launch failed before training because `--force` tried to remove the pre-existing dry-run output directory from a background sandbox. The replay design itself remains the same: start from the verified 82M anchor, train private adapters at scale 1.0 with reconstructed optimizer/RNG/LR process, prove the known 83M/84M/85M/final86 states match layerwise exchange readout and gradient anatomy before extension, then materialize alpha0.75 only for evaluation states.

Relaunched as fresh managed task:
- Task ref: `s208_t8_tool1`
- Label: `guarded alpha075 fresh replay from 82M`
- Output directory: `experiments/archive/representation_and_objectives/training/runs/alpha075_exact_replay_from82M_seed43022`
- Command uses `--allow_extend --require_cuda --wait_for_gpu` and no `--force`.

The expensive relaunch is justified because it decides whether the strongest projected endpoint can be completed truthfully for official AoA/fast evaluation: success produces genuine alpha0.75 `chck_90M` and `chck_100M`; any mismatch against the known layerwise exchange readout and gradient anatomy 83M-86M prefix aborts before extension and prevents an invalid developmental lineage.

## Triangle provenance readiness

Added cheap CPU readiness verifier:
- Script: `experiments/archive/representation_and_objectives/scripts/triangle_readiness_verify.py`
- JSON: `experiments/archive/representation_and_objectives/data/triangle_readiness/triangle_readiness.json`
- Summary: `research/documents/representation_and_objectives/data/triangle_readiness/triangle_readiness.md`

Result: `TRIANGLE_READY_FOR_GUARDED_ENDPOINT_READOUT`, `all_ok: true`. All three arms (`compact_view_reinvest_historical`, `compact_repeat_reinvest_gc`, `adjbreak_reinvest_gc`) have intended stream SHA, stream label, recipe, 100 saved checkpoints, and `hf_model/chck_100M/model.safetensors`. This establishes that pending endpoint/posthoc readout should not be blocked by data/recipe/provenance mismatches, provided it uses the repaired earlier analysis implementation-equivalence summary rather than the superseded lr26 comparison.
