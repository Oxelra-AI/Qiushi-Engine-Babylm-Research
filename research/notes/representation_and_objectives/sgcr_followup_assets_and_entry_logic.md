# sgcr followup assets and entry logic — SGCR post-endpoint follow-up assets and expensive-work entry logic

This step prepared, verified, and made inert the two follow-up experiments that the
sgcr endpoint interpretation decision tree names, so that when the pending SGCR endpoint (`s87_t29_tool1`
train / `s87_t37_tool1` eval) delivers, the next expensive action can start without
launch delay and without launching the wrong follow-up. No GPU training and no
official evaluation were launched. SGCR managed tasks were not polled.

## Decision tree at the SGCR endpoint (unchanged from sgcr endpoint interpretation, made executable here)

Read the SGCR official-compatible vector first with:
`python3 -B experiments/archive/representation_and_objectives/scripts/sgcr_endpoint_interpreter.py`
It compares against matched depth (41.0276), legal40k 8x480 (41.1406), the visible
41.80 leader, and the sgcr official span burden burden map (COMPS/GlobalPIQA are the on-mechanism axes).

1. **SGCR clears or nearly clears 41.80** → protect the endpoint, then reproduce
   seed43122 (endpoint robustness, same data order seed 43, only init/train RNG
   changed to 43122/43123). Do NOT run the uniform control in this branch.
   - Train: `bash experiments/archive/representation_and_objectives/scripts/guarded_sgcrK50d64_train_seed43122.sh`
   - Eval: `bash experiments/archive/representation_and_objectives/scripts/guarded_sgcrK50d64_eval_seed43122.sh`
   - Controller: `scripts/sgcrK50d64_seed43122_posttrain_eval_controller.py` (dry-run preflight passes with only expected missing future artifacts).

2. **SGCR sub-frontier but improves COMPS/GlobalPIQA or Overall over matched depth
   enough to change the route** → run exactly one mass-matched uniform-residual
   control to separate support-dependent allocation from generic auxiliary capacity.
   - Train: `bash experiments/archive/representation_and_objectives/scripts/guarded_sgcr_uniformK50d64_train_seed43022.sh`
   - Eval: `bash experiments/archive/representation_and_objectives/scripts/guarded_sgcr_uniformK50d64_eval_seed43022.sh`
   - Controller: `scripts/sgcr_uniformK50d64_posttrain_eval_controller.py` (dry-run preflight passes with only expected missing future artifacts).
   - Attribution reading: if SGCR treatment beats uniform on COMPS/GlobalPIQA at
     matched extra parameters and matched training-token residual mass, the gain is
     support-dependent routing, not generic capacity. If SGCR ~= uniform, the effect
     is generic auxiliary capacity and support routing is not the mechanism.

3. **SGCR flat or damaging on COMPS and GlobalPIQA vs matched depth** → do NOT run
   either follow-up. Close exact-prefix SGCR as implemented and route back to a
   different representation, objective, or data mechanism.

## Why the uniform control is a valid single attribution comparison

sgcr followup assets and entry logic CPU preflight (`data/sgcr_uniform_control_preflight/…json`,
`preflight_passed: true`) verified that the existing corrected entity gate stats SGCR module can express
the control with the same construction and the same training-token residual mass:
- treatment mass-weighted residual `0.06428007036`, uniform mass-weighted residual
  `0.06428009272`, abs diff `2.24e-08` (matched to ~1e-8);
- treatment low-support/common residual type-mean ratio `2.529` (support routing
  present) vs uniform ratio `1.0` (routing removed, one ordinary-token rho);
- identical extra parameters (base 38,421,952; new 1,073,536; total 39,495,488);
- uniform cold effective table max diff vs standard `0.0` and K=0 exact recovery;
- exact-prefix decomposition SHA `b450cc45…`, histogram `{1:16384,2:19333,3:3629,4:571,5:63,6:16,7:4}`, 68,660 slots, pool tokens 13,942,644, used types 39,320.

An actual corrected entity gate stats trainer `--dry_run` at the target 12×384 shape with
`--sgcr_uniform_gate` confirmed the manifest emits `sgcr_uniform_gate: true`,
`sgcr_gate_mass_weighted_residual_ordinary_used = 0.06428009271621704`, matched
tokenizer SHAs, `zero_sharing_verified: true`, and the same parameter counts. The
uniform-control post-training controller enforces `sgcr_uniform_gate == True` and
the mass-residual equality before scoring.

## Evidence required before expensive follow-ups

None of these three follow-ups may start before the SGCR endpoint is read. Each is a
single decisive comparison that continues, changes, or closes the support-sharing
route:
- seed43122 reproduction answers whether a promising endpoint is seed-robust;
- uniform control answers whether a sub-frontier on-mechanism gain is support-routing
  or generic capacity;
- if SGCR is flat/damaging on COMPS+GlobalPIQA, no follow-up runs and the route
  closes. This prevents extending a failing line by adding trajectories, doses, or
  renamed variants.

## Files created this step
- `scripts/sgcr_uniform_control_preflight.py` + `data/sgcr_uniform_control_preflight/sgcr_uniform_control_preflight.json` + `notes/sgcr_uniform_control_preflight.md`
- Uniform control: `scripts/guarded_sgcr_uniformK50d64_train_seed43022.sh`, `training/scripts/full_eval_legal40k_12x384_sgcrUniformK50d64_seed43022.py`, `scripts/sgcr_uniformK50d64_posttrain_eval_controller.py`, `scripts/guarded_sgcr_uniformK50d64_eval_seed43022.sh`
- seed43122 reproduction: `scripts/guarded_sgcrK50d64_train_seed43122.sh`, `training/scripts/full_eval_legal40k_12x384_sgcrK50d64_seed43122.py`, `scripts/sgcrK50d64_seed43122_posttrain_eval_controller.py`, `scripts/guarded_sgcrK50d64_eval_seed43122.sh`
- Trainer uniform dry-run evidence: `data/sgcr_uniform_control_trainer_dryrun/` (stdout manifest only; `--dry_run`, no checkpoints written)
