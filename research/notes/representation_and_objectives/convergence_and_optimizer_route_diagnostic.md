# convergence and optimizer route diagnostic — compliant-substrate convergence diagnostic and optimizer-route reading

CPU-only reading of already-written training logs. No GPU work, no polling of managed
tasks, no new corpus/tokenizer/model. This sharpens the next route choice after the
pending SGCR endpoint by testing whether the compliant AdamW-0.001 recipe is
under-optimizing (which a LAMB-0.007 "train harder" route would target) or already
converged on the MLM objective (which points the residual gap at data/representation).

## Sources
- Depth run log (2,529 steps): `training/runs/legal40k_12x384_depth_compact_view_reinvest_seed43022/training_log.jsonl`
- SGCR K=50/d64 run log (52 sampled lines): `training/runs/legal40k_12x384_sgcrK50d64_compact_view_reinvest_seed43022/training_log.jsonl`
- legal40k 8x480 baseline log: `training/runs/legal40k_accum_compact_view_reinvest_seed43022/training_log.jsonl`

## Final MLM loss (all at 100M words, 2,529 steps, cosine to LR 0)
- legal40k 8x480 baseline: 2.4957
- depth 12x384: 2.5247
- SGCR 12x384 K=50/d64: 2.5413

Depth and SGCR (deeper/narrower, same 12x384) end with slightly higher MLM loss than
the shallower/wider 8x480 baseline. MLM loss does not select downstream winners
(as established by the corrected-tokenizer comparison), so this is not a ranking; it only shows SGCR's extra rare-token
routing does not lower training loss, and depth's capacity shift does not lower it either.

## Depth loss trajectory: converged well before budget end
Per-checkpoint depth loss at exposure milestones:
- 10M 4.4322, 20M 3.7887, 30M 3.5013, 40M 3.1052, 50M 2.9087, 60M 2.8615,
  70M 2.5337, 80M 2.5086, 90M 2.5206, 100M 2.5247

Descent: 50->100M = 0.3840; 80->90M = -0.0119; 90->100M = -0.0041. After ~70M words the
per-checkpoint loss is non-monotonic (2.5086, 2.5206, 2.5247) — noise-dominated, not
descending. The very final window (steps 0.95-1.00, mean 2.4445) is lower only because
of the near-zero cosine-decay LR endgame, not continued learning.

**Reading:** the compliant AdamW LR 0.001 / warmup 0.06 / fixed WWM 0.15 / 2,529-step
recipe is MLM-converged, not under-optimized. A naive LAMB-0.007 "optimize harder /
more steps" hypothesis is weakly motivated by the trajectory: extra optimization amount
would not obviously lower MLM loss further, and MLM loss is not the bottleneck. The
leader's LAMB-0.007 + 64->256 sequence curriculum most plausibly changes *what* is
learned (target-token exposure raised ~1.6-2.0x, shorter-context first) and rides on
*factually dense* FineWeb simplification-pair data (leader EWoK 56.07, COMPS 53.57,
GlobalPIQA 39.67), not on raw optimization amount on the same corpus.

This does not close a sequence-curriculum route — the faithful word-boundary chunking
route (family specific tokenizer predictor) raises target-token exposure and could still change learned content
— but it argues against spending a compliant 100M run purely on optimizer/LR change with
the current data and fixed 256 length.

## SGCR training dynamics: well-behaved, no pathology
SGCR sampled loss vs depth at matching steps:
- babylm2026 live surface identical (10.6459); SGCR -0.3726 (component correction helps early rare-token fit);
- steps 100-1600 SGCR within +/-0.03 of depth, mostly slightly below;
- steps 1700-2529 SGCR mostly marginally above depth (final +0.0166).

This is the signature of an auxiliary module that changes rare-row representation without
disrupting global optimization. No divergence, no instability. The endpoint official-
compatible vector remains the only valid SGCR route signal; training loss cannot rank it.

## Managed-task status (observed, not polled as result)
- SGCR eval `s87_t37_tool1`: watcher found all 100 checkpoints, confirmed chck_100M at
  earlier analysis, selected GPU0, launched official evaluation at 16:12:38Z. Now in scoring
  phase; pristine collation summary does not yet exist. Interpreter
  `scripts/sgcr_endpoint_interpreter.py` still correctly waiting.
- Full-EWoK `s92_t8_tool1`: still running.
Do not poll; runtime delivers terminal results.

## Route implication for the step after SGCR
- If SGCR clears/nearly clears 41.80: protect + reproduce seed43122 (sgcr followup assets and entry logic assets).
- If SGCR sub-frontier with only tiny COMPS/GlobalPIQA bump amid broad depth losses:
  close exact-prefix SGCR (strategist earlier analysis), do NOT auto-run the uniform control.
- Redirect target, given this diagnostic + earlier analysis negative tokenizer-seam result:
  the residual gap is a data/representation/learned-content problem. Highest-value
  distinct compliant candidates are (a) faithful sequence-curriculum experience
  utilization on the compact-view stream (family specific tokenizer predictor, changes learned content, not
  just optimization amount), and (b) strengthening the validated compact-view data
  mechanism itself, not a pure optimizer/LR swap on fixed 256-length data.
