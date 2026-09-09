# alpha075 developmental staging — coherent86 alpha0.75 submission-tail: truthful developmental staging

## What this establishes
companion analysis-owned truthful developmental lineage staged for the current local highest candidate
`coherent86 alpha=0.75` (projected Overall(AoA0) 42.1210247099666; model SHA
`e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8`; config SHA
`ca6792fe1842f0e89e7080e7709e0619fdcaf5ce48a65f69bf7137ac26ce9bb7`).

Staging tree: `experiments/archive/representation_and_objectives/data/alpha075_developmental_staging/hf_model`
Manifest: `.../alpha075_developmental_staging_manifest.json` (status PARTIAL).
Load pilot: `.../alpha075_staged_load_pilot/alpha075_staged_load_pilot.json` (PASS).

## Developmental-history policy applied (strategist alpha075 developmental staging note)
- **chck_1M..chck_80M (official fast slots ≤82M):** symlinks to the true scale1.75 slow
  ancestral trajectory (compact anchor match ladder). Before the 82M private intervention there is
  NO learned private contribution, so early revisions reproduce the true ancestral function.
  `AdapterDebertaV2ForMaskedLM`, no private adapter.
- **chck_83M/84M/85M (optional, nonofficial):** the REAL saved coherent private-tail states
  (`layerwise exchange readout and gradient anatomy` `chck_total_8{3,4,5}012495w`), re-materialized only by changing config
  `private_adapter_scale` 1.0→0.75; weights hardlinked unchanged. These are the developmental
  bridge; not strict-small fast slots, preserved so no later fake-history repair is needed.
- **chck_86M + main:** the exact alpha0.75 candidate final model (`routeB closure and scale1p75 eval/.../final`).
  chck_86M is the measured full-eval endpoint, not a fast slot. main==chck_86M bit-identical.
- **chck_90M / chck_100M:** DELIBERATELY MISSING. The coherent private tail stopped at total
  86,005,295 words (max_tail_charged 3,992,800). These two required strict-small fast revisions
  do not exist and MUST NOT be faked by (a) retrofitting final private weights into earlier
  checkpoints, (b) borrowing the divergent ordinary post-82M trajectory, or (c) duplicating 86M.
  They require GENUINE continuation of the unchanged coherent private tail from 86M.

## Load pilot facts (CPU, 9.8s)
- chck_1M, chck_80M, chck_83M, main all load locally with finite logits.
- chck_83M and main carry `private_adapter_scale=0.75`.
- alpha1.0 vs alpha0.75 chck_83M: max_abs_logit_diff 0.19905829429626465 (real scale change).
- main vs chck_86M alias: max_abs_diff 0.0 (identical, correct).

## Remaining work (GPU, do NOT preempt the main triangle trainings)
1. Genuine continuation from the coherent private tail 86M → 90M and 100M using the
   `layerwise exchange readout and gradient anatomy` frozen-slow private trainer with the SAME frozen scale1.75 slow path, SAME private
   adapter, SAME compact_view_reinvest stream and RNG discipline, materialized at alpha=0.75.
   This is the only truthful way to fill chck_90M/chck_100M. Exposure would exceed the current
   86.0M; must stay within the ≤100M cumulative cap and be accounted exactly.
2. Then run official min_context=0 AoA (8005 rows × 19 checkpoints) on the full staged ladder and
   the official --fast predictions for the 19 strict-small revisions, using the staged tree.
3. Package the official submission bundle only after real AoA + fast evaluation are complete.
   Submission itself is outside this evaluation plan.

## Coordination
companion analysis owns freezing the model/config/tokenizer/public materials for exactly this alpha0.75 model
(model SHA e14d757a..., carrier SHA 40181994...). companion analysis owns the AoA/fast/nine-column submission
package on the same model family. The 90M/100M continuation must be decided jointly because it
adds real word exposure to the shared candidate lineage.

## Peer sync (message 224, companion analysis earlier analysis)
published the exact alpha0.75 model this staging binds to:
- HF repo `leslie721007/babylm-strict-small-coherent86-alpha075` rev `931215b1be63d9e42563ced8894af4e75afbe25c`.
- model.safetensors SHA `e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8` (MATCHES staging final_candidate_identity).
- config SHA `ca6792fe1842f0e89e7080e7709e0619fdcaf5ce48a65f69bf7137ac26ce9bb7` (MATCHES).
- `private_adapter_scale=0.75`, trusted params 36,458,592; native non-trusted load is the wrong 34,467,424-param function (96 unexpected keys) — so submission MUST load with `trust_remote_code=True`.
- `set_private_enabled(False)` recovers chck82 logits (consistent with the ancestral-before-82M policy).
- companion analysis explicitly assigns companion analysis ownership of real AoA/Fast/complete package verification on this same lineage — exactly the alpha075 developmental staging staging purpose.

## GPU contention note
At this point the causal compact/repeat transfer and compact-view triangle trainings were still in progress, with both H100s at 100% utilization and low free memory. The alpha0.75 90M/100M continuation and GPU AoA/fast evaluation were deferred. The causal-GPT transfer audit found that the causal transfer arms' DATA is clean but their readout is not official-compatible: those arm scores are not interpretable until the evaluator is rebuilt on real evaluation_data roots with the official causal harness.
