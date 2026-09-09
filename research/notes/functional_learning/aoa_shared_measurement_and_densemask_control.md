# aoa shared measurement and densemask control: Shared-ancestry AoA measurement and dense-mask/sparse-label control readiness

## Why this step mattered

The evaluation repair synthesis loader repair is now accepted infrastructure: `AutoModel.from_pretrained(..., trust_remote_code=True)` reaches the repaired `FrozenSlowPrivateDebertaV2Model` with exact hidden-state equality to the trusted MLM encoder. aoa shared measurement and densemask control therefore moved beyond the loader diagnosis toward restoring evaluation throughput and mechanism-testing throughput for the `Qiushi-BabyLM-36M-Strict-Small-v5` question.

Two problems were addressed:

1. **AoA must be measured efficiently and truthfully.** Coherent86 and both dense candidates share the same true ancestral trajectory through `chck_80M`, then differ only at their final endpoints. The correct computation is to evaluate the 17 common ancestral checkpoints once, preserve their provenance, evaluate each final endpoint separately, and assemble each candidate trajectory with the shared block plus its exact endpoint.
2. **A measured AoA score of exactly zero must not be confused with missing-checkpoint placeholder zero.** The official `AoAEvaluator` can return 0.0 after real extraction/scoring, e.g. through its significance/valid-word behavior. Measurement status must come from evidence files and scorer completion, not `score != 0`.

## New AoA script and evidence model

Built:

- `experiments/archive/functional_learning/scripts/shared_aoa.py`

The script supports:

- `shared`: compute shared ancestral checkpoint surprisal once.
- `endpoint`: compute one candidate endpoint surprisal.
- `assemble`: combine shared + endpoint evidence and run official `AoAEvaluator` scoring.
- `smoke`: run a bounded end-to-end extraction→serialization→assembly→scoring test.
- `status`: summarize which full blocks exist.

It writes explicit manifests containing:

- checkpoint config and weight SHA256 provenance,
- CDI word/context file SHA256 provenance,
- expected and observed result counts per step,
- finite/nonfinite surprisal counts,
- assembled trajectory step names and word counts,
- official `aoa_score.json`, richer `aoa_score_manifest.json`, and an `aoa_payload_for_comparison` that marks whether the score is measured.

## Important bug found and fixed

The first smoke run produced complete extraction/serialization but scoring warnings because the endpoint step was named `final`. The official `AoAEvaluator.extract_step_number` cannot parse `final`, so endpoint entries were ignored during curve fitting.

Repair: endpoint steps are now labeled with parseable actual exposure:

- coherent86: `endpoint_86.005295M`
- dense seeds: `endpoint_89.168037M`

This is scientifically important because the final endpoint can influence the fitted acquisition curve; it must not be silently dropped.

## End-to-end smoke results

### 3-word / 2-ancestral-checkpoint smoke

Command:

```bash
python experiments/archive/functional_learning/scripts/shared_aoa.py smoke --target coherent86 --max-words 3 --ancestral-limit 2 --gpu 0
```

Evidence:

- `experiments/archive/functional_learning/data/shared_aoa/shared_ancestry/test_w3_a2/manifest.json`
- `experiments/archive/functional_learning/data/shared_aoa/endpoints/coherent86/test_w3/manifest.json`
- `experiments/archive/functional_learning/data/shared_aoa/assembled/coherent86/test_w3_a2/aoa_manifest.json`

Result:

- Shared block: 66/66 expected finite entries.
- Endpoint block: 33/33 expected finite entries.
- Assembled trajectory: 99/99 finite entries, no missing steps, no wrong counts.
- Official scorer completed and returned raw correlation 0.0 / leaderboard 0.0. The manifest marks this as a **measured zero** with evidence, not a missing-checkpoint placeholder.

### 1-word / all-17-ancestral-checkpoint smoke

Command:

```bash
CUDA_VISIBLE_DEVICES="" python experiments/archive/functional_learning/scripts/shared_aoa.py smoke --target coherent86 --max-words 1 --ancestral-limit 17 --gpu 0
```

Evidence:

- `experiments/archive/functional_learning/data/shared_aoa/assembled/coherent86/test_w1_a17/aoa_manifest.json`

Result:

- Shared all-17 block: 221/221 finite entries.
- Endpoint block: 13/13 finite entries.
- Assembled trajectory: 234/234 finite entries over all 18 steps, no missing steps, no wrong counts.
- Official scorer completed and returned measured zero because there were too few valid words for correlation. This confirms the full early-stop step list can execute through the scoring path.

## Full AoA jobs submitted without H100 contention

Full AoA evaluation reused shared ancestral inference and was in progress on CPU at the time of this note:

- full shared 17-checkpoint ancestral extraction, output `experiments/archive/functional_learning/data/shared_aoa/shared_ancestry/full`.
- coherent86 endpoint extraction, output `experiments/archive/functional_learning/data/shared_aoa/endpoints/coherent86/full`.
- dense seed62064 endpoint extraction, output `experiments/archive/functional_learning/data/shared_aoa/endpoints/dense_seed62064/full`.
- dense seed62065 endpoint extraction, output `experiments/archive/functional_learning/data/shared_aoa/endpoints/dense_seed62065/full`.

After these complete, run:

```bash
python experiments/archive/functional_learning/scripts/shared_aoa.py assemble --target coherent86
python experiments/archive/functional_learning/scripts/shared_aoa.py assemble --target dense_seed62064
python experiments/archive/functional_learning/scripts/shared_aoa.py assemble --target dense_seed62065
```

The resulting manifests under `experiments/archive/functional_learning/data/shared_aoa/assembled<target>/full/aoa_manifest.json` should be used by the comparison script.

## Trusted comparison update

Built:

- `experiments/archive/functional_learning/scripts/trusted_comparison_with_aoa.py`

It extends evaluation repair synthesis comparison logic by deciding AoA measurement status from aoa shared measurement and densemask control manifest evidence, not from nonzero score. It currently remains incomplete because the full repaired SuperGLUE payloads and full AoA manifests are not finished.

Current output:

- `experiments/archive/functional_learning/data/trusted_comparison_with_aoa/trusted_comparison_with_aoa.json`
- `.md` companion file.

Important semantics:

- `projected_overall_aoa0` remains conditional arithmetic with shared placeholder zero.
- `measured_overall` requires `complete_measured_evidence` from the aoa shared measurement and densemask control AoA manifest.
- A full measured AoA score may be exactly 0.0 and still count as measured if extraction and official scoring completed.

## Dense-mask/sparse-label causal control verification

The dense focus training changed both input geometry and supervised target coverage. The prepared control keeps sparse labels but applies dense masks, testing whether dense input-side clue removal can account for the source-responsive/entity gains without the full dense target coverage.

Built a row-level invariant verifier:

- `experiments/archive/functional_learning/scripts/verify_densemask_sparselabel_rows.py`

Evidence:

- `experiments/archive/functional_learning/data/densemask_sparselabel_row_verify/row_verify.json`
- `.md` companion file.

Result:

- Status: PASS.
- Checked 33 first-macro Qwen rows.
- Dense-mask/sparse-label labels exactly equal sparse labels, including label values.
- Dense-mask/sparse-label masks exactly equal dense masks.
- Labels are subsets of masks.
- Aggregate label/mask tokens: 247 / 1598, ratio 0.15456821026282855.

This confirms the first-macro implementation of the causal contrast, not the training outcome. The full 80-update profile already predicts roughly the same ratio: 28,590 sparse label tokens / 176,607 dense mask tokens = 0.161884863 for seed62064.

## Current unresolved evidence

- Repaired coherent86 and dense seed62064 SuperGLUE evaluations were in progress; their results were not yet incorporated here.
- Dense seed62065 repaired SuperGLUE had not started.
- Full shared-ancestry and endpoint AoA extractions were in progress on CPU; assembly and scoring were incomplete.
- Dense-mask/sparse-label training remained a proposed contrast, conditional on the repaired evaluations supporting further investigation of dense acquisition.
