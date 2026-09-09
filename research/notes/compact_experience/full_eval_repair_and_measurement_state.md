# full eval repair and measurement state mixed-objective full-evaluation repair and measurement state

## Why this step changed the evaluator before running

mixed objective measurement and mechanism plan left a scientifically important measurement dependency: the causal attention verification mixed-objective models had completed training, but the complete nine-column evaluation scripts still allowed several ways for a result to stop being the score of the two frozen causal attention verification 100M checkpoints. I repaired the measurement path before spending the dual-H100 evaluation run.

The repair follows :

1. **Old output isolation.** The launcher now writes to a fresh root: `data/mixed_objective_full_eval_100M_strict/`, not the older mixed objective measurement and mechanism plan root. Preflight refuses to run if `per_target/*.json` already exists there.
2. **Exact target set.** The scorer now accepts exactly two JSONs: `mixed_causal50_100M.json` and `mixed_causal15_100M.json`. Extra/missing per-target JSONs are treated as failure rather than silently entering the ranking.
3. **Unconditional Overall recomputation.** `score_mixed_objective_full_eval.py` now recomputes Overall from the current task records after AoA normalization and stores the old cached value only as provenance.
4. **AoA integrity.** `full_overall_eval_runner.py` now checks that the AoA helper returns status `AOA_LOCAL_CKPTS_DONE`, exactly the 19 strict-small steps (`chck_1M..9M`, `chck_10M..100M`), no missing or unexpected steps, a single nonzero row count, finite surprisals, and a finite raw AoA before marking `official_aoa_done`.
5. **Report parsing.** The old loose zero-shot parser fallback to the last number in the report was removed; only explicit official average headers are accepted.
6. **Model identity/provenance.** The launcher records endpoint `model.safetensors` SHA256 hashes in `preflight.json`; the scorer checks the payload identity, endpoint path, training metrics, intended and realized causal fractions, full checkpoint list, and hash consistency.
7. **Shell failure propagation.** The launcher now exits nonzero if either evaluator or the strict scorer fails.

Static checks passed:

```bash
bash -n experiments/archive/compact_experience/scripts/launch_mixed_objective_full_eval_100M.sh
PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
  experiments/archive/compact_experience/scripts/score_mixed_objective_full_eval.py \
  experiments/archive/compact_experience/scripts/full_overall_eval_runner.py
```

## SuperGLUE local definition audit

The local full evaluator fine-tunes and predicts on seven official-filtered tasks: BoolQ, MultiRC, RTE, WSC, MRPC, QQP, and MNLI. `experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_pipeline/calculate_results_from_pred.py` defines `_calculate_glue_results` by computing accuracy for each subtask and returning the unweighted mean across subtasks. This matches the local scorer's SuperGLUE column.

A caveat remains for server equivalence: `scripts/print_results_table.py` in the same repository maps MRPC and QQP to F1 for markdown reporting. Therefore the local number should be treated as official-style and internally consistent with `calculate_results_from_pred.py`, but final submission equivalence still needs direct server-compatible collation/official-server validation if a candidate becomes strong.

## Mixed-objective training metrics that guide later separator design

These are from causal attention verification training metrics only, not from evaluation scores or AoA internals.

| arm | causal fraction | MLM batches | causal batches | MLM targets | causal targets | total targets/word | MLM targets per MLM-batch word | estimated WWM expansion vs 0.15 | dense-MLM WWM mask prob to match total targets |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| causal50 | 0.50 | 1258 | 1257 | 10,791,423 | 71,614,727 | 0.8240615 | ~0.2157 | ~1.438 | ~0.573 |
| causal15 | 0.15 | 2138 | 377 | 18,354,321 | 21,483,561 | 0.39837882 | ~0.2159 | ~1.439 | ~0.277 |

If either mixed-objective arm has useful complete-eval signal, the first separator should not immediately claim causal-objective complementarity. A dense-MLM separator can match total supervised target count approximately with fixed WWM mask probability around 0.57 (for causal50) or 0.277 (for causal15), while recording that high-mask MLM changes context corruption and is not equivalent to uncorrupted causal supervision. A stronger separator may need to control three quantities separately: supervised targets/word, unique target positions/word, and corrupted positions/word.

## Active background measurement

The specified evaluation had started; completion was pending:

```bash
bash experiments/archive/compact_experience/scripts/launch_mixed_objective_full_eval_100M.sh
```

Expected result files:

- `data/mixed_objective_full_eval_100M_strict/preflight.json`
- `data/mixed_objective_full_eval_100M_strict/per_target/mixed_causal50_100M.json`
- `data/mixed_objective_full_eval_100M_strict/per_target/mixed_causal15_100M.json`
- `data/mixed_objective_full_eval_100M_strict/mixed_objective_full_eval_summary.json`
- `data/mixed_objective_full_eval_100M_strict/mixed_objective_full_eval_summary.md`

Interpretation after delivery should first compare complete Overall against clean-Qwen `41.34429066479573` and the visible leader display `41.8`, then inspect the column profile. AoA remains only a terminal aggregate readout from the frozen full ladder.
