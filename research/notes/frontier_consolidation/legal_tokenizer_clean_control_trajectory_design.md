# legal tokenizer clean control trajectory design legal-tokenizer clean-control trajectory design

## Why this experiment is being run

The fully legal spatial repair route status-tokenizer compact-view-reinvest endpoint reached Overall 41.257770896404615, below the live Strict-Small leader 41.8. The old 42.0331347900748 compact-view-reinvest endpoint remains strong mechanism evidence but is not submission-valid because its tokenizer was trained outside the Strict-Small 10M tokenizer-data budget. The current scientific question is therefore not yet which new objective to invent, but whether the compact-view reinvestment treatment effect survives under the legal spatial repair route status tokenizer coordinate when compared to a matched clean-Qwen control.

The previous earlier analysis attempt to train a 20M clean control failed before metrics due to OOM and would have been scientifically weak even if successful: the trajectory calibration shows early treatment direction can reverse by 100M, with mature cross-task information only around 70M/80M. The base trainer cannot resume. A 20M-only clean run would therefore risk forcing duplicate training if ambiguous.

## Expensive-work decision

One memory-safe clean-Qwen trajectory to 80M is the shortest single run that can answer both early and mature questions without duplicate training. It retains 1M checkpoints, so `chck_20M`, `chck_70M`, and `chck_80M` are available for matched comparisons against the already trained spatial repair route status-tokenizer reinvest trajectory.

Decision role of the run:

- If reinvest exceeds clean broadly at 70M/80M under the same spatial repair route status legal tokenizer, the compact-view reinvestment mechanism survives the legal tokenizer coordinate. The next research should preserve compactness and source reinvestment while improving legal tokenizer / optimization efficiency.
- If reinvest is weak, vanished, or reversed at 70M/80M, the old mechanism does not transfer reliably under the legal tokenizer coordinate. The next research should turn to learning-signal repairs such as paired source-view consistency or information/relation-weighted masking rather than simply running another endpoint.
- 20M is retained only as early behavior evidence; it must not be used as the sole route selector.

## Frozen control recipe

Training wrapper: `experiments/archive/frontier_consolidation/scripts/wait_and_train_clean_control_80m.py`.

Underlying launcher: `experiments/archive/frontier_consolidation/scripts/train_compliant_model.py`.

Training was launched; completion is not established here.

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/wait_and_train_clean_control_80m.py \
  --gpu 0 \
  --run-dir experiments/archive/frontier_consolidation/training/runs/complianttok_cleanqwen_seed43022_80M \
  --max-word-exposure 80000000 \
  --min-free-mib 72000 \
  --wait-timeout-sec 21600 \
  --poll-sec 120 \
  --check-hash \
  --count-words
```

Preflight evidence from legal tokenizer clean control trajectory design:

- clean-Qwen stream: `experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_100M.jsonl`
- SHA256: `728192f8e8c5855aaa52a6b6940ff3a4fd6aa8f87c98aca4ff018dbc04cb0345`
- row/word count: 643,810 rows, exactly 100,000,000 words
- tokenizer: `experiments/archive/frontier_consolidation/data/compliant_tokenizer`
- tokenizer SHA in metadata: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`
- model/optimization: DeBERTa-v2 8x480, 8 heads, ffn_mult 4, seq256, batch256, WWM fixed 0.15, AdamW lr 0.001, warmup 0.06, weight decay 0.01, seed 43, init seed 43022, train RNG seed 43023, checkpoint every 1M words
- run is a fixed-tokenizer scientific control only; tokenizer-training text plus clean-Qwen pretraining text exceeds a 10M union, so it is not a submission endpoint.

## Matched reinvest references

The reinvest trajectory already exists:

`experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2`

It contains checkpoints from 1M through 100M, including `chck_20M`, `chck_70M`, and `chck_80M`.

Reinvest `chck_20M` cheap-column evaluation was launched, but its completed result was unavailable when this note was written.

A mature-reference evaluation was launched:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/eval_reinvest_mature_refs.py \
  --gpu 1 \
  --exposures 70 80 \
  --min-free-mib 60000 \
  --wait-timeout-sec 18000 \
  --poll-sec 120
```

Expected reinvest per-target outputs:

- `experiments/archive/frontier_consolidation/data/legal20m_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_20M.json`
- `experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_70M.json`
- `experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_80M.json`

## Planned clean evaluations after training delivery

After clean training completes with valid `scientific_metrics.json` and checkpoints, evaluate the clean checkpoints on the same cheap columns as the reinvest references:

- BLiMP
- Supplement
- EWoK
- Entity
- COMPS
- GlobalPIQA_parallel
- GlobalPIQA_nonparallel
- Reading

Use an isolated output root, e.g.:

- `experiments/archive/frontier_consolidation/data/legal_mature_clean_control_eval`
- `experiments/archive/frontier_consolidation/data/legal_mature_clean_control_collate`

Suggested per-target names:

- `complianttok_cleanqwen_seed43022_20M`
- `complianttok_cleanqwen_seed43022_70M`
- `complianttok_cleanqwen_seed43022_80M`

Do not run SuperGLUE/AoA for this mechanism-control screen unless a later decision needs a full endpoint-like scalar. The scientific comparison is the matched cheap-column trajectory under identical tokenizer, architecture, seed, objective, and exposure.

## Comparison script

CPU summarizer prepared in legal tokenizer clean control trajectory design:

`experiments/archive/frontier_consolidation/scripts/compare_legal_treatment_trajectory.py`

Default output:

- `experiments/archive/frontier_consolidation/data/legal_treatment_trajectory/legal_treatment_trajectory_comparison.json`
- `research/documents/frontier_consolidation/data/legal_treatment_trajectory/legal_treatment_trajectory_comparison.md`

The script computes per-column reinvest-minus-clean deltas and a seven-column mean using BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA mean, and Reading. It also records raw eight-column deltas before averaging the two GlobalPIQA families.

At note time the comparison file exists but is incomplete because the clean trajectory and mature reinvest refs have not yet terminal-delivered.
