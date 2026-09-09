# state probe and fullcycle wwm launch — state-learning probe and full-cycle WWM baseline launch

## State-learning probe result

Script run:

- `experiments/archive/initial_model_studies/scripts/probe_state_learning.py`

Output:

- `experiments/archive/initial_model_studies/data/state_learning_probe.json`

The probe compared official WWM seed42/43 and state coherent vs corrupted 1m profile state coherent/corrupted seed42/43 models on:

- the state story materialization and smoke training-template procedural distribution (`train_seed42` sample, 120 pairs);
- the disjoint-vocabulary held-out procedural probe (120 pairs sampled from 480 rows).

The important result is that the state-trained models are essentially indistinguishable from the official WWM baselines and from each other. They do not show a strong learned state-update capability.

Compact summary:

| model | split | coherent acc | corrupted acc | both-context acc | coherent margin | corrupted margin | event sensitivity |
|---|---|---:|---:|---:|---:|---:|---:|
| official WWM seed42 | train | 0.500 | 0.500 | 0.000 | -0.0193 | +0.0193 | ~0 |
| official WWM seed43 | train | 0.533 | 0.467 | 0.000 | +0.0249 | -0.0249 | ~0 |
| state coherent seed42 | train | 0.517 | 0.483 | 0.000 | +0.0025 | -0.0025 | ~0 |
| state corrupted seed42 | train | 0.517 | 0.483 | 0.000 | +0.0025 | -0.0025 | ~0 |
| state coherent seed43 | train | 0.525 | 0.475 | 0.000 | -0.0048 | +0.0048 | ~0 |
| state corrupted seed43 | train | 0.525 | 0.475 | 0.000 | -0.0048 | +0.0048 | ~0 |
| official WWM seed42 | held-out | 0.575 | 0.425 | 0.000 | +0.2294 | -0.2294 | ~0 |
| official WWM seed43 | held-out | 0.575 | 0.425 | 0.000 | +0.1019 | -0.1019 | ~0 |
| state coherent seed42 | held-out | 0.583 | 0.417 | 0.000 | +0.3472 | -0.3472 | ~0 |
| state corrupted seed42 | held-out | 0.583 | 0.417 | 0.000 | +0.3472 | -0.3472 | ~0 |
| state coherent seed43 | held-out | 0.575 | 0.425 | 0.000 | +0.0898 | -0.0898 | ~0 |
| state corrupted seed43 | held-out | 0.575 | 0.425 | 0.000 | +0.0898 | -0.0898 | ~0 |

Interpretation:

- The simple state story materialization and smoke/69 procedural-state setup was not strongly learned as a state-update task.
- The identical coherent/corrupted model behavior implies the 2.7k state target tokens at 1M were too weak or too stereotyped relative to the dominant official WWM loss.
- The official Entity null result in state coherent vs corrupted 1m profile is therefore not a clean learned-but-no-transfer result; it is mostly a failure to induce a strong internal procedural-state capability.
- Do not make another nearby story variant immediately. If this family is revisited, the design must change target budget, update geometry, or architecture, and must be checked on the procedural probe before official evaluation.

## Full-cycle WWM baseline construction

A full-scale end-to-end baseline is required: local 1M probes alone cannot establish full-budget performance.

I cloned the trusted masked trainer to:

- `experiments/archive/initial_model_studies/training/scripts/babylm_masked_train_fullcycle.py`

The clone differs only in official-corpus selection for non-JSONL runs: when requested exposure exceeds the unique official corpus, it caps the unique pool at the 10M official corpus and constructs epoch-wise shuffled passes until the requested exposure is reached. Repeated word exposures are counted and recorded in `selection_epochs` inside `example_order_manifest.json`. The trusted original `babylm_masked_train.py` was not modified.

Smoke run:

- run id: `babylm_fullcycle_smoke_20k`
- script compiled and trained successfully.
- Note: the small smoke requested 20k exposure, so the pre-existing safety logic expanded the unique pool to 20k and did not exercise multi-epoch cycling. For the real 100M run, requested exposure exceeds the 10M corpus, so the patched full-cycle path caps the pool at 10M and cycles epochs.

## Full 100M protected WWM baseline launched

Background task launched:

- task label: `state probe and fullcycle wwm launch full-cycle WWM 100M protected baseline`
- intended run id: `babylm_fullcycle_wwm_seed42_100M`

Configuration:

- official corpus: `BabyLM-community/BabyLM-2026-Strict-Small`, revision `c92ab16b4f08858304b0815706065b3354d8fc0a`
- exposure: 100,000,000 words = 10 official epochs over the 10M corpus
- model: BERT MLM 8 layers, hidden 256, 8 heads, FFN mult 4
- tokenizer: baseline16k
- objective: WWM, mask probability 0.15
- sequence length: fixed 256, position capacity 512
- optimizer: AdamW, lr 0.001, cosine schedule over 1221 steps
- batch size: 512 examples (160 words/example; about 81,920 words/step)
- checkpoint interval: 1,000,000 words, giving dense checkpoint evidence for later official fast trajectory and submission-material repair if needed
- seed/init/RNG: 42/456/789

Expected outputs:

- `experiments/archive/initial_model_studies/training/runs/babylm_fullcycle_wwm_seed42_100M`
- `scientific_metrics.json`, `example_order_manifest.json`, `tokenization_coupling_summary.json`, `hf_model/`, checkpoint branches under `hf_model/chck_*M`.

After the run finishes, the next work is to verify actual exposure, epoch metadata, checkpoint set, save/load, and then begin official nine-column coordinate evaluation (full zero-shot, GlobalPIQA, AoA, and SuperGLUE fine-tuning). This is baseline evidence, not final submission.
