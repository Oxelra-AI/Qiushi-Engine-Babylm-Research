# chck82 endpoint evidence and reproduction state — chck_82M endpoint evidence and reproduction state

## Why this note exists

This is a research-facing synthesis for the first complete above-frontier BabyLM Strict-Small endpoint candidate. It keeps four threads separate:

1. the current public target to beat;
2. the measured score of the selected endpoint;
3. legal accounting and provenance evidence;
4. the still-running from-corpus training reproduction.

It is not a final submission package and it does not turn checkpoint exposure selection into the requested general learning principle.

## Current public Strict-Small target

`experiments/archive/representation_and_objectives/scripts/refresh_babylm_leaderboard_live.py` was rerun in chck82 endpoint evidence and reproduction state. It fetched the Hugging Face leaderboard Space config at `2026-08-31T22:49:04.085632+00:00` and parsed 122 Strict-Small rows.

Top visible Strict-Small row from `experiments/archive/representation_and_objectives/data/babylm2026_live_surface/strict_small_top30.json` and `research/notes/representation_and_objectives/babylm2026_live_surface.md`:

| rank | model | repo | Overall | BLiMP | Supp | EWoK | Entity | COMPS | GlobalPIQA | SuperGLUE | Reading | AoA |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | wwm_curriculum_simplification_40k | go76dof/wwm_curriculum_simplification_40k | 41.80 | 67.20 | 56.01 | 56.07 | 28.45 | 53.57 | 39.67 | 69.79 | 5.42 | 0.0 |

The leader model card in `data/external/go76dof-wwm-curriculum-simplification-40k-Hugging-Face.md` describes a 34,677,952-parameter DeBERTa-v2 style MLM trained on 9,999,969 words of FineWeb simplification pairs, 40k SentencePiece BPE, LAMB, sequence length 64→256, WWM epochs 1–7 and token masking epochs 8–10, with leaderboard Overall 41.80.

## Official-source context now attached to the endpoint evidence

Official challenge/source files read in chck82 endpoint evidence and reproduction state support the following:

- Strict/Strict-Small leaderboard submission uses a predictions file produced by the BabyLM evaluation pipeline, with checkpoint outputs required for a full BabyLM Challenge submission. Source excerpt: `data/external/BabyLM-Evaluation-2026.md`, lines 156–174.
- External language-learned tools and data augmentation must fit the same learned-word accounting; approved teacher families include Qwen 3.5 up to 9B. Source excerpt: `data/external/FAQs.md`, lines 126–141.
- The official evaluation repository describes Strict/Strict-Small zero-shot, human-likeness, and finetuned SuperGLUE columns. Source excerpt: `data/external/BabyLM-Evaluation-2026.md`, lines 208–244.

## Endpoint identity and preserved files

Canonical endpoint:

`experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M`

Preserved copy:

`experiments/archive/representation_and_objectives/data/chck82_endpoint_snapshot/checkpoint/chck_82M`

Snapshot manifest:

`experiments/archive/representation_and_objectives/data/chck82_endpoint_snapshot/manifest.json`

Important identity facts:

- `model.safetensors` SHA256: `93ceb76adf5a33d349f1de33e988e6ed0c2b2a547dbd92cf83cc952f8e2591b3`.
- Actual cumulative exposure at selected checkpoint: 82,012,495 words.
- Full training ladder: 100 checkpoints available from `chck_1M` to `chck_100M` in the source run; official AoA subset present.
- Config: `AdapterDebertaV2ForMaskedLM`, 8 layers, hidden size 480, 8 heads, vocab 16,384, adapter bottleneck 128, adapter scale 1.75.
- Training recipe: legal compact-view reinvest 10M pool and 100M stream, legal tokenizer trained from the 10M pool, seeds 43/43022/43023, AdamW LR 0.001, warmup 0.06, fixed WWM 0.15, batch 256, seq length 256, checkpoint cadence 1M words, 100M LR horizon.

The earlier analysis preflight dossier at `experiments/archive/representation_and_objectives/data/chck82_reproducibility_preflight/chck82_reproducibility_preflight.json` verifies:

- 10M pool SHA: `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`.
- 100M stream SHA: `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`.
- tokenizer vocab size 16,384 and endpoint-vs-training vocab maps identical.
- fixed selected checkpoint for reproduction: `chck_82M`; no new neighboring-checkpoint search.

## Score measurement now repeated by two independent local paths

First complete score from endpoint sweep and dualview route:

`experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_verification/summary/scale1p75_chck82_full_verification.json`

Independent hardened evaluation reproduction from earlier analysis/167:

`experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/summary/scale1p75_100M_full_eval_hardened_summary.json`

Comparison file:

`experiments/archive/representation_and_objectives/data/chck82_eval_comparison/chck82_eval_comparison.json`

Repeated score vector:

| source | Overall | margin vs 41.80 | cheap7 | BLiMP | Supp | EWoK | Entity | COMPS | SuperGLUE | GlobalPIQA | Reading | AoA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| endpoint sweep and dualview route first complete score | 41.942254 | +0.142254 | 43.960000 | 68.49 | 62.94 | 50.06 | 28.31 | 52.19 | 69.760288 | 37.58 | 8.15 | 0.0 |
| earlier analysis hardened rerun | 41.942481 | +0.142481 | 43.959450 | 68.491284 | 62.937811 | 50.055453 | 28.314042 | 52.191175 | 69.766181 | 37.577670 | 8.148714 | 0.0 |

The hardened rerun changes Overall by only `+0.00022695351662349594`; maximum absolute column delta is `0.005893446487505116`, and all rounded leaderboard-style columns agree. This establishes measurement repeatability for the endpoint score under a fresh isolated evaluator, not training reproduction.

## From-corpus reproduction currently running





Output run:

`experiments/archive/representation_and_objectives/training/runs/adapter128_scale1p75_from_corpus_fixed82M_seed43022`

Prepared comparison script:

`experiments/archive/representation_and_objectives/scripts/compare_from_corpus_reproduction.py`

The comparison script was run in chck82 endpoint evidence and reproduction state while the training run was incomplete and correctly reported `PENDING` because `scientific_metrics.json` is not yet present. Output:

`experiments/archive/representation_and_objectives/data/from_corpus_reproduction_compare/from_corpus_reproduction_compare.json`

Early training-log inspection confirms the run is on the intended coordinate:

- first row loss `9.837543487548828`, matching the original ladder exactly;
- babylm2026 live surface cumulative exposure 39,370 words;
- `seq_len=256`, `mask_mode=wwm`, nominal mask probability `0.15`;
- early loss/warmup rows are finite and monotonically entering training.

Therefore there is no early wrong-recipe signal that would justify stopping the expensive run.

After the from-corpus reproduction completes, run:

`python3 -B experiments/archive/representation_and_objectives/scripts/compare_from_corpus_reproduction.py`

The decisive comparison is whether reproduced `chck_82M/model.safetensors` is bit-identical to SHA `93ceb76adf5a33d349f1de33e988e6ed0c2b2a547dbd92cf83cc952f8e2591b3`. If it is identical, the repeated endpoint sweep and dualview route/166 score evidence transfers to the regenerated endpoint. If it differs, evaluate the reproduced `chck_82M` itself with the hardened evaluator before making any score statement.

## Scientific separation from the open mechanism route

The score-bearing endpoint is an exposure-selected checkpoint from an existing legal scale1.75 adapter trajectory. It is a strong practical SOTA candidate, but it does not by itself explain the desired sample-efficient learning principle.

The central unresolved mechanism remains context-conditioned alternative binding. Prior evidence showed scale1.75 improves broad aggregate score but carries matched EWoK and GlobalPIQA hard-surface costs relative to its legal16k base; the cost is in the trained representation/trajectory rather than removable adapter-output amplitude. companion analysis's dual-view shared private pathway remains the strongest current representation-forming route, but companion analysis's broad 20M aligned-vs-shuffled panel reported broad-score damage and is being repaired there. companion analysis should not duplicate that implementation unless a later group result asks for a specific independent reproduction.
