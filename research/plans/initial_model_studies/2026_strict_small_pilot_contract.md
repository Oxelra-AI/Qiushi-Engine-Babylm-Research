# earlier analysis — 2026 Strict-Small Pilot Experiment Contract

Phase: Explore / construction specification  
Purpose: convert the verified 2026 rules, leaderboard, dataset, and evaluation code into a first executable research scaffold. This file is not a final report and does not claim a result.

## 1. Official score object to optimize

The current live 2026 leaderboard Space was parsed from:

- `https://huggingface.co/spaces/BabyLM-community/BabyLM-Leaderboard-2026`
- saved parse: `experiments/archive/initial_model_studies/data/leaderboard_probe/leaderboard_config_parsed.json`
- leaderboard source snapshots: `experiments/archive/initial_model_studies/data/leaderboard_probe/src__leaderboard__read_evals.py`, `src__submission__eval_submission.py`, `src__submission__check_validity.py`

For Strict and Strict-Small, the current Space computes the displayed averages as follows (`src__leaderboard__read_evals.py`, lines 360-377):

```text
NLP_BENCHMARK_KEYS = ("blimp", "blimp_supplement", "ewok", "entity_tracking", "comps", "glue")
global_piqa = results["global_piqa"] or 0
nlp_vals = [BLiMP, BLiMP-Supplement, EWoK, Entity Tracking, COMPS, (Super)GLUE] + [GlobalPIQA]
human_vals = [Reading, AoA]
NLP Average        = mean(nlp_vals)       # 7 columns
Human-like Average = mean(human_vals)     # 2 columns
Overall Average    = mean(nlp_vals + human_vals)  # 9 columns
```

Important consequences:

- The 2026 public Space is **not** the 2025 Findings 50/50 macro average. Each of the 9 displayed benchmark columns has equal weight in Overall.
- Human-like performance has 2/9 of Overall, not 1/2, but current Strict-Small public rows have extremely low AoA/human-like values, so a valid AoA/Reading gain is still a high-leverage route.
- Missing tasks count as zero. Leaderboard code explicitly zeroes old/missing `aoa_surprisals` and old/missing `entity_tracking_filtered`; submissions must be collated with the current 2026 pipeline.
- Reading is itself the mean of self-paced reading and eye-tracking delta %R2 scores. AoA is taken from the new `AoA_word/aoa_score.json` plus `aoa_surprisals` presence.

Current parsed public Strict-Small target from the live config:

| rank | model | overall | NLP | human-like | notes |
|---:|---|---:|---:|---:|---|
| 1 | `wwm_curriculum_simplification_40k` | 41.80 | 52.97 | 2.71 | strong NLP, AoA 0.0, Reading 5.42 |
| 2 | `RecGPT-10M` | 41.53 | 52.40 | 3.46 | strong BLiMP, low entity, AoA 0.0 |
| 3 | `Wordpiece-24-4` | 41.31 | 52.85 | 0.92 | tokenizer route, AoA 0.0 |
| 9 | `deberta-base-75k-sam_ext-s1` | 40.62 | 48.74 | 12.19 | much better human-like via AoA 22.9 but lower NLP |

A model with the current top NLP (~53) and modest human-like 8 would score `(7*53 + 2*8)/9 = 43.0`, enough to beat 41.8. A model with NLP 51.5 and human-like 12 would score 42.72. Therefore the shortest principled route is not merely maximizing GLUE/BLiMP; it is preserving competitive NLP while producing valid AoA/Reading improvements.

## 2. Official data and budget anchor

Use the official 2026 Strict-Small training set unless a later evidence-backed data intervention justifies replacing part of it.

- HF dataset: `BabyLM-community/BabyLM-2026-Strict-Small`
- dataset commit / sha from HF API: `c92ab16b4f08858304b0815706065b3354d8fc0a`
- license tag: MIT
- description: Detoxified 10M Strict-Small BabyLM Training Dataset, total 10M tokens/words as advertised by BabyLM 2026
- files:
  - `bnc_spoken.train.txt` (~4.06 MB)
  - `childes.train.txt` (~15.16 MB)
  - `gutenberg.train.txt` (~14.03 MB)
  - `open_subtitles.train.txt` (~12.16 MB)
  - `simple_wiki.train.txt` (~8.86 MB)
  - `switchboard.train.txt` (~0.12 MB)

Budget/accounting contract:

- Training corpus must remain at or below 10M whitespace-separated words.
- Total exposure must remain at or below 100M words for Strict-Small.
- If tokenizer is trained, its language-data exposure counts against the same 10M corpus source. The default pilot should train tokenizer only on the official 10M corpus or use the official baseline 16,384-token tokenizer; do not import external linguistic tokenizers trained on larger language corpora.
- Synthetic/augmented data is not part of the pilot. If later used, augmenter training text must be counted under the closed-system rule.
- Required checkpoint names for HF/evaluation compatibility: `chck_1M` ... `chck_9M`, `chck_10M`, `chck_20M`, ..., `chck_100M`, plus final `main`.

## 3. Official evaluation command surface

Repository:

- `experiments/archive/initial_model_studies/repos/babylm-eval`
- current checked commit: `6f825c2` (`Fix fast eval collation for EWoK`)

Evaluation data setup requirements:

```bash
cd experiments/archive/initial_model_studies/repos/babylm-eval/strict
python scripts/download_evals.py
python evaluation_pipeline/global_piqa/dl.py
# full EWoK requires accepting terms at https://huggingface.co/datasets/ewok-core/ewok-core-1.0, then:
python -m evaluation_pipeline.ewok.dl_and_filter
# unzip fast EWoK zip if needed with password BabyLM2025
```

Full evaluation and submission sequence for a model path or HF repo:

```bash
cd experiments/archive/initial_model_studies/repos/babylm-eval/strict
bash scripts/eval_zero_shot.sh <MODEL_PATH> causal evaluation_data/full_eval main
bash scripts/eval_zero_shot_global_piqa.sh <MODEL_PATH> causal strict-small evaluation_data/full_eval evaluation_data/fast_eval
bash scripts/eval_aoa.sh <MODEL_PATH> causal strict-small evaluation_data/full_eval/aoa/cdi_childes.json results
bash scripts/eval_finetuning.sh --model_path <MODEL_PATH> --lr 3e-5 --bsz 32 --big_bsz 16 --max_epochs 10 --wsc_epochs 30 --seed 42
bash scripts/eval_zero_shot_fast_all_revisions.sh <MODEL_PATH> causal strict-small evaluation_data/fast_eval
bash scripts/collate_preds.sh <MODEL_PATH> causal strict-small
```

Pilot evaluation should start smaller than full official evaluation:

1. Verify model loads with `AutoModelForCausalLM` and `AutoTokenizer`.
2. Run a single fast zero-shot task on `main` or `chck_1M` to validate interface and output directories.
3. Run the fast set for one or two checkpoints (`chck_1M`, `chck_2M`) by editing a local copy of the loop script or directly invoking `eval_zero_shot_fast.sh`.
4. Only after interface and dataset downloads are confirmed should a full 100M-exposure schedule be launched.

## 4. Scientific mechanism under test

The competitive analysis and breakthrough gap literature picture showed a MoEP/AMLM tradeoff under the 2025 metric: MoEP was stronger on human-likeness; AMLM was stronger on NLP. The 2026 eval and experiment plan/4 2026 audit changes the exact target but strengthens the same scientific bottleneck: public 2026 Strict-Small top rows largely obtain Overall through NLP columns while AoA is often 0 or negative. The first experiment should test whether a model can preserve competitive causal-LM NLP while generating a valid developmental trajectory signal.

Core hypothesis:

> In 10M-word pretraining, a causal model needs separate representational channels for (i) broad next-token distributional competence and (ii) word-level developmental trajectory / morphology-sensitive learning. Sparse routing alone may diversify computation; difficulty-aware auxiliary prediction alone may over-specialize hard tokens; a small morphology-aware side channel tied into routing may raise AoA/Reading without collapsing BLiMP/GLUE.

This is stronger than a generic MoEP+AMLM mixture because it predicts which columns should move:

- sparse routing: improves Reading/AoA/entity trajectory and may help early fast-eval curves;
- difficulty-aware auxiliary denoising: improves BLiMP, COMPS, GLUE, maybe GlobalPIQA, but risks AoA collapse if over-weighted;
- multi-granularity morphology side channel: improves COMPS, entity tracking, morphology-sensitive sub-behavior and AoA without replacing the causal scorer.

## 5. First implementation scaffold

The pilot implementation is `experiments/archive/initial_model_studies/training/scripts/babylm_pilot_train.py`.

### Script contract

Create in the next Execute step:

```text
experiments/archive/initial_model_studies/training/scripts/babylm_pilot_train.py
```

Inputs:

```text
--dataset_id BabyLM-community/BabyLM-2026-Strict-Small
--dataset_revision c92ab16b4f08858304b0815706065b3354d8fc0a
--variant dense_causal | sparse_causal | sparse_aux | sparse_aux_morph
--tokenizer baseline16k | train16k_on_official10m
--max_word_exposure 1000000  # pilot; later 100000000
--checkpoint_words 1000000   # pilot; later official schedule
--seed 42
--out_hf_dir $QIUSHI_AI_LAB_RUN_DIR/hf_model
--metrics_json $QIUSHI_AI_LAB_RUN_DIR/metrics.json
```

Outputs:

```text
$QIUSHI_AI_LAB_RUN_DIR/hf_model/                 # final main-compatible HF model
$QIUSHI_AI_LAB_RUN_DIR/hf_model/chck_1M/          # pilot checkpoint; for full run use all required names
$QIUSHI_AI_LAB_RUN_DIR/training_log.jsonl
$QIUSHI_AI_LAB_RUN_DIR/metrics.json               # losses, word exposures, token exposures, throughput, parameter count
$QIUSHI_AI_LAB_RUN_DIR/data_manifest.json          # dataset id, revision, files, word counts, hashes if downloaded
$QIUSHI_AI_LAB_RUN_DIR/tokenizer_manifest.json     # tokenizer source/training exposure
```

Proposed pilot scientific arguments (run label `babylm_pilot_dense_1M`):

```text
  --dataset_id BabyLM-community/BabyLM-2026-Strict-Small \
  --dataset_revision c92ab16b4f08858304b0815706065b3354d8fc0a \
  --variant dense_causal \
  --tokenizer baseline16k \
  --max_word_exposure 1000000 \
  --checkpoint_words 1000000 \
  --seed 42
```


### Variants

Keep the first variants small enough to fit fast iteration, but HF-compatible:

**V0: dense_causal**

- GPT2LMHeadModel-style decoder, smaller than official 98.4M if needed for pilot speed.
- Use vocab 16,384 and context 512 or 1024.
- Purpose: verify corpus ingestion, checkpoint naming, HF save/load, and one fast eval.

**V1: sparse_causal**

- MoEP-like decoder: early dense layers, middle routed parallel blocks, later dense layer.
- Must subclass or wrap into a HF-compatible causal LM with `config.json`, `AutoModelForCausalLM` loading via `trust_remote_code=True` if necessary.
- Router outputs/logs should be saved: entropy, load balance, route distribution by token frequency and by word type.
- Purpose: test whether path diversity changes Reading/AoA/entity behavior before adding objective complexity.

**V2: sparse_aux**

- V1 plus a small auxiliary denoising/masked-span head active only during training.
- Difficulty weights are estimated from token-level causal loss EMA on the official corpus, not from evaluation data.
- Keep causal LM head primary and evaluation-compatible.
- Start with low auxiliary weight (e.g. 0.1 after warmup), because AMLM-Hard-Decay showed human-like collapse when difficulty focus dominates.

**V3: sparse_aux_morph**

- V2 plus a small character/morpheme side encoder tied to token embeddings.
- The side channel must learn only from the official 10M text. No external morphological segmenter trained on outside language data.
- Minimal first design: character-CNN or byte-CNN over token string, gated into token embedding with a learned scalar gate; log gate magnitude by token frequency and suffix pattern.

### Stopping and scale-up logic

Pilot (`1M` exposure): success means the scaffold runs, saves a loadable HF model and `chck_1M`, produces valid predictions for at least one fast task, and records word-exposure counts. Pilot scientific conclusions should be weak; it is for feasibility and early signal.

First comparative run (`10M` exposure): compare V0/V1/V2 on fast zero-shot curves and validation loss, with `chck_1M`...`chck_10M`. Stronger if V1 or V2 improves Reading/entity/BLiMP tradeoff without loss collapse.

Full candidate run (`100M` exposure): only after one variant shows promising fast-eval pattern. Save all official checkpoints. Run full eval and collate.

## 6. Immediate next Execute work

1. Implement `babylm_pilot_train.py` with V0 first; do not implement all variants before proving the pipeline.
2. Download or stream official Strict-Small training files with revision pinning and compute whitespace word counts per file.
3. Use the official baseline 16k tokenizer for V0 if easily accessible from `BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small`; otherwise train a 16k BPE on the official 10M and record that tokenizer training exposure uses the same corpus.
4. Train V0 to 1M word exposure and save `hf_model/chck_1M` plus `hf_model` main.
5. Validate HF loading and run one fast zero-shot evaluation script from `babylm-eval/strict` against the pilot checkpoint.
6. Only then implement V1 sparse routing.

## 7. Open risks

- Full EWoK evaluation requires accepting dataset terms; local full eval may be delayed until terms/data access are available. Fast or partial eval can still validate the interface.
- The leaderboard backing dataset is private/401 to unauthenticated API calls; current public scores come from the Gradio config, which is enough for route selection but should be re-parsed before any submission claim.
- Custom sparse architecture may require `trust_remote_code=True` and careful HF repository packaging. The first V0 dense model avoids this risk.
- AoA scoring is sensitive to all required checkpoint surprisals. A model can receive 0 AoA if the new `aoa_surprisals` key is missing even if the final model is good.
