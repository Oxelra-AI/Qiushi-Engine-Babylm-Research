# v0 1m pipeline and first signal — V0 1M Pipeline Clean and First Official Fast Signal

Purpose: finish the tokenizer/checkpoint portability repair, validate the official-compatible V0 lifecycle at the first meaningful 1M exposure point, and preserve the evidence for the next research decision.

## 1. Tokenizer portability repair is now clean

Earlier 10k runs (`babylm_pilot_dense_debug10k`, `babylm_pilot_dense_tokfix10k`, `babylm_pilot_dense_tokfix2_10k`) exposed a real portability failure: the official baseline tokenizer loads, but `save_pretrained` preserved `tokenizer_class: TokenizersBackend`, which clean `AutoTokenizer.from_pretrained` cannot import. The BabyLM eval script could still run by warning/fallback, but that is not acceptable for reproducible HF submission.

Final repair in `experiments/archive/initial_model_studies/training/scripts/babylm_pilot_train.py`:

- reconstruct the tokenizer as `PreTrainedTokenizerFast` from the baseline Rust backend;
- post-process each saved `tokenizer_config.json` inside `save_hf_checkpoint()` to force `tokenizer_class: PreTrainedTokenizerFast` and preserve `<s>`, `</s>`, `<unk>`, `<pad>`, `<mask>`.

Clean verification run:

- run dir: `experiments/archive/initial_model_studies/training/runs/babylm_pilot_dense_tokfix3_10k`
- verifier: `experiments/archive/initial_model_studies/scripts/verify_babylm_checkpoint.py`
- verification record: `runs/babylm_pilot_dense_tokfix3_10k/checkpoint_verify.json`
- both `hf_model/` and `hf_model/chck_1M/` load with:
  - `AutoTokenizer` class: `PreTrainedTokenizerFast`
  - `AutoModelForCausalLM` class: `GPT2LMHeadModel`
  - parameter count: 2,510,336
- local root path with `revision='chck_1M'` also loads successfully, matching the official eval loop's expected call pattern.
- official fast BLiMP on the clean 10k checkpoint still runs and scores 50.13, with no tokenizer-class warning.

## 2. Completed V0 1M pilot

Trainer: `experiments/archive/initial_model_studies/training/scripts/babylm_pilot_train.py`. Run label: `babylm_pilot_dense_v0_1M`; configured time limit: 1800 seconds. Recorded scientific arguments:

```text
  --dataset_id BabyLM-community/BabyLM-2026-Strict-Small \
  --dataset_revision c92ab16b4f08858304b0815706065b3354d8fc0a \
  --variant dense_causal --tokenizer baseline16k \
  --max_word_exposure 1000000 --checkpoint_words 1000000 \
  --seed 42 --seq_length 256 --words_per_example 160 \
  --batch_size 64 --n_layer 4 --n_embd 256 --n_head 4 --log_every 20
```

Artifacts:

- run dir: `experiments/archive/initial_model_studies/training/runs/babylm_pilot_dense_v0_1M`
- `data_manifest.json`: official 2026 Strict-Small dataset pinned to `c92ab16b4f08858304b0815706065b3354d8fc0a`, all six files hashed, total counted 10,000,000 whitespace words, selected exposure 1,000,000 words.
- `tokenizer_manifest.json`: official 16,384-token BabyLM baseline tokenizer source recorded.
- `hf_model/` and `hf_model/chck_1M/`: complete local HF model directories.
- `scientific_metrics.json`, `training_log.jsonl`, `stdout.log`, `stderr.log`: training record.

Training numbers:

- model: dense GPT2LMHeadModel-style causal decoder
- parameter count: 7,419,392
- context length: 256
- batch size: 64 examples × 160 whitespace words/example ≈ 10,240 words/batch
- steps: 98
- word exposure: exactly 1,000,000
- loss first: 9.7693
- loss last: 5.6379
- loss mean last 20: 5.7339
- elapsed inside script: 18.4 s; wall duration including download/setup: 138.9 s

Portability verification:

- verifier record: `experiments/archive/initial_model_studies/training/runs/babylm_pilot_dense_v0_1M/checkpoint_verify.json`
- both root and `chck_1M` load cleanly with `PreTrainedTokenizerFast` + `GPT2LMHeadModel`.
- local root with `revision='chck_1M'` loads successfully.

## 3. First official fast BLiMP signal at 1M words

Official evaluator command used the 2026 repo under:

`experiments/archive/initial_model_studies/repos/babylm-eval/strict`

and ran:

```bash
python -m evaluation_pipeline.sentence_zero_shot.run \
  --model_path_or_name <.../babylm_pilot_dense_v0_1M/hf_model/chck_1M> \
  --backend causal \
  --task blimp \
  --data_path evaluation_data/fast_eval/blimp_fast \
  --save_predictions \
  --revision_name chck_1M \
  --batch_size 64 \
  --output_dir <.../eval_results_blimp_fast_1M>
```

Result:

- report: `experiments/archive/initial_model_studies/training/runs/babylm_pilot_dense_v0_1M/eval_results_blimp_fast_1M/chck_1M/chck_1M/zero_shot/causal/blimp/blimp_fast/best_temperature_report.txt`
- predictions: same directory, `predictions.json`
- fast BLiMP average accuracy: **53.63**
- 10k toy checkpoint comparison: 50.13

Interpretation:

- This is still not a SOTA result and should not be overread: the model is tiny (7.4M params), trained on only 1M word exposure, and evaluated only on fast BLiMP.
- It is a real signal that the official-compatible dense causal path learns above chance by 1M words under the exact 2026 tokenizer/eval interface.
- The path is now clean enough for more informative execution: fast evaluations beyond BLiMP (supplement, entity, reading, possibly GlobalPIQA if local data is ready) and then either a 10M V0 curve or the first V1 sparse-routing implementation.

## 4. Immediate next research action

Before implementing V1 sparse routing, collect a slightly broader V0 1M fast-eval profile from the already-trained checkpoint if it is cheap:

- BLiMP Supplement fast
- entity_tracking_fast
- reading fast (if compatible with the checkpoint/path)
- optionally EWoK fast if available locally and no gated-data issue appears

This will show whether the dense baseline's early signal is only syntax/BLiMP or whether it already exposes the human-like/NLP tradeoff targeted by sparse routing. If V0 1M is uniformly weak outside BLiMP, proceed to implement V1 sparse routing. If V0 1M has a specific weakness (e.g. entity/reading or supplement), use that to define the V1 router logging and comparison columns.
