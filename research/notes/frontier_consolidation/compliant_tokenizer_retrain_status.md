# compliant tokenizer retrain status — compliant-tokenizer retrain status

## Why this is now load-bearing

The previous `compact_view_reinvest` seed43022 endpoint at official-coordinate Overall 42.0331347900748 is no longer a submit-ready Strict-Small result, because it inherited the `BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict` 16k tokenizer whose training data provenance is the 100M Strict corpus. Since tokenizer training data counts against the Strict-Small language-data budget, all results on that tokenizer remain useful scientific evidence but not valid final submission evidence.

## Tokenizer trained

Script: `experiments/archive/frontier_consolidation/scripts/train_compliant_tokenizer.py`

Output: `experiments/archive/frontier_consolidation/data/compliant_tokenizer`

Training source: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`

Verified properties:

- pool rows: 64,740
- pool words: 10,000,000
- pool SHA256: `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`
- tokenizer type: byte-level BPE using the old tokenizer's structure
- vocab size: 16,384
- special IDs: `<unk>`=0, `<s>`=1, `</s>`=2, `<pad>`=3, `<mask>`=4
- tokenizer JSON SHA256: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`

## Matched retrain launcher

Script: `experiments/archive/frontier_consolidation/scripts/train_compliant_model.py`

It changes only the tokenizer path to the new compliant tokenizer and otherwise freezes:

- data file and metadata
- DeBERTa-v2 8×480, 8 heads, ffn_mult=4
- seed 43, extra_init_seed 43022, train_rng_seed 43023
- batch 256, seq_length 256, max_seq_length 256
- WWM fixed 0.15
- AdamW lr 0.001, warmup_fraction 0.06, weight_decay 0.01
- checkpoint_words 1M, max_word_exposure 100M

Two arms are required:

1. `reinvest`: `cleanqwen_fineweb_compact_view_reinvest_100M.jsonl`, SHA256 `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`, 647,400 rows / 100,000,000 words.
2. `clean_qwen`: `qwen_aligned_100M.jsonl`, SHA256 `728192f8e8c5855aaa52a6b6940ff3a4fd6aa8f87c98aca4ff018dbc04cb0345`, 643,810 rows / 100,000,000 words.

The clean-Qwen control uses the same compliant tokenizer so that `reinvest-clean` remains like-for-like.

## Current execution state

Initial full retrain launches:

- Reinvest: failed before producing metrics or checkpoints.
- Clean-Qwen: failed before producing metrics or checkpoints.

Both failures were CUDA OOM during DeBERTa attention before producing `scientific_metrics.json`. The logs show GPU 0 had only about 59 MiB free at allocation time. This is an environment/resource collision, not model evidence. Run dirs contain only launcher files, `example_order_manifest.json`, and empty `training_log.jsonl`; no checkpoint was produced.

I updated `train_compliant_model.py` to set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. This only affects allocator behavior and does not change the scientific recipe. The next run should use fresh run directories and wait until enough H100 memory is available before starting.

## Next needed work

Relaunch both full 100M retrains with fresh run dirs, after confirming enough free GPU memory. If only one H100 is free, run `reinvest` first because it determines submission recovery; run `clean_qwen` as soon as a second H100 is available for like-for-like treatment effect. After checkpoints exist, evaluate both endpoints under official-compatible coordinates and compare:

- absolute compliant Overall vs visible 41.8 leader
- compliant `reinvest-clean_qwen` treatment effect vs old tokenizer treatment effect
- which columns moved under tokenizer replacement

Do not weaken the compliance rule or infer scientific failure from the OOM attempt.
