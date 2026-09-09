# V0 pilot pipeline validation

This note records completed pilot measurements, not a proposed design.

## What ran

1. Wrote `experiments/archive/initial_model_studies/training/scripts/babylm_pilot_train.py` (V0 dense causal GPT-2-style trainer).
2. Ran a 10k-word debug with `babylm_pilot_train.py`, run label `babylm_pilot_dense_debug10k`, and `--max_word_exposure 10000`.
   - returncode 0, duration ~31.6s.
   - run dir: `experiments/archive/initial_model_studies/training/runs/babylm_pilot_dense_debug10k`
3. Downloaded official BabyLM 2026 strict eval data into
   `experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data` (175 files, fast+full).
4. Ran the official fast BLiMP zero-shot eval against the pilot `chck_1M` checkpoint via
   `python -m evaluation_pipeline.sentence_zero_shot.run --backend causal --task blimp --data_path evaluation_data/fast_eval/blimp_fast`.

## Verified facts (evidence)

- **Official corpus word count is exactly 10,000,000 whitespace words** across the six files
  (bnc_spoken 762,073; childes 2,841,101; gutenberg 2,557,721; open_subtitles 2,282,877;
  simple_wiki 1,531,437; switchboard 24,791). File sha256 hashes recorded in
  `runs/babylm_pilot_dense_debug10k/data_manifest.json`. This confirms the ≤10M-word budget anchor
  precisely and gives per-source proportions for later data-mixing decisions.
- **The pilot trainer completes its data/model/output pipeline**: reads official data over network, trains on CUDA
  (2 GPUs visible), saves HF `hf_model/` and `hf_model/chck_1M/` (config.json, model.safetensors,
  tokenizer.json), writes `training_log.jsonl`, `metrics.json`/`scientific_metrics.json`,
  `data_manifest.json`, `tokenizer_manifest.json`. Debug loss fell 9.72 → 9.30 over 8 steps.
- **Model → official eval interface is compatible**: `AutoModelForCausalLM.from_pretrained(chck_1M, revision=chck_1M)`
  loads and the BabyLM fast BLiMP script produced a real score of **50.13** (chance-level, as expected for a
  2.51M-param model trained on only 10k words). The pipeline, not the score, is the result here.

## Real issue found and fixed

- The official baseline tokenizer saves `tokenizer_class = "TokenizersBackend"`, which
  `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` **cannot re-import**
  (`ValueError: Tokenizer class TokenizersBackend does not exist`). The eval script only survived
  because it silently fell back to `PreTrainedTokenizerFast`.
- Fix applied in `babylm_pilot_train.py`: re-wrap the loaded tokenizer as a portable
  `PreTrainedTokenizerFast` (carrying `<s>`, `</s>`, `<unk>`, `<pad>`, `<mask>`) before saving, so all
  checkpoints reload cleanly and remain HF-submittable. Must re-verify on the next run.

## Not yet done / next

- Re-run V0 at the real pilot scale (1M word exposure, larger model near baseline shape) with the
  tokenizer fix and confirm clean `AutoTokenizer` reload.
- Confirm `chck_1M` local-subdir loading path used by the eval loop scripts (they pass `revision=chck_1M`
  against a repo path; local-dir revision semantics need an explicit check or a checkpoint-dir layout that
  the loop can consume).
- Only after V0 at 1M is clean should V1 sparse routing be implemented.

## Remaining limitations

- The shared HF cache was read-only; downloads succeeded with cache warnings. This did not block the pilot.
- EWoK full eval still requires accepting dataset terms; fast BLiMP/entity/reading are available now.
