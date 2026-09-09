# bytealphabet tokenizer repair and retrain priority — legal byte-alphabet tokenizer repair and retrain priority

## Reason for the Route Change

The compliant-tokenizer reinvest retrain and clean-Qwen control remained unresolved. This note uses CPU-only tokenization audits, not their active training outputs, to test whether the newly legal 10M-trained tokenizer is a robust byte-level tokenizer for the official scorer.

The spatial repair route status tokenizer was legally trained on the correct `compact_view_reinvest` 10M pool, but the script did **not** pass `initial_alphabet=ByteLevel.alphabet()` to `BpeTrainer`. Its training pool contains no literal newline characters inside `text` fields, while official Supplement dialogue/QA strings contain newlines. This created an avoidable tokenizer-coverage vulnerability.

## Evidence

### Broad byte/UNK audit

Script: `experiments/archive/frontier_consolidation/scripts/tokenizer_byte_unk_audit.py`

Outputs:
- `experiments/archive/frontier_consolidation/data/tokenizer_byte_unk_audit/tokenizer_byte_unk_audit.json`
- `research/documents/frontier_consolidation/data/tokenizer_byte_unk_audit/tokenizer_byte_unk_audit.md`

Key results:
- spatial repair route status tokenizer SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`
- 10M pool SHA: `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`
- spatial repair route status tokenizer missing ByteLevel alphabet entries: 71.
- It produced zero `<unk>` on its own allowed 10M pretraining pool.
- It produced 1,799 `<unk>` events in the broad official-eval scan.

### Scored-text localization

Script: `experiments/archive/frontier_consolidation/scripts/scored_text_unk_localization.py`

Outputs:
- `experiments/archive/frontier_consolidation/data/tokenizer_scored_text_unk_localization/scored_text_unk_localization.json`
- `research/documents/frontier_consolidation/data/tokenizer_scored_text_unk_localization/scored_text_unk_localization.md`

With padding/truncation disabled and likely scored fields only, spatial repair route status tokenizer produced:
- total likely-scored strings: 441,864
- total spatial repair route status-tokenizer `<unk>` events: 1,513
- by family: BLiMP 0, EWoK 0, Reading/AoA 44, SuperGLUE 523, Supplement 946.
- Supplement hotspots are exactly scored dialogue/QA string newlines:
  - `supplement_filtered/turn_taking.jsonl`: 244 `sentence_good` + 244 `sentence_bad` unknowns.
  - `supplement_filtered/qa_congruence_tricky.jsonl`: 165 + 165 unknowns.
  - `supplement_filtered/qa_congruence_easy.jsonl`: 64 + 64 unknowns.
- Example: `Who cleaned?\nDavid cleaned.` encodes the newline as `<unk>` under the spatial repair route status tokenizer.

Official zero-shot code uses model tokenizer strings with `add_special_tokens=False`; the scorer used here does not normalize those newlines away before scoring. Thus this is a real scored-input tokenizer construction issue, not an unscored metadata artifact.

### Legal same-pool byte-alphabet repair tokenizer

Script: `experiments/archive/frontier_consolidation/scripts/train_alphabet_compliant_tokenizer.py`

Output tokenizer:
- `experiments/archive/frontier_consolidation/data/compliant_tokenizer_bytealphabet`
- SHA256 `b2b317e655a96f2c14eb559ce9fa7573f180904d84819cacfb5f61174cf355cf`

Construction:
- same allowed 10M reinvest pool as spatial repair route status; no evaluation text used to train vocabulary
- same 16,384 vocabulary size
- same special IDs `<unk>`=0, `<s>`=1, `</s>`=2, `<pad>`=3, `<mask>`=4
- same template normalizer / pre-tokenizer / post-processor / decoder
- only repair: `BpeTrainer(initial_alphabet=ByteLevel.alphabet())`

Verification:
- zero missing ByteLevel alphabet entries
- sample with newline, degree symbol, dash, and Persian text encodes with zero `<unk>`

### spatial repair route status vs bytealphabet tokenizer repair and retrain priority tokenizer comparison

Script: `experiments/archive/frontier_consolidation/scripts/compare_compliant_tokenizers.py`

Outputs:
- `experiments/archive/frontier_consolidation/data/compare_compliant_tokenizers/compare_compliant_tokenizers.json`
- `research/documents/frontier_consolidation/data/compare_compliant_tokenizers/compare_compliant_tokenizers.md`

Key results:
- Shared token strings: 16,313 / 16,384 = 0.995667.
- Training pool token count ratio bytealphabet tokenizer repair and retrain priority/spatial repair route status: 1.00032155 (+0.032%).
- Likely scored evaluation text token ratio bytealphabet tokenizer repair and retrain priority/spatial repair route status: 1.00072884 (+0.073%).
- spatial repair route status eval `<unk>`: 1,513; bytealphabet tokenizer repair and retrain priority eval `<unk>`: 0.
- bytealphabet tokenizer repair and retrain priority has zero `<unk>` on both the allowed 10M pool and likely scored evaluation strings.

Interpretation: this is a surgical legal tokenizer-construction repair, not a new data route, benchmark-conditioned vocabulary choice, or model tweak. It preserves the legal 10M pool and almost all BPE merges while removing an avoidable scored newline/byte coverage defect.

## Consequence for running managed work

Because the spatial repair route status-tokenizer retrain is now known to carry scored-input `<unk>` defects in Supplement/Reading/SuperGLUE, it is no longer the clean endpoint for the submission question. It may still be useful if it delivers, but only as a flawed-tokenizer contrast.

The clean-Qwen fixed-tokenizer control was cancelled. Its tokenizer was found to be flawed, so the run was no longer an appropriate fixed-tokenizer comparison and was not a submission endpoint.

The original compliant-tokenizer reinvest training remained unresolved. Any completed result would be informative as a flawed-tokenizer contrast, not as the main endpoint.

The repaired submission-relevant reinvest retrain was launched:
- label: `bytealphabet compliant-tokenizer reinvest retrain`
- run dir: `experiments/archive/frontier_consolidation/training/runs/bytealphatok_reinvest_seed43022`
- tokenizer: `experiments/archive/frontier_consolidation/data/compliant_tokenizer_bytealphabet`, SHA `b2b317e655a96f2c14eb559ce9fa7573f180904d84819cacfb5f61174cf355cf`
- same frozen corpus: `cleanqwen_fineweb_compact_view_reinvest_100M.jsonl`, SHA `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`
- same frozen recipe: DeBERTa-v2 8x480, seq256, batch256, WWM 0.15, AdamW lr 0.001, seed 43, init seed 43022, train RNG 43023, 100M words, 1M checkpoint cadence.

The expensive-work decision is justified by the cheap audits above: the repaired run decides whether the compact-view reinvestment mechanism clears official Strict-Small evaluation under a legal tokenizer that covers the official scoring surface. No data, model, seed, objective, or optimization factor is changed.

## Updated evaluation identity

I wrote a separate launcher and wait wrapper:
- `experiments/archive/frontier_consolidation/scripts/train_bytealphabet_compliant_model.py`
- `experiments/archive/frontier_consolidation/scripts/wait_and_train_bytealphabet_compliant.py`

Both AST-check and preflight passed for reinvest; preflight also passed for clean-Qwen if a later fixed-tokenizer control is needed.

I patched the inspector and post-delivery driver so byte-alphabet endpoints can be inspected/evaluated under a distinct target name and expected tokenizer SHA:
- inspector: `experiments/archive/frontier_consolidation/scripts/inspect_compliant_retrain.py`
- driver: `experiments/archive/frontier_consolidation/scripts/compliant_postdelivery_driver.py`

Dry-run command for the repaired endpoint passed:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/compliant_postdelivery_driver.py \
  --arm reinvest --gpu 1 \
  --target bytealphatok_reinvest_seed43022 \
  --run-dir experiments/archive/frontier_consolidation/training/runs/bytealphatok_reinvest_seed43022 \
  --expected-tokenizer experiments/archive/frontier_consolidation/data/compliant_tokenizer_bytealphabet \
  --expected-tokenizer-sha256 b2b317e655a96f2c14eb559ce9fa7573f180904d84819cacfb5f61174cf355cf \
  --dry-run
```

Dry-run output:
- postdelivery JSON: `experiments/archive/frontier_consolidation/data/compliant_postdelivery_driver/bytealphatok_reinvest_seed43022_postdelivery_driver.json`
- collate summary path: `experiments/archive/frontier_consolidation/data/compliant_pristine_collate/bytealphatok_reinvest_seed43022/pristine_collate_bytealphatok_reinvest_seed43022_summary.json`

## Next action when runtime delivers tasks

Do not poll. When a task is delivered:

1. If byte-alphabet reinvest training completes successfully, inspect `experiments/archive/frontier_consolidation/training/runs/bytealphatok_reinvest_seed43022` using the patched inspector with expected SHA `b2b317...55cf`. Require complete 100M exposure, `hf_model/chck_100M`, 19-checkpoint AoA ladder, vocab 16,384, and byte-alphabet tokenizer SHA.
2. Evaluate with the patched driver under target `bytealphatok_reinvest_seed43022`. Use cheap zero-shot+Reading first, then finish SuperGLUE, official min_context=0 AoA with 8,005 rows/checkpoint, and pristine collation unless hard upper-bound arithmetic makes 41.8 unreachable.
3. If the original compliant-tokenizer reinvest training completes, inspect/evaluate only if useful as a flawed-tokenizer contrast, not as the main endpoint. Its tokenizer SHA is `91b775...e8f9` and carries known scored-text `<unk>` defects.
4. Do not relaunch the spatial repair route status-tokenizer clean-Qwen control. If a fixed-tokenizer clean control becomes scientifically necessary, use the bytealphabet tokenizer repair and retrain priority byte-alphabet tokenizer instead, but only after the submission-relevant reinvest endpoint is evaluated or if GPU availability makes it noncompetitive with the main endpoint.
