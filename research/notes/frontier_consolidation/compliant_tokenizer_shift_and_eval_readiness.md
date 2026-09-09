# compliant tokenizer shift and eval readiness — compliant-tokenizer geometry and evaluation readiness

## Research role

The central experiment is now the isolated retrain of `compact_view_reinvest` and the matched `clean_qwen` control with a 16k BPE tokenizer trained only on the allowed 10M Strict-Small pool. The invalid old-tokenizer 42.0331347900748 endpoint remains scientific evidence for the density-reinvestment principle but cannot be treated as a submit-ready Strict-Small result.

compliant tokenizer shift and eval readiness did not start new GPU work. The two compliant tokenizer retrain status managed retrains remain unresolved:

- Compliant-tokenizer `reinvest` retraining was pending in `experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2`.
- Compliant-tokenizer `clean_qwen` control retraining was pending in `experiments/archive/frontier_consolidation/training/runs/complianttok_cleanqwen_seed43022_r2`.

Do not infer training results until the runtime delivers their terminal records and the run directories are inspected.

## Tokenizer identity and pool geometry

The old tokenizer path in the diagnostic was verified by SHA256 to match both invalid old-tokenizer endpoints:

- old/template tokenizer: `9cc4f9073675da3f817a6020e7b0833cf3234c0131515d68f4065b00f14933ac`
- invalid `compact_view_reinvest` tokenizer: same SHA256
- invalid clean-Qwen tokenizer: same SHA256
- compliant tokenizer: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`

The compliant tokenizer metadata remains:

- path: `experiments/archive/frontier_consolidation/data/compliant_tokenizer`
- training pool: `cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`
- pool SHA256: `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`
- pool rows/words: 64,740 / 10,000,000
- vocab size: 16,384
- special IDs: `<unk>`=0, `<s>`=1, `</s>`=2, `<pad>`=3, `<mask>`=4

## Training-pool tokenization shift

CPU diagnostic script:

- `experiments/archive/frontier_consolidation/scripts/tokenizer_shift_analysis.py`

Outputs:

- `experiments/archive/frontier_consolidation/data/tokenizer_shift_analysis/tokenizer_shift_analysis.json`
- `research/documents/frontier_consolidation/data/tokenizer_shift_analysis/tokenizer_shift_analysis.md`

Main results:

| pool | rows | words | old tok/word | new tok/word | new/old token ratio | old trunc@256 | new trunc@256 | new-only trunc | visible group Δ mean | total token Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reinvest_10M | 64,740 | 10,000,000 | 1.4756 | 1.4627 | 0.99155 | 15,883 | 15,117 | 232 | +0.2306 | -124,979 |
| clean_qwen_10M | 64,381 | 10,000,000 | 1.4771 | 1.4652 | 0.99215 | 16,630 | 15,872 | 251 | +0.2354 | -116,161 |

Interpretation: the compliant tokenizer is not a crude source of extra fragmentation on the actual training pools. It is slightly more compact in tokens/word on both arms, slightly reduces seq256 truncation, and increases visible word groups by ~0.23 rows on average under the trainer's same WWM grouping rule. The geometry shift is very similar for `reinvest` and `clean_qwen`, which supports a like-for-like treatment comparison after retraining. It does not prove score recovery because vocabulary identity changes 15.7% of token strings and MLM target identities all change.

Relational-marker check: 12/204 tested marker forms changed segmentation; length increases mainly for capitalized `Inside`, `Behind`, `Into`, `Different`, and lowercase `heavier`. Core lowercase spatial/function forms such as `in`, `on`, `under`, `over`, `above`, `below`, `near`, `between`, `inside`, `outside`, `through`, `across`, `around`, `from`, `to`, `into`, `onto`, `within`, etc. mostly keep the same token length. This makes a large spatial/EWoK loss from marker fragmentation alone less likely, but still leaves model-learning effects open.

## Official evaluation text tokenization shift

CPU diagnostic script:

- `experiments/archive/frontier_consolidation/scripts/eval_text_tokenizer_shift.py`

Outputs:

- `experiments/archive/frontier_consolidation/data/eval_text_tokenizer_shift/eval_text_tokenizer_shift.json`
- `research/documents/frontier_consolidation/data/eval_text_tokenizer_shift/eval_text_tokenizer_shift.md`

It read 512,074 official evaluation text units across BLiMP, Supplement, EWoK, Entity, COMPS, generated GlobalPIQA files used by the official-coordinate collation, Reading, and SuperGLUE train/valid text fields. Family aggregates:

| family | texts | seq | new/old token ratio | total token Δ | old trunc % | new trunc % | new-only trunc | old-only trunc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 119,750 | 256 | 1.0042 | +5,122 | 0.000 | 0.000 | 0 | 0 |
| Supplement | 10,436 | 256 | 1.0016 | +269 | 0.000 | 0.000 | 0 | 0 |
| EWoK | 30,472 | 256 | 1.0128 | +3,536 | 0.000 | 0.000 | 0 | 0 |
| EWoK_concat | 15,236 | 256 | 1.0128 | +3,536 | 0.000 | 0.000 | 0 | 0 |
| Entity | 56,898 | 256 | 1.0015 | +14,076 | 0.053 | 0.053 | 0 | 0 |
| COMPS | 182,056 | 256 | 1.0099 | +26,324 | 0.000 | 0.000 | 0 | 0 |
| GlobalPIQA_parallel | 515 | 256 | 1.0061 | +94 | 0.000 | 0.000 | 0 | 0 |
| GlobalPIQA_nonparallel | 300 | 256 | 1.0091 | +67 | 0.000 | 0.000 | 0 | 0 |
| Reading_sentence | 1,726 | 256 | 1.0059 | +118 | 0.000 | 0.000 | 0 | 0 |
| Reading_word | 1,726 | 256 | 1.0035 | +7 | 0.000 | 0.000 | 0 | 0 |
| SuperGLUE | 92,959 | 512 | 0.9954 | -73,034 | 8.937 | 8.840 | 10 | 100 |

Interpretation: on official evaluation text, zero-shot columns become only slightly longer under the compliant tokenizer (roughly +0.2% to +1.3% tokens) and do not create new seq256 truncation. SuperGLUE finetuning text becomes slightly shorter and has slightly fewer 512-token truncations. Thus if the compliant endpoint loses much score, the likely cause is not simple evaluation-text truncation. It would more likely be changed pretraining target geometry/vocabulary, initialization interaction, or genuine reliance of the old route on 100M-trained tokenizer statistics.

## Evaluation harness prepared

New script:

- `experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py`

Preflight outputs:

- `experiments/archive/frontier_consolidation/data/compliant_full_eval/complianttok_reinvest_seed43022_preflight.json`
- `experiments/archive/frontier_consolidation/data/compliant_full_eval/complianttok_clean_qwen_seed43022_preflight.json`

Preflight status: all required evaluation code/data/helper paths exist; `model_path_exists=false` for both arms because the retrains have not yet finished. The script AST-parsed successfully. It uses:

- current pristine strict snapshot: `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict`
- current 7,618-row `ewok_filtered`
- generated GlobalPIQA files from `experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval`, matching the previous official-coordinate collation lineage
- repaired AoA helper `experiments/archive/frontier_consolidation/scripts/aoa_local_ckpts_minctx.py` with `min_context=0` and expected 8,005 rows/checkpoint
- final collation path: `experiments/archive/representation_and_objectives/scripts/stage_pristine_collate.py`

I verified the AoA source files `cdi_childes.json` and `cdi_human.csv` are byte-identical between the INITIAL_MODEL_STUDIES local strict repo and the pristine strict snapshot, so using the existing repaired helper does not create an AoA data-coordinate mismatch.

## Next execution after retrain delivery

After both compliant-tokenizer training arms complete, inspect the results and run directories. For each arm, require:

- `scientific_metrics.json` exists
- `word_exposure` = 100,000,000
- `vocab_size` = 16,384
- tokenizer path/label points to `experiments/archive/frontier_consolidation/data/compliant_tokenizer`
- checkpoint ladder includes `chck_1M`..`chck_9M` and `chck_10M`..`chck_100M`
- `hf_model/chck_100M/model.safetensors` exists

If `reinvest` finishes, the minimum decisive evaluation sequence is:

1. Run zero-shot official columns first, especially BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading, using `evaluate_compliant_endpoint.py --arm reinvest`. These are cheaper than SuperGLUE+AoA and can decide whether full evaluation remains scientifically valuable.
2. If the seven zero-shot/Reading columns preserve a plausible path to >41.8, run SuperGLUE and AoA for the same arm and collate.
3. Run the same evaluation for `clean_qwen` as soon as its retrain exists to compute compliant `reinvest-clean_qwen` treatment effect.

Do not alter batch size, sequence length, corpus, seeds, model architecture, objective, or tokenizer in the compliance recovery arm unless the retrain fails mechanically and the change is only execution scheduling/resource management. If both retrains time out before launching because memory never cleared, relaunch the same frozen commands when memory is available. If the compliant reinvest endpoint is below the public target, use the tokenizer-shift evidence above to localize the loss rather than weakening the compliance rule.
