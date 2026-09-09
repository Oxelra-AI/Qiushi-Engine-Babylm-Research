# earlier analysis static-prior masking validation

CPU validation that generated static-prior trainer preserves WWM mask budget and reallocates mask pressure on real legal-pool examples before any GPU training is considered.

This is a CPU-only validation of the generated trainer's actual masking function; it is not a model-training result.

## Samples and mode comparisons

### front_changed_plus_early_filler
- rows: 1024; words: 143191; source_counts: {'cleanqwen_fineweb_compact_view_reinvest': 1024}

| mode | split | group mask | token mask | selected mean prior | prior lift | rel lex lift | high1.50 lift | Δ group vs fixed | Δ token vs fixed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wwm_fixed:relation_only_v1 | all | 0.1501 | 0.1510 | 1.0129 | 0.9994 | 0.9862 | 0.9889 | 0.0000 | 0.0000 |
| wwm_fixed:relation_only_v1 | changed | 0.1501 | 0.1510 | 1.0129 | 0.9994 | 0.9862 | 0.9889 | 0.0000 | 0.0000 |
| wwm_fixed:relation_only_v1 | other | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| wwm_fixed:relation_info_v1 | all | 0.1501 | 0.1510 | 1.0205 | 0.9999 | 0.9862 | 0.9851 | 0.0000 | 0.0000 |
| wwm_fixed:relation_info_v1 | changed | 0.1501 | 0.1510 | 1.0205 | 0.9999 | 0.9862 | 0.9851 | 0.0000 | 0.0000 |
| wwm_fixed:relation_info_v1 | other | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| wwm_static_prior:relation_only_v1 | all | 0.1524 | 0.1502 | 1.0571 | 1.0430 | 1.5662 | 1.7330 | 0.0023 | -0.0008 |
| wwm_static_prior:relation_only_v1 | changed | 0.1524 | 0.1502 | 1.0571 | 1.0430 | 1.5662 | 1.7330 | 0.0023 | -0.0008 |
| wwm_static_prior:relation_only_v1 | other | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| wwm_static_prior:relation_info_v1 | all | 0.1502 | 0.1505 | 1.0446 | 1.0235 | 1.3842 | 1.5186 | 0.0000 | -0.0004 |
| wwm_static_prior:relation_info_v1 | changed | 0.1502 | 0.1505 | 1.0446 | 1.0235 | 1.3842 | 1.5186 | 0.0000 | -0.0004 |
| wwm_static_prior:relation_info_v1 | other | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### stride_full_pool
- rows: 512; words: 79179; source_counts: {'cleanqwen_fineweb_compact_view_reinvest': 24, 'qwen_pair_packed': 93, 'gutenberg': 101, 'childes': 114, 'simple_wiki': 64, 'open_subtitles': 84, 'bnc_spoken': 31, 'switchboard': 1}

| mode | split | group mask | token mask | selected mean prior | prior lift | rel lex lift | high1.50 lift | Δ group vs fixed | Δ token vs fixed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wwm_fixed:relation_only_v1 | all | 0.1504 | 0.1507 | 1.0005 | 0.9994 | 0.9991 | 0.9788 | 0.0000 | 0.0000 |
| wwm_fixed:relation_only_v1 | changed | 0.1480 | 0.1475 | 1.0134 | 0.9966 | 0.9912 | 0.9473 | 0.0000 | 0.0000 |
| wwm_fixed:relation_only_v1 | other | 0.1505 | 0.1508 | 1.0000 | 0.9995 | 1.0002 | 0.9813 | 0.0000 | 0.0000 |
| wwm_fixed:relation_info_v1 | all | 0.1504 | 0.1507 | 1.0002 | 0.9994 | 0.9991 | 0.9774 | 0.0000 | 0.0000 |
| wwm_fixed:relation_info_v1 | changed | 0.1480 | 0.1475 | 1.0192 | 0.9973 | 0.9912 | 0.9226 | 0.0000 | 0.0000 |
| wwm_fixed:relation_info_v1 | other | 0.1505 | 0.1508 | 0.9994 | 0.9995 | 1.0002 | 0.9813 | 0.0000 | 0.0000 |
| wwm_static_prior:relation_only_v1 | all | 0.1521 | 0.1504 | 1.0397 | 1.0385 | 1.6228 | 1.7817 | 0.0017 | -0.0003 |
| wwm_static_prior:relation_only_v1 | changed | 0.1513 | 0.1482 | 1.0658 | 1.0482 | 1.6518 | 1.7857 | 0.0033 | 0.0007 |
| wwm_static_prior:relation_only_v1 | other | 0.1521 | 0.1505 | 1.0386 | 1.0381 | 1.6217 | 1.7820 | 0.0016 | -0.0004 |
| wwm_static_prior:relation_info_v1 | all | 0.1503 | 0.1505 | 1.0219 | 1.0211 | 1.4154 | 1.5346 | -0.0001 | -0.0002 |
| wwm_static_prior:relation_info_v1 | changed | 0.1489 | 0.1481 | 1.0455 | 1.0231 | 1.3737 | 1.4478 | 0.0010 | 0.0006 |
| wwm_static_prior:relation_info_v1 | other | 0.1503 | 0.1506 | 1.0209 | 1.0210 | 1.4180 | 1.5399 | -0.0002 | -0.0002 |

## Input files

- trainer: `experiments/archive/frontier_consolidation/scripts/masking_curriculum_trainer_static_prior.py`
- pool_10m: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`
- tokenizer: `experiments/archive/frontier_consolidation/data/compliant_tokenizer`
- prior_json: `experiments/archive/frontier_consolidation/data/static_token_mask_prior/static_token_mask_prior.json`
- seq_length: `256`
- batch_size: `256`
- num_seeds: `3`
- seed0: `43023`
- rows_front: `1024`
- rows_stride: `512`

Full JSON: `experiments/archive/frontier_consolidation/data/static_prior_masking_validation_repaired/static_prior_masking_validation.json`
