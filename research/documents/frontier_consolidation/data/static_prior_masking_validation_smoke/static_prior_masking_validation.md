# earlier analysis static-prior masking validation

CPU validation that generated static-prior trainer preserves WWM mask budget and reallocates mask pressure on real legal-pool examples before any GPU training is considered.

This is a CPU-only validation of the generated trainer's actual masking function; it is not a model-training result.

## Samples and mode comparisons

### front_changed_plus_early_filler
- rows: 512; words: 71691; source_counts: {'cleanqwen_fineweb_compact_view_reinvest': 512}

| mode | split | group mask | token mask | selected mean prior | prior lift | rel lex lift | high1.50 lift | Δ group vs fixed | Δ token vs fixed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wwm_fixed:relation_only_v1 | all | 0.1507 | 0.1511 | 1.0130 | 1.0003 | 1.0019 | 1.0100 | 0.0000 | 0.0000 |
| wwm_fixed:relation_only_v1 | changed | 0.1507 | 0.1511 | 1.0130 | 1.0003 | 1.0019 | 1.0100 | 0.0000 | 0.0000 |
| wwm_fixed:relation_only_v1 | other | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| wwm_fixed:relation_info_v1 | all | 0.1507 | 0.1511 | 1.0203 | 1.0004 | 1.0019 | 1.0058 | 0.0000 | 0.0000 |
| wwm_fixed:relation_info_v1 | changed | 0.1507 | 0.1511 | 1.0203 | 1.0004 | 1.0019 | 1.0058 | 0.0000 | 0.0000 |
| wwm_fixed:relation_info_v1 | other | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| wwm_static_prior:relation_only_v1 | all | 0.1503 | 0.1484 | 1.0562 | 1.0430 | 1.5755 | 1.7388 | -0.0004 | -0.0026 |
| wwm_static_prior:relation_only_v1 | changed | 0.1503 | 0.1484 | 1.0562 | 1.0430 | 1.5755 | 1.7388 | -0.0004 | -0.0026 |
| wwm_static_prior:relation_only_v1 | other | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| wwm_static_prior:relation_info_v1 | all | 0.1503 | 0.1504 | 1.0436 | 1.0233 | 1.3988 | 1.5289 | -0.0004 | -0.0007 |
| wwm_static_prior:relation_info_v1 | changed | 0.1503 | 0.1504 | 1.0436 | 1.0233 | 1.3988 | 1.5289 | -0.0004 | -0.0007 |
| wwm_static_prior:relation_info_v1 | other | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### stride_full_pool
- rows: 256; words: 39420; source_counts: {'cleanqwen_fineweb_compact_view_reinvest': 12, 'childes': 69, 'bnc_spoken': 11, 'gutenberg': 43, 'qwen_pair_packed': 50, 'open_subtitles': 50, 'simple_wiki': 21}

| mode | split | group mask | token mask | selected mean prior | prior lift | rel lex lift | high1.50 lift | Δ group vs fixed | Δ token vs fixed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wwm_fixed:relation_only_v1 | all | 0.1501 | 0.1514 | 0.9969 | 0.9981 | 0.9535 | 0.9642 | 0.0000 | 0.0000 |
| wwm_fixed:relation_only_v1 | changed | 0.1507 | 0.1543 | 1.0097 | 1.0055 | 1.0050 | 1.1324 | 0.0000 | 0.0000 |
| wwm_fixed:relation_only_v1 | other | 0.1501 | 0.1513 | 0.9963 | 0.9977 | 0.9507 | 0.9560 | 0.0000 | 0.0000 |
| wwm_fixed:relation_info_v1 | all | 0.1501 | 0.1514 | 0.9958 | 0.9984 | 0.9535 | 0.9596 | 0.0000 | 0.0000 |
| wwm_fixed:relation_info_v1 | changed | 0.1507 | 0.1543 | 1.0157 | 1.0052 | 1.0050 | 1.1284 | 0.0000 | 0.0000 |
| wwm_fixed:relation_info_v1 | other | 0.1501 | 0.1513 | 0.9949 | 0.9981 | 0.9507 | 0.9514 | 0.0000 | 0.0000 |
| wwm_static_prior:relation_only_v1 | all | 0.1504 | 0.1497 | 1.0341 | 1.0353 | 1.5592 | 1.7417 | 0.0003 | -0.0017 |
| wwm_static_prior:relation_only_v1 | changed | 0.1528 | 0.1547 | 1.0534 | 1.0491 | 1.6509 | 2.0200 | 0.0021 | 0.0004 |
| wwm_static_prior:relation_only_v1 | other | 0.1503 | 0.1495 | 1.0332 | 1.0347 | 1.5540 | 1.7281 | 0.0002 | -0.0018 |
| wwm_static_prior:relation_info_v1 | all | 0.1503 | 0.1516 | 1.0175 | 1.0202 | 1.3916 | 1.5336 | 0.0001 | 0.0001 |
| wwm_static_prior:relation_info_v1 | changed | 0.1542 | 0.1570 | 1.0417 | 1.0309 | 1.4884 | 1.7573 | 0.0035 | 0.0027 |
| wwm_static_prior:relation_info_v1 | other | 0.1501 | 0.1513 | 1.0164 | 1.0197 | 1.3863 | 1.5229 | -0.0000 | 0.0000 |

## Input files

- trainer: `experiments/archive/frontier_consolidation/scripts/masking_curriculum_trainer_static_prior.py`
- pool_10m: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`
- tokenizer: `experiments/archive/frontier_consolidation/data/compliant_tokenizer`
- prior_json: `experiments/archive/frontier_consolidation/data/static_token_mask_prior/static_token_mask_prior.json`
- seq_length: `256`
- batch_size: `256`
- num_seeds: `2`
- seed0: `43023`
- rows_front: `512`
- rows_stride: `256`

Full JSON: `experiments/archive/frontier_consolidation/data/static_prior_masking_validation_smoke/static_prior_masking_validation.json`
