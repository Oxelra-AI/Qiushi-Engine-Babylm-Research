# extractive view pool preflight and training plan — Extractive view pool preflight and training plan

## What was built

Two extractive-view 10M pool variants, each replacing compact generated views with source-only text selections while preserving exact row word counts and filler:

1. **extractive_wide**: content-maximizing, source-wide (keeps content words, drops function words)
2. **extractive_balanced**: density-matched to compact (~63-65% content fraction), source-wide

Both pools: 64,740 rows, exactly 10,000,000 words, zero word-count mismatches vs compact pool, all 61,734 filler rows byte-identical. Uses only original source words in source order. No model training, evaluation, upload, or leaderboard action.

Pool SHAs:
- `extractive_wide_10M.jsonl`: `8554f61f810353ab...`
- `extractive_balanced_10M.jsonl`: `17bf6ba1478fc2a3...`

Script: `scripts/extractive_view_pool_preflight.py`
Data: `data/extractive_view_pool_preflight/`

## Quantitative comparison

### Content density (mean content_fraction)
| Arm | Density | Delta vs compact |
|-----|---------|-----------------|
| compact | 0.6474 | — |
| repeat | 0.4884 | -0.1590 |
| extractive_wide | 0.8107 | +0.1633 |
| extractive_balanced | 0.6321 | **-0.0153** |

### Source content coverage (mean)
| Arm | Coverage | Delta vs compact |
|-----|----------|-----------------|
| compact | 0.6570 | — |
| repeat | 0.5981 | -0.0589 |
| extractive_wide | 0.9821 | +0.3251 |
| extractive_balanced | 0.7923 | +0.1353 |

### Tail content coverage (mean)
| Arm | Tail cov | Delta vs compact |
|-----|----------|-----------------|
| compact | 0.7024 | — |
| repeat | 0.0474 | -0.6550 |
| extractive_wide | 0.9999 | +0.2975 |
| extractive_balanced | 0.8136 | +0.1112 |

### Source-absent content (total words across 12,155 pairs)
| Arm | Absent |
|-----|--------|
| compact | 17,891 |
| repeat | 0 |
| extractive_wide | 0 |
| extractive_balanced | 0 |

### Changed block active tokens (legal spatial repair route status tokenizer)
| Arm | Active tokens | Delta vs compact | % delta |
|-----|--------------|-----------------|---------|
| compact | 607,909 | — | — |
| repeat | 583,511 | -24,398 | -4.01% |
| extractive_wide | 614,500 | +6,591 | +1.08% |
| extractive_balanced | 598,081 | **-9,828** | -1.62% |

### Source position decile coverage
| D | compact | repeat | ext_wide | ext_balanced |
|---|---------|--------|----------|-------------|
| 0 | 0.6828 | 1.0000 | 0.8777 | 0.8308 |
| 1 | 0.6263 | 1.0000 | 0.9518 | 0.7266 |
| 2 | 0.6206 | 1.0000 | 0.9870 | 0.7750 |
| 3 | 0.6318 | 0.9969 | 0.9972 | 0.7740 |
| 4 | 0.6352 | 0.9484 | 0.9998 | 0.7565 |
| 5 | 0.6408 | 0.7100 | 1.0000 | 0.7665 |
| 6 | 0.6390 | 0.3525 | 1.0000 | 0.7923 |
| 7 | 0.6596 | 0.1215 | 1.0000 | 0.7657 |
| 8 | 0.7088 | 0.0567 | 1.0000 | 0.7153 |
| 9 | 0.7632 | 0.0585 | 1.0000 | 0.8748 |

Both extractive variants cover all deciles at 72-100%, while compact varies 62-76% and repeat drops to near zero above decile 5.

## Structural observation

An extractive view cannot simultaneously match compact on all dimensions because compact generates 17,891 source-absent content words (~17.1% of compact content). These novel words create compact's specific density-coverage profile, its fluent paraphrased surface forms, and its grammatical coherence. The extractive view, using only source words, necessarily has zero source-absent content and produces telegraphic/chunky text rather than fluent sentences.

The **extractive_balanced** variant is the closest achievable match to compact:
- Content density: 0.6321 vs 0.6474 (difference -0.0153)
- Source coverage: 0.7923 vs 0.6570 (extractive actually covers MORE because it doesn't "spend" words on novel content)
- Token count: -9,828 vs compact (-1.62% in changed block)
- Decile coverage: more uniform than compact across all deciles
- Surface form: telegraphic source fragments vs fluent generated text

These residual imbalances are NOT confounds to fix — they are the treatment differences that make the experiment informative.

## Recommended primary training arm: extractive_balanced

The **extractive_balanced** variant is the strongest single arm because:
1. Closest to compact in content density (the best-controlled property)
2. Has broad source-position coverage (matches compact's tail access)
3. Uses zero source-absent words (isolates generation effect)
4. Has only 1.62% fewer changed-block tokens than compact

The comparison `extractive_balanced vs compact vs repeat` directly answers the core question.

## Training plan (not yet launched)

### Recipe (matching compliant tokenizer retrain status stock DeBERTa reference)
- Architecture: stock `DebertaV2ForMaskedLM` 8×480, 34,467,424 params
- Tokenizer: legal spatial repair route status (SHA `91b775...`, vocab 16,384)
- Seeds: 43 / 43022 (extra_init) / 43023 (train_rng)
- Optimizer: AdamW lr 0.001, wd 0.01, betas (0.9, 0.98), warmup 0.06
- Masking: fixed WWM p=0.15, 80/10/10 corruption
- Batch 256, seq 256, max_word_exposure 100,000,000
- lr_total_steps 2,529 (100M LR horizon)
- Checkpoints: every 10M words (chck_10M through chck_100M)

### Preparation required before training
1. Materialize 100M training stream from extractive_balanced_10M.jsonl (10 passes)
2. Verify the 100M stream has exactly 100M words and row structure compatible with `masking_curriculum_trainer.py`

### Comparison arms (existing or awaiting)
- **compact_view_reinvest**: compliant tokenizer retrain status stock DeBERTa 100M → Overall 41.258 at 100M
- **repeat_compact_reinvest**: compliant tokenizer retrain status variant, matched 100M
- **clean_qwen** (control): clean control trained and representation frontier comparison at 70M/80M

### Evaluation plan
- Official-compatible cheap selected scoring at all 10M-interval checkpoints
- Primary readout: late-band 60M–100M means
- Primary metrics: cheap6 without GlobalPIQA, cheap5 without GlobalPIQA/Reading, EWoK+Entity, Supplement, Entity, COMPS
- GlobalPIQA/Reading-only movement is insufficient to promote

## Predeclared outcome interpretation

| Outcome | Meaning |
|---------|---------|
| extractive_balanced ≈ compact > repeat | Source-wide content selection sufficient; generation not necessary for the principle |
| compact > extractive_balanced > repeat | Coverage contributes, but generated re-expression adds downstream value |
| compact > extractive_balanced ≈ repeat | Source-wide access alone insufficient; compact re-realization/context/abstraction is central |
| extractive_balanced > compact | Higher source coverage outweighs generated quality (unlikely) |
| Local-only signal without stable families | Another local/downstream split; do not promote |

## Training gate

Training remains conditional on materialization and verification of the 100M stream and identification of the exact comparison checkpoints from the existing compact/repeat runs.
