# frequency concentration probe token-frequency concentration probe
Created: 2026-09-06T13:02:47Z

This probe responds to functional_learning's tied-softmax mechanism question. The representative DeBERTa/RoBERTa/GPT models tie input embeddings and output readout rows, so a lexical-row path is possible. Here I test a simple version for the COMPACT_EXPERIENCE ALN-vs-OFF Wikipedia effect: whether per-target source-use changes concentrate on low-frequency token IDs or on tokens whose marginal frequency changed most in the aligned-Qwen 10M pool.

## Pool token counts

| role | rows | words | total tokens | elapsed s | pool |
|---|---:|---:|---:|---:|---|
| OFF | 64381 | 10000000 | 14896688 | 13.8 | `experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/official_only_10M.jsonl` |
| ALN | 64381 | 10000000 | 14804797 | 14.4 | `experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl` |

## Correlations by seed and target class

The effect column is per-target ALN−OFF `gain_T_vs_N`; positive means aligned-Qwen training gave more true-source benefit. Correlations are Pearson correlations over target tokens within the class.

| seed | token_class | n_targets | mean_delta_gain_T_vs_N | corr_with_log1p_OFF_freq | corr_with_delta_log1p_freq | corr_with_ALN_log1p_freq | frac_delta_freq_positive |
|---|---|---|---|---|---|---|---|
| seed43022 | nonoverlap | 2400 | -0.5934 | -0.1524 | -0.0102 | -0.1538 | +0.4442 |
| seed43022 | overlap | 1200 | +3.5975 | -0.0088 | -0.0027 | -0.0091 | +0.4758 |
| seed43122 | nonoverlap | 2400 | -0.8806 | -0.0909 | +0.0136 | -0.0906 | +0.4442 |
| seed43122 | overlap | 1200 | +4.1998 | -0.0059 | +0.0260 | -0.0042 | +0.4758 |

## Two-seed summary

| token_class | n_seed_rows | mean_effect | mean_corr_log1p_OFF_freq | mean_corr_delta_log1p_freq | mean_corr_ALN_log1p_freq |
|---|---|---|---|---|---|
| nonoverlap | 2 | -0.7370 | -0.1216 | +0.0017 | -0.1222 |
| overlap | 2 | +3.8986 | -0.0074 | +0.0116 | -0.0066 |

## Reading

The large source-recurring ALN−OFF effect is not explained by a simple rare-token concentration pattern if it is similar across OFF-frequency quartiles and only weakly correlated with OFF token frequency or marginal ALN−OFF token-frequency change. A strong positive correlation with frequency change would instead make lexical-row exposure a serious alternative. This probe is only lexical-row evidence; it cannot identify attention or residual circuits.

Quartile table: `experiments/archive/relation_learning/data/frequency_concentration_probe/frequency_quartile_effects.csv`. Per-target merged rows: `experiments/archive/relation_learning/data/frequency_concentration_probe/aln_off_wikipedia_target_effects_with_frequency.csv`.
