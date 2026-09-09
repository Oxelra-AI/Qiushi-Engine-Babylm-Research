# cs 4m residualized screen residualized C/S 4M screen construction

This construction repairs the raw C/S arms by selecting within matched source × source-position × tokenizer-compression × quality cells and by using residual scores rather than raw C/S scores.

## Key validation
- Arms written: r_strat_a, r_strat_b, c_unique_high, c_unique_control, s_unique_high, s_unique_control; each has 4,000,000 words: True.
- C-unique high/control disjoint: True; S-unique high/control disjoint: True.
- C-unique-high/S-unique-high overlap: 4447 rows = 0.178.
- Strict-confound max SMD, C high vs control: 0.074 (pronoun_rate); S high vs control: 0.112 (pronoun_rate).

## Mean intended-score movements
- c_unique_high: C=0.505350, S=0.416434, c_unique_resid=0.063124, s_unique_resid=-0.015934.
- c_unique_control: C=0.386801, S=0.418473, c_unique_resid=-0.063800, s_unique_resid=0.016460.
- s_unique_high: C=0.444756, S=0.441105, c_unique_resid=-0.037391, s_unique_resid=0.026616.
- s_unique_control: C=0.448259, S=0.394548, c_unique_resid=0.036597, s_unique_resid=-0.026340.

## Interpretation before H100 training
These arms are much closer to a mechanism-distinguishing screen than the raw C/S materialization because source-internal position, coarse genre/quality, and tokenizer-compression distributions are held by cell quotas, and the high/control contrasts are driven by residual intended scores. They should still be treated as a screening instrument: inspect `screen_summary.json`, the arm metadata, and a few text samples before launching training.

Summary JSON: `experiments/archive/compact_experience/data/cs_4m_residualized_screen/screen_summary.json`
Full-pool features: `experiments/archive/compact_experience/data/cs_4m_residualized_screen/full_pool_features.csv`
