# Positional Score Terms and the Compact Effect

Status: coordinate checks and short smokes completed; three missing factorial cells were initiated and their final effects remained unmeasured.

GPT2/RoBERTa transfer comparisons changed several factors together. A cleaner test compared compact-minus-repeat in full DeBERTa with compact-minus-repeat after removing c2p/p2c positional score terms. A no-disentangle pair alone was insufficient because the matched full-DeBERTa repetition cell was missing.

The three required new cells were full-model repetition, no-disentangle compact and no-disentangle repetition. The existing full compact cell could be reused only with exact-coordinate checks. No-disentangle retained relative attention and absolute input positions but used an empty positional-attention-type list and had no positional key/query projections.

Both streams had 647,400 rows and 100,000,000 words, 30,050 changed-row occurrences and 3,005 changed IDs. The common recipe was 8 layers, hidden size 480, 8 heads, FFN multiplier 4, seeds 43/43022/43023, WWM 0.15, AdamW learning rate 0.001, weight decay 0.01, warmup 0.06, batch/sequence length 256 and 2,529-update learning-rate horizon. Full/no-disentangle parameter counts were 34,467,424/30,773,344.

A requested 1,000,000-word prefix would cut a row; the corrected whole-row smoke used 999,918 words and passed on full-repeat and no-disentangle compact. Its checkpoint label did not make this a 1M treatment-effect measurement.

Survival of the compact contrast on stable families would show that c2p/p2c scores were not necessary in this coordinate. Collapse would initially implicate the removed package together with parameter-count and score-normalization changes, not identify c2p versus p2c individually. Predeclared readouts included 80M and 100M, paired-item intervals, cheap6 excluding GlobalPIQA, cheap5 excluding GlobalPIQA/Reading and family components. Loss or GlobalPIQA-only movement was insufficient.
