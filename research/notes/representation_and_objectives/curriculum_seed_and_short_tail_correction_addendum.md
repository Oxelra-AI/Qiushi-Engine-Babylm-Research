# Curriculum Seed and Short-Tail Correction

Status: first attempt unusable; repaired fixed-256 parity passed; curriculum endpoint still unmeasured in this record.

The matched baseline used data seed 43, initialization seed 43022 and training RNG seed 43023. The first curriculum attempt omitted the last setting and would instead use 43 for masking/dropout. It failed before useful output, and any putative endpoint would have confounded curriculum with training randomness.

The repair explicitly retained training seed 43023 and short chunks down to one token. At sequence length 64, requiring at least eight tokens would have dropped 1,058 tokens across 304 rows in the checked subset.

Fixed-256 parity on 200,064 words matched all 1,296 rows. The first effective batch contained 39,370 words and identical inputs, attention masks and word groups. Both paths selected 8,247 masked tokens; initialization matched over 172 tensors with maximum difference 0. The first AdamW update had identical loss 10.683919814336338, gradient norm 2.291218 and post-update maximum state difference 0.

The repaired curriculum specified lengths 64 through 20M words, 128 through 50M and 256 through 100M, effective batch 256, microbatch 64, learning rate 0.001, warmup 0.05 and weight decay 0.01. It was a whole-training-schedule intervention: shorter early chunks change optimizer-update count and the learning-rate trajectory. It must not be described as context length alone or as identical warmup to every earlier baseline.

The fixed-256 reference cheap7 was 43.107851816373675 and Overall 41.140577774478444. The recorded decision thresholds were 43.1079 for improvement over baseline and approximately 43.77 for escalation to full evaluation. No 12x384 curriculum interaction was yet established.
