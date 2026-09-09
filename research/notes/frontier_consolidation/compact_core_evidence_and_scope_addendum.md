# Compact-Core Evidence Before Reinvestment

Status: completed fast screen at one seed; full evaluation and replication were still pending.

The decisive contrast was compact semantic views versus exact repetition on the same core, not the later reinvestment arm. Both retained the same overlay, DeBERTa-v2 8x480, 16k tokenizer, sequence length 256, WWM, AdamW, seeds 43/43022/43023 and 100M exposure.

| Metric | Repeat | Compact view | Difference |
|---|---:|---:|---:|
| BLiMP | 67.09 | 66.93 | -0.16 |
| Supplement | 58.00 | 65.60 | +7.60 |
| EWoK | 48.64 | 51.55 | +2.91 |
| Entity | 25.11 | 27.30 | +2.19 |
| Entity full | 25.42 | 27.85 | +2.43 |
| COMPS | 51.09 | 52.18 | +1.09 |
| GlobalPIQA mean | 34.105 | 35.135 | +1.03 |
| Reading | 8.02 | 8.25 | +0.23 |
| equal7 | 41.7221 | 43.8493 | +2.1271 |
| equal7, full Entity | 41.7664 | 43.9279 | +2.1614 |

The view model ended with higher MLM loss despite the broader fast-screen gains. The source assessment described compact views as moderately noisy, not proposition-perfect. The result supported compressed same-source views over repetition in this coordinate, without proving a pure information-density mechanism.

Full evaluation was necessary because SuperGLUE and AoA could change the complete result. A negative reinvestment extension would not refute the core contrast: reinvestment separately asks whether saved words can buy useful additional sources. A complete ranking or seed-stable treatment claim remained premature.
