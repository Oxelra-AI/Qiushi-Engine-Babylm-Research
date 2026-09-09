# rtd closure and muon route decision state-dependent RTD/MLM probe

Purpose: compare the same balanced hard-RTD head-fitting, shortcut-resistance, and MLM-vs-RTD gradient geometry across existing checkpoints before any delayed RTD tail training.

Probe data: 5184 examples, 799882 words, 21 batches (15 head-fit batches, 6 held-out batches).

## Summary

| checkpoint | hard bal acc | hard AUROC | random-hard AUROC gap | repl rate | trunk cosine | trunk RTD/MLM | rel cos | lambda1 rotation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mlm20_step35 | 0.6477 | 0.7159 | 0.2136 | 0.6968 | +0.0218 | 0.3186 | -0.0591 | 17.56° |
| mlm80_step35 | 0.6871 | 0.7622 | 0.1884 | 0.5224 | +0.0338 | 0.2506 | -0.2359 | 13.95° |
| rtd20_step094 | 0.7023 | 0.7913 | 0.1276 | 0.6925 | +0.0266 | 0.2258 | +0.1901 | 12.65° |

## Direct comparisons

| comparison | Δhard AUROC | Δrandom-hard AUROC gap | Δtrunk cosine | Δtrunk RTD/MLM | Δrotation |
|---|---:|---:|---:|---:|---:|
| mlm80_minus_mlm20 | +0.0463 | -0.0252 | +0.0121 | -0.0680 | -3.61° |
| rtd20_minus_mlm20 | +0.0754 | -0.0860 | +0.0048 | -0.0928 | -4.91° |

## Interpretation

- 80M vs 20M changes trunk cosine by +0.0121, RTD/MLM trunk norm ratio by -0.0680, hard AUROC by +0.0463, and random-hard AUROC gap by -0.0252.
- The RTD-trained 20M encoder differs from matched MLM20 by trunk cosine +0.0048, hard AUROC +0.0754, random-hard AUROC gap -0.0860, and lambda1 update rotation -4.91 degrees.

Elapsed: 189.7s
