# earlier analysis conservation-vector readout

File-only readout of already scored interventions. It compares each family-delta vector's broad mean with its vector amplitude. No model loading, training, evaluation, GPU work, upload, or leaderboard action occurred.

## Measurement

For a vector of family deltas Δ, `net = mean(Δ)`, `RMS = sqrt(mean(Δ²))`, and `centered_RMS = sqrt(mean((Δ-net)²))`. A small `|net|/RMS` means the score vector moves substantially while the broad mean changes little.

## Fixed-budget dose: V-R over common 10M–80M

| dose | rho | stable6 net | stable6 RMS | centered RMS | |net|/RMS | ex-Entity net | ex-Entity RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| dose1 | 0.042352 | -0.0625 | 0.5356 | 0.5319 | 0.117 | -0.1868 | 0.5308 |
| dose1p82 | 0.077120 | +0.1414 | 0.6939 | 0.6794 | 0.204 | -0.0436 | 0.5920 |
| dose2p64 | 0.111872 | +0.3968 | 0.9912 | 0.9083 | 0.400 | +0.0636 | 0.5729 |

Across the three dose points, stable6 V-R RMS has slope 6.553494441889523 per rho and correlation 0.9848392454203106; centered RMS has slope 5.414047993640581 and correlation 0.9922686833494584. The ex-Entity net stays small, with trend [(0.042352, -0.18675000000000044), (0.07712, -0.04362500000000047), (0.111872, 0.06362500000000026)].

Interpretation: the high-dose V-R vector is not a clean broad addition. The all-family mean grows mainly because Entity enters the average, while the non-Entity broad mean remains close to zero compared with the family-vector amplitude.

## Visible 80M breadth reference point

| contrast | stable6 net | stable6 RMS | centered RMS | ex-Entity net | ex-Entity RMS | vector |
|---|---:|---:|---:|---:|---:|---|
| D1_VminusB | +0.4650 | 1.5864 | 1.5167 | -0.0620 | 1.0478 | BLiMP:-0.87, COMPS:-1.18, EWoK:-0.15, Entity:+3.10, Reading:+0.07, Supplement:+1.82 |
| D1_BminusCold | +0.7717 | 1.1820 | 0.8953 | +0.8960 | 1.2931 | BLiMP:+2.59, COMPS:+0.83, EWoK:+0.97, Entity:+0.15, Reading:+0.14, Supplement:-0.05 |
| D1_VminusCold | +1.2367 | 1.7075 | 1.1773 | +0.8340 | 1.1773 | BLiMP:+1.72, COMPS:-0.35, EWoK:+0.82, Entity:+3.25, Reading:+0.21, Supplement:+1.77 |
| D1_BminusR | +0.5575 | 1.0247 | 0.8598 | +0.3890 | 0.9317 | BLiMP:+1.14, COMPS:+1.31, EWoK:-0.78, Entity:+1.40, Reading:-0.45, Supplement:+0.72 |
| D1_VminusR | +1.0225 | 2.1524 | 1.8940 | +0.3270 | 1.2286 | BLiMP:+0.27, COMPS:+0.13, EWoK:-0.93, Entity:+4.50, Reading:-0.38, Supplement:+2.54 |

The current visible V-B point has Entity +3.10 and Supplement +1.82, but BLiMP -0.87 and COMPS -1.18; its ex-Entity mean is -0.062 while RMS is about one score point. This matches reply: source-related pairing can help exact/state-style routing without predicting broad ex-Entity gain.

## Entity operation strata

| contrast | official mean | neutral mean | zero/nonzero net | zero/nonzero RMS | vector |
|---|---:|---:|---:|---:|---|
| deberta_basin1 | +4.0175 | -1.4539 | -1.4539 | 8.3349 | nonzero_ops:+6.75, zero_ops:-9.66 |
| deberta_basin2 | +2.5392 | -2.1226 | -2.1226 | 7.3077 | nonzero_ops:+4.87, zero_ops:-9.12 |
| deberta_breadth_minus_repeat | +1.0234 | -4.6008 | -4.6008 | 9.6094 | nonzero_ops:+3.84, zero_ops:-13.04 |
| deberta_view_minus_breadth | +2.9941 | +3.1470 | +3.1470 | 3.1553 | nonzero_ops:+2.92, zero_ops:+3.38 |
| roberta_max | +0.2587 | +0.1367 | +0.1367 | 0.2284 | nonzero_ops:+0.32, zero_ops:-0.05 |

V-B is positive in both Entity operation strata, unlike V-R and B-R. That means the same-sign Entity specificity is real for this surface, but it is not broad on the visible 80M family vector.

## Mature scale1p75 item mechanism and cross route tradeoff archive

- mature cheap7 n: 8
- mature cheap7 mean_net: -0.31999999999999945
- mature cheap7 mean_abs_net: 0.3639285714285709
- mature cheap7 mean_rms_family_delta: 1.5937029780816754
- mature cheap7 mean_centered_rms_family_delta: 1.5109276749961031
- mature cheap7 mean_abs_net_over_rms: 0.2460109230925913
- mature cheap7 count_all_positive: 0
- mature cheap7 labels: ['Minfreq50 support-floor:80M:cheap7 80M', 'Word-mean MLM:80M:cheap7 80M', 'Strict content-innovation WWM:80M:cheap7 80M', 'Muon wdmatched lr0.008:80M:cheap7 80M', 'Muon20->AdamW:80M:cheap7 80M', 'Muon40->AdamW:80M:cheap7 80M', 'Expanded FineWeb compact-view:100M:cheap7 100M', 'Expanded FineWeb source-breadth:100M:cheap7 100M']

Largest mature archive vector movements after subtracting the broad mean:

| label | exposure | net | RMS | centered RMS | |net|/RMS | max column |
|---|---:|---:|---:|---:|---:|---|
| Word-mean MLM:80M:cheap7 | 80M | -0.2457 | 2.1033 | 2.0889 | 0.117 | GlobalPIQA +4.53 |
| Muon wdmatched lr0.008:80M:cheap7 | 80M | -0.1607 | 1.7696 | 1.7623 | 0.091 | GlobalPIQA +3.48 |
| Muon20->AdamW:80M:cheap7 | 80M | -0.1436 | 1.6554 | 1.6491 | 0.087 | Entity -3.47 |
| Expanded FineWeb source-breadth:100M:cheap7 | 100M | -0.3686 | 1.6957 | 1.6552 | 0.217 | Entity -3.52 |
| Expanded FineWeb compact-view:100M:cheap7 | 100M | +0.1757 | 1.4325 | 1.4217 | 0.123 | GlobalPIQA +2.57 |
| Minfreq50 support-floor:80M:cheap7 | 80M | -0.1679 | 1.4170 | 1.4070 | 0.118 | GlobalPIQA +2.54 |
| Strict content-innovation WWM:80M:cheap7 | 80M | -1.0529 | 1.6241 | 1.2366 | 0.648 | Supplement -2.72 |
| Muon40->AdamW:80M:cheap7 | 80M | -0.5964 | 1.0520 | 0.8665 | 0.567 | Entity -1.84 |

The scale1p75 item mechanism and cross route tradeoff mature archive has mean cheap7 movement -0.320 with no all-column positive record, but mean RMS much larger than the net. This turns the old route failures into quantitative evidence for a finite-budget redistribution regularity rather than isolated negative results.

## Cross-architecture selected compact-vs-repeat

- gpt2_causal_compact_minus_repeat:cheap7: n=6, mean_net=+0.0777, mean_abs_net=0.3673, mean_RMS=1.4125, mean_centered_RMS=1.3415, mean_|net|/RMS=0.262
- gpt2_causal_compact_minus_repeat:stable5_exEntity: n=6, mean_net=-0.0065, mean_abs_net=0.1952, mean_RMS=0.6927, mean_centered_RMS=0.6622, mean_|net|/RMS=0.254
- gpt2_causal_compact_minus_repeat:stable5_noGlobalPIQA_noReading: n=6, mean_net=-0.1513, mean_abs_net=0.2300, mean_RMS=0.9227, mean_centered_RMS=0.8845, mean_|net|/RMS=0.224
- gpt2_causal_compact_minus_repeat:stable6_noGlobalPIQA: n=6, mean_net=-0.1446, mean_abs_net=0.2388, mean_RMS=0.8572, mean_centered_RMS=0.8151, mean_|net|/RMS=0.275
- roberta_compact_minus_repeat:cheap7: n=2, mean_net=-0.2450, mean_abs_net=0.2871, mean_RMS=1.1104, mean_centered_RMS=1.0470, mean_|net|/RMS=0.244
- roberta_compact_minus_repeat:stable5_exEntity: n=2, mean_net=-0.1715, mean_abs_net=0.1715, mean_RMS=1.1543, mean_centered_RMS=1.1360, mean_|net|/RMS=0.146
- roberta_compact_minus_repeat:stable5_noGlobalPIQA_noReading: n=2, mean_net=-0.2620, mean_abs_net=0.2620, mean_RMS=1.1706, mean_centered_RMS=1.1392, mean_|net|/RMS=0.222
- roberta_compact_minus_repeat:stable6_noGlobalPIQA: n=2, mean_net=-0.2429, mean_abs_net=0.2429, mean_RMS=1.0841, mean_centered_RMS=1.0495, mean_|net|/RMS=0.220

## Item-threshold movement

| label | gains | losses | net | changed | net/changed | source |
|---|---:|---:|---:|---:|---:|---|
| coherent86 alpha1 minus chck82 anchor | 3116 | 3231 | -115 | 6347 | -0.0181 | `experiments/archive/frontier_consolidation/data/truthful_coherent86_carrier/truthful_coherent86_carrier_manifest.json` |
| coherent private-alpha monotonic decision pattern | 3116 | 3231 | -115 | 6347 | -0.0181 | `research/documents/frontier_consolidation/data/alpha_sweep_decision_patterns/alpha_sweep_decision_patterns.md` |

## Research reading

The current finite-budget hypothesis is sharpened, not completed: content-presentation changes often create substantial family-vector and item-threshold movement while broad non-Entity means remain small. The still-running MAX-geometry clean result decides whether content admission relative to matched clean placement gives a genuine broad positive leg; if it survives, the additive lever is which experience enters the 10M/100M budget, while source-related companions remain a narrower state/entity specificity mechanism requiring breadth/permuted follow-up. If it collapses, the conservation reading becomes stronger and the program should shift away from more presentation variants.

## Files
- dose_csv: `experiments/archive/frontier_consolidation/data/conservation_vector_readout/dose_common10_80_net_norm.csv`
- visible_reference_csv: `experiments/archive/frontier_consolidation/data/conservation_vector_readout/visible_reference_net_norm.csv`
- archive_csv: `experiments/archive/frontier_consolidation/data/conservation_vector_readout/archive_net_norm.csv`
- gpt2_roberta_csv: `experiments/archive/frontier_consolidation/data/conservation_vector_readout/gpt2_roberta_net_norm.csv`
- item_turnover_csv: `experiments/archive/frontier_consolidation/data/conservation_vector_readout/item_turnover_rows.csv`
- summary_json: `experiments/archive/frontier_consolidation/data/conservation_vector_readout/conservation_vector_summary.json`
- summary_md: `research/documents/frontier_consolidation/data/conservation_vector_readout/conservation_vector_summary.md`
- plot: `experiments/archive/frontier_consolidation/data/conservation_vector_readout/dose_vr_net_vs_norm.png`
- plot: `experiments/archive/frontier_consolidation/data/conservation_vector_readout/archive_mature_net_vs_norm.png`
- plot: `experiments/archive/frontier_consolidation/data/conservation_vector_readout/visible_80M_vminusb_family_vector.png`
