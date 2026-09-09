# aoa developmental signal audit — AoA developmental signal audit before training

## Why this audit was needed
The aoa analysis and curriculum design training proposal is not acceptable as written: it would use the exact official AoA words as a pretraining mask list, and it described ten epoch-specific 10M files rather than one documented ≤10M corpus repeated within the epoch limit. No training was launched. It treats the AoA file only as an evaluation-analysis object and asks whether a transferable, legal developmental schedule is supported by existing evidence.

## Corpus frequency pattern of official AoA words
### clean-Qwen 10M
- total words: 10,000,000; CDI token hits: 815,224; CDI types with hits: 503
- Pearson human AoA vs log corpus count: -0.1977; Spearman: -0.2026
  - early_le20: types 32, hits 71025, mean/type 2219.53, median/type 1016.50
  - mid_21_25: types 203, hits 315215, mean/type 1552.78, median/type 660
  - late_26_30: types 212, hits 349423, mean/type 1648.22, median/type 453.50
  - very_late_gt30: types 57, hits 79561, mean/type 1395.81, median/type 151
### official pool 10M
- total words: 10,000,000; CDI token hits: 832,977; CDI types with hits: 503
- Pearson human AoA vs log corpus count: -0.1984; Spearman: -0.2042
  - early_le20: types 32, hits 73115, mean/type 2284.84, median/type 1038.00
  - mid_21_25: types 203, hits 321324, mean/type 1582.88, median/type 668
  - late_26_30: types 212, hits 357205, mean/type 1684.93, median/type 453.00
  - very_late_gt30: types 57, hits 81333, mean/type 1426.89, median/type 151
### prior source-block first-pass order
- total words: 10,000,000; CDI token hits: 815,224; CDI types with hits: 503
- Pearson human AoA vs log corpus count: -0.1977; Spearman: -0.2026
  - early_le20: types 32, hits 71025, mean/type 2219.53, median/type 1016.50
  - mid_21_25: types 203, hits 315215, mean/type 1552.78, median/type 660
  - late_26_30: types 212, hits 349423, mean/type 1648.22, median/type 453.50
  - very_late_gt30: types 57, hits 79561, mean/type 1395.81, median/type 151

Interpretation: the post-hoc benchmark words are not a clean source-only developmental axis. In clean-Qwen and in the official pool, earlier-acquired words are actually more frequent per type on average, while the learned trajectory still fails to become a positive official AoA score. The failure is therefore not a simple global-frequency shortage that can be repaired by targeting benchmark words; it is a mismatch between corpus exposure, context distribution, optimization dynamics, and the official curve fit. AoA arithmetic alone is not enough.

## Existing trajectory pattern
### clean_qwen_seed43022
- represented CDI words in surprisal file: 328; rows: 124640
- corr human AoA vs final surprisal: Pearson 0.0340, Spearman 0.0259
- corr human AoA vs total drop: Pearson 0.0351, Spearman 0.0250
- corr human AoA vs log half-drop time: Pearson -0.1108, Spearman -0.1362, n=252
  - chck_1M: early_le20:9.36, mid_21_25:9.43, late_26_30:9.61, very_late_gt30:9.60
  - chck_10M: early_le20:8.90, mid_21_25:8.97, late_26_30:8.95, very_late_gt30:9.05
  - chck_100M: early_le20:8.57, mid_21_25:8.38, late_26_30:8.64, very_late_gt30:8.57
### clean_qwen_seed43122
- represented CDI words in surprisal file: 328; rows: 124640
- corr human AoA vs final surprisal: Pearson 0.0568, Spearman 0.0668
- corr human AoA vs total drop: Pearson -0.0241, Spearman -0.0325
- corr human AoA vs log half-drop time: Pearson -0.1309, Spearman -0.1506, n=254
  - chck_1M: early_le20:9.44, mid_21_25:9.47, late_26_30:9.66, very_late_gt30:9.36
  - chck_10M: early_le20:9.38, mid_21_25:9.28, late_26_30:9.35, very_late_gt30:9.12
  - chck_100M: early_le20:8.57, mid_21_25:8.29, late_26_30:8.64, very_late_gt30:8.60
### devcurr_firstpass_seed43022
- represented CDI words in surprisal file: 328; rows: 124640
- corr human AoA vs final surprisal: Pearson 0.0191, Spearman -0.0011
- corr human AoA vs total drop: Pearson 0.0728, Spearman 0.0737
- corr human AoA vs log half-drop time: Pearson 0.1976, Spearman 0.1912, n=216
  - chck_1M: early_le20:9.03, mid_21_25:9.30, late_26_30:9.49, very_late_gt30:9.50
  - chck_10M: early_le20:9.35, mid_21_25:9.29, late_26_30:9.48, very_late_gt30:9.26
  - chck_100M: early_le20:8.82, mid_21_25:8.61, late_26_30:8.77, very_late_gt30:8.87
### devcurr_firstpass_seed43122
- represented CDI words in surprisal file: 328; rows: 124640
- corr human AoA vs final surprisal: Pearson 0.0012, Spearman -0.0135
- corr human AoA vs total drop: Pearson 0.0719, Spearman 0.0855
- corr human AoA vs log half-drop time: Pearson 0.1494, Spearman 0.1643, n=247
  - chck_1M: early_le20:9.24, mid_21_25:9.36, late_26_30:9.55, very_late_gt30:9.36
  - chck_10M: early_le20:9.14, mid_21_25:9.19, late_26_30:9.55, very_late_gt30:9.10
  - chck_100M: early_le20:8.60, mid_21_25:8.38, late_26_30:8.51, very_late_gt30:8.53

The already-trained safe first-pass source-order intervention produced AoA=0.0 for both seeds. For seed43022 it also lowered the broad official-like Overall from the inherited clean-Qwen 41.3443 to 40.6475, with Entity dropping 25.76→23.72 and GlobalPIQA 36.62→34.225. Thus source-block timing has already been falsified as a useful SOTA lever in this form.

## Consequence for next execution
Do not train any curriculum that uses the official AoA target list as a mask or sampling signal. A legal next candidate must be a permutation or schedule over one ≤10M documented corpus, with each row appearing at most once per epoch, and must be justified by training-only or external developmental evidence such as source type, lexical/conceptual difficulty, dialogue/concreteness proxies, or independent norms not identical to the official target file. Before a 100M run, build a small auditable corpus-order or source-reweighting design and predict its effect on both AoA trajectory and the broad task surface; if it mainly attacks AoA while risking Entity/GlobalPIQA/SuperGLUE, it is not the main route. The semantic-view contrast remains more directly connected to the 41.8 gap; wait for its score evidence rather than consuming H100 time on an unsupported AoA curriculum.

Machine-readable audit: `experiments/archive/frontier_consolidation/data/aoa_developmental_audit/aoa_developmental_signal_audit.json`

## Update: official curve-fit reproduction and route implication

A follow-up CPU-only reproduction of the official AoA curve-fit logic is saved at `experiments/archive/frontier_consolidation/data/aoa_developmental_audit/official_aoa_curvefit_audit.json` with note `research/notes/frontier_consolidation/official_aoa_curvefit_audit.md`. It distinguishes several populations that the earlier shorthand could conflate: 504 CDI rows in `cdi_human.csv`; 503 types observed in the clean-Qwen corpus at least once; 328 words represented in the local `surprisal.json` files after context filtering; 290 candidate words with a computable child AoA in the reproduced scorer; and 155--197 valid fitted model AoAs depending on model/seed, with 118 words common across all four existing models.

The exact official mechanism is trajectory-based: average surprisal per word and checkpoint, fit a bounded sigmoid to negative surprisal over log10 word exposure, compute the fitted exposure where the curve crosses the midpoint between tokenizer-scaled random-chance surprisal and that word's minimum surprisal, then Pearson-correlate fitted model AoA with child AoA; p>0.1 maps the official score to 0.0. Reproduction results: clean-Qwen seed43022 r=-0.1109 p=0.1382, clean-Qwen seed43122 r=-0.1019 p=0.1543, devcurr-firstpass seed43022 r=-0.0572 p=0.4794, devcurr-firstpass seed43122 r=-0.1047 p=0.1691; all threshold to official AoA 0.0. On the common 118-word fitted subset, no model has a positive significant relation; clean seed43022 is significantly negative and the others are negative but non-significant.

Matched broad-score evidence for the prior legal first-pass source-order intervention should be stated with both seeds: seed43022 Overall 41.3443 -> 40.6475 (Δ -0.6968), seed43122 Overall 40.6501 -> 40.5656 (Δ -0.0844), two-seed mean Δ -0.3906. Entity decreases in both seeds, but GlobalPIQA decreases only in seed43022 and rises in seed43122, so GlobalPIQA damage is not a replicated mechanism. The replicated conclusion is narrower and stronger: a corpus-only first-pass source ordering altered timing proxies but did not create a positive official AoA curve-fit relation and did not improve Overall.

Scientific consequence: do not spend H100 time on another pure AoA ordering/curriculum run unless it is part of a broader, independently motivated corpus/objective experiment with a full broad-task readout. AoA remains useful as a passive trajectory measurement. The near-term SOTA route should be driven by broad task-surface evidence: the peer semantic-view contrast when it finishes, a cleaned seq256-safe FineWeb/source-breadth contrast if semantic views are weak or as an orthogonal factor, and a possible tokenizer-aware exposure/packing control. Any future developmental curriculum must be defined without official AoA target identities or labels and over one documented ≤10M corpus repeated no more than 10 epochs.
