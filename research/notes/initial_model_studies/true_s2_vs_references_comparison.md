# true s2 100m available coordinate — true S2 100M vs S1/protected/visible leader

Evidence JSON: `experiments/archive/initial_model_studies/data/true_s2_vs_references_comparison.json`

| column | true S2 100M | S1 100M | S2-S1 | protected 8×480 | S2-protected | visible leader | S2-leader |
|---|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 64.24 | 66.84 | -2.60 | 66.76 | -2.52 | 67.20 | -2.96 |
| Supplement | 59.09 | 60.31 | -1.22 | 59.88 | -0.79 | 56.01 | +3.08 |
| EWoK | 51.64 | 52.02 | -0.38 | 52.19 | -0.55 | 56.07 | -4.43 |
| Entity | 18.47 | 20.24 | -1.77 | 22.62 | -4.15 | 28.45 | -9.98 |
| COMPS | 50.61 | 52.26 | -1.65 | 52.19 | -1.58 | 53.57 | -2.96 |
| GlobalPIQA | 38.63 | 37.60 | +1.03 | 35.63 | +3.00 | 39.67 | -1.03 |
| Reading | 7.52 | 7.25 | +0.27 | 7.62 | -0.10 | 5.42 | +2.10 |

## Aggregate available columns

- 7-column available mean (BLiMP/Supp/EWoK/Entity/COMPS/GlobalPIQA/Reading): true S2 41.458, S1 42.361, protected 42.414; S2-S1 -0.903.
- 6-column NLP subset without SuperGLUE: true S2 47.114, S1 48.213; S2-S1 -1.098.

## Interpretation

- True S2 is not a positive overall curriculum component on the official-corpus baseline16k S1 base: it improves only GlobalPIQA (+1.03 vs S1) and Reading (+0.27), while damaging BLiMP (-2.60), Supplement (-1.22), EWoK (-0.38), Entity (-1.77), and COMPS (-1.65).
- The 100M word-clock curriculum successfully tests the real leader-style schedule; the result must not be replaced by compressed 10M probes. Its column pattern says late token masking/length curriculum over official corpus/baseline16k trades grammar/entity/EWoK for GlobalPIQA/nonparallel PIQA rather than closing the leader package.
- S2 remains far from the visible leader on Entity (-9.98), EWoK (-4.43), COMPS (-2.96), BLiMP (-2.96), and GlobalPIQA (-1.04). It exceeds the leader on Supplement and Reading, but these are not the central gaps.
- Because both S1 and S2 fail to move Entity/EWoK upward, the missing leader factor is unlikely to be architecture shape plus curriculum alone under official-corpus/baseline16k/AdamW. The next high-value route should test the legal representation/data side (40k tokenizer and/or legally reconstructed simplification pairs) or the planned GPT-BERT/MNTP hybrid, not continue stacking curriculum variants on S1.
