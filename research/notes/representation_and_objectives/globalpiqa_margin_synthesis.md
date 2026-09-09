# fw globalpiqa relevant substrate — GlobalPIQA margin synthesis

## Core result

The official prediction artifacts had shown GlobalPIQA_parallel at ~22–28%, but not whether wrong answers were near ties. The CPU margin reader mirrors the official MLM length-normalized completion scoring and records all four option scores. It shows the failure is mostly not a near-tie/calibration issue.

| target | parallel acc | chance-adjusted | rank1 | rank2 | rank3 | rank4 | 52-row hard rank2/rank3/rank4 | hard mean top-correct nats |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `legal40k8x480_43022` | 22.33 | -0.036 | 23 | 24 | 26 | 30 | 11/18/23 | 1.953 |
| `legal16k8x480_43022` | 26.21 | 0.016 | 27 | 22 | 28 | 26 | 10/20/22 | 1.783 |
| `depth12x384_43022` | 24.27 | -0.010 | 25 | 28 | 26 | 24 | 13/18/21 | 1.859 |
| `inherited16k_noncompliant_43022` | 25.24 | 0.003 | 26 | 18 | 36 | 23 | 10/24/18 | 1.838 |

## Scientific reading

- The best compliant endpoint (`legal40k8x480_43022`) gets 23/103 correct, with correct-option ranks 1/2/3/4 = 23/24/26/30. On the 52 cross-endpoint hard rows, ranks are 2/3/4 = 11/18/23 and mean top-minus-correct is 1.95 nats; only 8 hard wrong rows have margin <=0.50 nats.
- Legal16k, depth, and inherited-tokenizer endpoints show the same pattern. Even the noncompliant 42.033 endpoint has hard-row ranks 2/3/4 = 10/24/18 and mean top-minus-correct 1.84 nats.
- Therefore GlobalPIQA_parallel is a stable conditional-world-knowledge/reasoning weakness of the lineage, not a legal-tokenizer artifact and not mainly a simple answer-position bias or small-margin scoring artifact. Inference-time calibration could be studied only on corpus-derived validation, but it is unlikely to supply the whole SOTA gap.
- Future compact/breadth FW results should be read with this margin lens: a real mechanism should improve hard-row rank distribution and reduce top-minus-correct margins, not merely flip a few noisy labels at 100M.

## Files

- summary JSON: `experiments/archive/representation_and_objectives/data/globalpiqa_margin_synthesis/globalpiqa_margin_synthesis.json`
- summary CSV: `experiments/archive/representation_and_objectives/data/globalpiqa_margin_synthesis/globalpiqa_margin_summary.csv`
- per-target detailed margins: `experiments/archive/representation_and_objectives/data/globalpiqa_margin_reader<target>_margins.json` and `<target>_parallel_rows.csv`
