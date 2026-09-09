# full ewok interaction synthesis — projected preserved compact-view yield

The qwen35 teacher recovery pilot was deliberately enriched for semantically fragile sources, so its low usable rate should not be read as the full 26,015-source yield without reweighting. This CPU-only projection compares feature distributions and estimates how much pair volume may survive the full ewok interaction synthesis preservation standard.

## Distribution contrast

- Full new prompts: 26,015 rows, 614,161 source words; risk-count fractions {'0': 0.48218335575629445, '1': 0.3342302517778205, '2': 0.14307130501633672, '3': 0.03690178743032866, '4': 0.003613300019219681}
- Pilot prompts: 256 rows, 6,912 source words; risk-count fractions {'0': 0.01171875, '1': 0.2578125, '2': 0.44140625, '3': 0.265625, '4': 0.0234375}

## Measured usable anchors

- Existing Qwen3.5 under full ewok interaction synthesis standard: 8,386/12,152 usable, 279,128 pair words.
- New Qwen3.5 pilot under full ewok interaction synthesis standard: 89/256 usable, 3,990 pair words.

## Estimated full-scale preserved volume

- pilot_overall_rate: new usable pair words ≈ 357804; with measured existing ≈ 636932 (0.426 of target), missing ≈ 857178.
- pilot_by_risk_count: new usable pair words ≈ 394272; with measured existing ≈ 673400 (0.451 of target), missing ≈ 820710.
- pilot_by_risk_count_and_length: new usable pair words ≈ 407993; with measured existing ≈ 687121 (0.460 of target), missing ≈ 806989.
- existing_by_risk_count: new usable pair words ≈ 688554; with measured existing ≈ 967682 (0.648 of target), missing ≈ 526428.
- existing_by_risk_count_and_length: new usable pair words ≈ 661227; with measured existing ≈ 940355 (0.629 of target), missing ≈ 553755.

Summary JSON: `experiments/archive/representation_and_objectives/data/fw_yield_projection/yield_projection.json`
