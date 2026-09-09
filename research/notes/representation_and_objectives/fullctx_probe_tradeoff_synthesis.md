# fullctx aux budget audit — full-context pivot-substitution tradeoff synthesis

## Question

The full-context probe was applied to the completed FW compact, row-block breadth, and interleaved breadth 100M checkpoints. The purpose was to test whether its margins are tied to the known EWoK/GlobalPIQA relation tradeoff or whether all endpoints already learn this local compatibility while still failing the downstream relation surfaces.

## Shared event set

- Event counts: {'compact': 2266, 'rowblock': 2266, 'interleaved': 2266}; common events: 2266; symmetric differences: {'compact_vs_rowblock': 0, 'compact_vs_interleaved': 0, 'interleaved_vs_rowblock': 0}

## Main full-context true-vs-shuffled-pivot margin

- compact: mean=0.487175, SE=0.029113

- rowblock: mean=0.545990, SE=0.030148

- interleaved: mean=0.540041, SE=0.031580

- event-level Pearson correlations: {'compact__rowblock': 0.8642265135481845, 'compact__interleaved': 0.889850035726196, 'interleaved__rowblock': 0.8914154092517362}

- range across arms = 0.058815; range / average event SD = 0.0408

## Known relation surfaces

- EWoK wrong-row interaction median (less negative better): compact -0.956, rowblock -0.505, interleaved -0.666. Stable-failure fraction among wrong rows: compact 0.689, rowblock 0.646, interleaved 0.682. Aggregate EWoK accuracy: compact 0.504, rowblock 0.496, interleaved 0.511.

- GlobalPIQA parallel accuracy: compact 24.27, rowblock 29.13, interleaved 26.21. Hard52 top-minus-correct margin (lower better): compact 1.717, rowblock 1.433, interleaved 1.660. Nonparallel accuracy: compact 53.0, rowblock 45.0, interleaved 56.0.

## Interpretation

The probe has a small model-specific offset in the same direction as row-block/interleaved relation diagnostics, but all three failing endpoints already show strong true-pivot preference on the identical events and event-level margins are almost the same object across arms. It is therefore not yet a validated training objective for repairing EWoK/GlobalPIQA failures.

The arm ordering of the probe margin (rowblock > interleaved > compact) is compatible with the GlobalPIQA-parallel and EWoK wrong-median/stable-failure tradeoff, but the absolute signal is already strongly positive in every endpoint, including compact and interleaved checkpoints that still leave 61/103 GlobalPIQA-parallel rows wrong for all arms and thousands of EWoK stable conditional reversals. The cross-arm offset is small relative to event variability and margins are highly paired across the identical events, so directly increasing this local margin may simply reinforce a compatibility relation the models already express rather than repair hard conditional choice/ranking.

## Consequence for training

Do not launch the full context pivot substitution probe full-context auxiliary unchanged. Before any continuation, the design must (1) include a same-exposure WWM reference, (2) debit every auxiliary view/forward as training exposure under Strict-Small accounting, (3) equalize nonsemantic gradient pressure between semantic and placebo arms, and (4) connect the auxiliary to hard-row or four-cell failures rather than only to local attested-pivot fit.

## Artifacts

- summary: `experiments/archive/representation_and_objectives/data/fullctx_probe_tradeoff_synthesis/fullctx_probe_tradeoff_synthesis.json`

- csv: `experiments/archive/representation_and_objectives/data/fullctx_probe_tradeoff_synthesis/probe_tradeoff_arm_metrics.csv`

