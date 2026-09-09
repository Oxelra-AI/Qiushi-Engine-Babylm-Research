# seed43222 threearm probe decomposition split in-window row-holdout controls

These pools remove paired source/companion co-occurrence from the MAX VIEW and REPEAT arms while preserving source and companion text exposure.

- pair count: 33,291
- original paired changed rows: 7,923
- source-only split rows: 4,640
- view companion-only split rows: 2,798
- repeat companion-only split rows: 2,798
- suffix rows reused from earlier analysis: 57,390
- both pools exact 10M words: True
- split-arm row length sequences identical: True

Pre-stated readout: compare split arms to CLEAN on token-nonoverlap rewrite gain and held-out copy gain at 80M/90M/100M. Near-CLEAN split behavior supports same-window relation practice; original-like split behavior supports cross-row repetition/augmentation budget.

Metadata: `experiments/archive/relation_learning/data/split_inwindow_rowholdout_pools/split_inwindow_rowholdout_metadata.json`
