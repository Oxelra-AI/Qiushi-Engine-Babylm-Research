# anchor matched controls — anchor-matched low-relation controls

## Purpose

The first anchor matched controls controls matched source and length but often differed in concrete content. This CPU-only repair collects controls that retain object/action/spatial/quantity/procedure anchors while suppressing explicit causal/change/contrast markers. The intended future comparison is transition structure versus anchored non-transition content, not merely concrete vocabulary versus unrelated text.

## Main counts

- Anchor-control candidate pool: 31252 sentences / 595209 words.
- Matching measurements are in `anchor_control_match_summary.csv`.

## Scientific reading

These controls make a later small probe cleaner, but they still do not justify any 100M run. The running FW compact/breadth endpoints must be read first. If they do not move the shared relation weakness, a small shared-coordinate probe can compare core transition slices against these anchor-matched controls before any larger route is considered.

## Files

- summary JSON: `experiments/archive/representation_and_objectives/data/anchor_matched_controls/anchor_matched_controls.json`
- anchor pool: `experiments/archive/representation_and_objectives/data/anchor_matched_controls/anchor_control_candidate_pool.jsonl`
- controls: `experiments/archive/representation_and_objectives/data/anchor_matched_controls/core_transition_anchor_control_50k.jsonl`, `experiments/archive/representation_and_objectives/data/anchor_matched_controls/core_transition_anchor_control_100k.jsonl`, `experiments/archive/representation_and_objectives/data/anchor_matched_controls/core_transition_anchor_control_200k.jsonl`
- summary CSV: `experiments/archive/representation_and_objectives/data/anchor_matched_controls/anchor_control_summary.csv`
- match summary CSV: `experiments/archive/representation_and_objectives/data/anchor_matched_controls/anchor_control_match_summary.csv`
