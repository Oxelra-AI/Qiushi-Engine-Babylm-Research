# fw absolute progress decision discipline — FW comparison decision discipline

The earlier analysis FineWeb compact-view vs whole-sentence source-breadth experiment has two distinct scientific meanings that must not be collapsed.

## 1. Within-pair mechanism meaning

Compact minus breadth at matched checkpoints identifies how the expanded FineWeb companion budget is being used:

- `compact_view`: 318,851 FineWeb companion words are faithful short semantic restatements of the same source propositions.
- `source_breadth`: those 318,851 words are coherent independent FineWeb sentences from non-overlapping source hashes.

The predeclared ±0.3 cheap7 rule remains useful for this **within-family** mechanism comparison. It does not by itself authorize further GPU training.

## 2. Absolute SOTA-progress meaning

Progress toward the active goal requires the winning arm to improve broadly over the existing fully legal compact-view-reinvest trajectory. The relevant active reference is the spatial repair route status legal same-pool-tokenizer endpoint/trajectory, not just the other FW arm.

Reference cheap7 values:

| Checkpoint | Legal reference cheap7 | Source |
|---|---:|---|
| 70M | 42.6086 | `data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_70M.json` |
| 80M | 42.9486 | `data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_80M.json` |
| 100M | 43.0057 | spatial repair route status endpoint in `data/compliant_endpoint_results/compliant_endpoint_results_summary.json` |

The current complete legal endpoint is Overall 41.257770896404615. Reaching the visible 41.8 target needs +0.542229 Overall, equivalent to about +0.6977 cheap7 if only the seven cheap columns move and SuperGLUE/AoA stay unchanged.

Caveat: the earlier analysis FW arms are not perfectly matched to the spatial repair route status legal reference. They use shared tokenizer (SHA `e70d167f`, trained on 9,681,149 words) and an expanded FineWeb pair construction (813,005 FW pair words). Therefore compact-vs-breadth is the clean causal comparison; reference deltas are a SOTA-progress test across a changed legal substrate.

## Operational rule now encoded

Updated files:

- `scripts/eval_fw_comparison.py`
  - parser already validated against tail rephase experiment design per-target JSON (`max_abs_error=0.0`),
  - now supports `--arm compact_view` or `--arm source_breadth` for minimum one-arm follow-up,
  - merges new 100M one-arm results into existing 70M/80M summary instead of overwriting.
- `scripts/fw_absolute_decision.py`
  - reads the earlier analysis evaluation summary,
  - reports each arm's absolute delta versus the legal reference at 70M/80M/100M,
  - writes `data/fw_absolute_decision/fw_absolute_decision.{json,md}` after real scores exist.

## Next decision sequence

1. Do not poll the managed trainings. When they complete, first verify both reached 100M and checkpoint ladders are complete.
2. Run cheap official-compatible evaluations at 70M and 80M for both arms:
   `python3 experiments/archive/frontier_consolidation/scripts/eval_fw_comparison.py --checkpoint 70M --checkpoint 80M --gpu 0`
3. Run:
   `python3 experiments/archive/frontier_consolidation/scripts/fw_absolute_decision.py`
4. Interpret the result in two layers:
   - within-family compact-vs-breadth delta identifies companion-budget mechanism;
   - absolute delta versus legal reference identifies whether the FW family is advancing toward SOTA.
5. Do **not** automatically launch the `source_repeat` attribution arm when compact-breadth lies within ±0.3. If both FW arms are flat or below legal reference at 70M/80M (and, if already worth checking, 100M), source-repeat would only explain a non-advancing family.
6. If one arm shows broad gain large enough to matter at 80M, use the minimum additional evaluation needed: usually cheap7 at 100M for that arm only via:
   `python3 experiments/archive/frontier_consolidation/scripts/eval_fw_comparison.py --checkpoint 100M --arm <winner> --gpu 0`
   then rerun fw absolute progress decision discipline decision.
7. Run full official-compatible evaluation or missing official columns only if the 100M cheap7 gain is large enough to plausibly close the complete nine-column gap.

No new GPU route is authorized by this note itself.
