# topology 2x2 scaffold and deberta pending state: topology 2×2 scaffold and pending DeBERTa state

## Research context

The active question is still a generalizable, rule-compliant data-efficient learning advance for BabyLM Strict-Small and beyond, not merely another endpoint carrier. The secured public fallback is scale1.75 `chck_82M` (public 41.94), and coherent86 alpha0.75 remains the strongest local endpoint carrier (Overall(AoA0) 42.1210) but is classified as amplitude-controlled redistribution.

causal transfer result synthesis showed that the original one-way GPT2 causal compact-vs-repeat construction did not support architecture-general compact semantic views: compact's mean cheap7 edge was +0.0777 but became negative without GlobalPIQA and Reading, and relation/state was negative. reciprocal multiview mechanism and scaffold sharpened the mechanism: causal examples only give paired lift to the later segment, but DeBERTa MLM reciprocal lift was dominated by copied tokens; compact non-copy lift was tiny (+0.0616 on rewrite side, negative on source side). The reciprocal scaffold therefore is not sufficient reason for new 100M runs. If the DeBERTa triangle scores revive the direction, isolate the interaction first using a low-cost 2×2.

## Group sync

Triangle scores remained unavailable. No GPU training or model evaluation was launched for this note.

## Corrected four-arm causal topology 2×2 scaffold

Builder: `scripts/build_causal_topology_2x2.py`.

Output: `data/causal_topology_2x2_scaffold/`.

The construction crosses:

- semantic content: `compact` vs `repeat`
- topology: `oneway` vs `reciprocal`

The important design repair is that `oneway` does not receive half the recurrence. Every arm uses the same 6,071 selected pairs, and every selected pair appears twice. In the `oneway` arms, the pair appears twice in the same causal gpt transfer experiment design base direction; in the `reciprocal` arms, it appears once in the causal gpt transfer experiment design base direction and once in the opposite direction. Thus pair recurrence and pair-word dose are held constant, while topology changes.

The builder was also repaired in this analysis so the base order is recovered from the exact causal gpt transfer experiment design full original-pair order assignment and then subset by `pair_id`, rather than regenerated over the shuffled subset. This makes the `oneway` cells closer to the earlier causal coordinate. A manifest-label bug was also repaired: the truncated filler row no longer starts with the `topology2x2_` pair prefix, so pair-row counts are correct.

Final scaffold facts from `manifest.json`:

- selected pairs: 6,071
- pair units per arm: 12,142
- pair words per arm: 423,512, matching the causal gpt transfer experiment design pair budget of 423,511 to +1 word
- filler words per arm: 9,576,488 from the same neutral filler source
- legal words per arm: exactly 10,000,000
- rows per arm: 73,877
- base order counts: source_first 3,035, view_first 3,036
- pair positions match across all four arms: true
- row word sequences match across all four arms: true
- neutral tokenizer SHA: `e6723383f9642945134a09a9641d52e595624d3c5a99622e2a47e4d912378102`
- rewrite/source word-multiset overlap among selected compact rewrites: mean 0.82945, median 0.83333. This is another warning that copied lexical material remains a large confound.

Final arm hashes:

| arm | SHA256 | raw tokens/epoch incl EOS | active 256-token positions/epoch | chunks/epoch | steps/epoch at batch 256 |
|---|---|---:|---:|---:|---:|
| compact_oneway | `13213897f9c54fe651ca407db0be4ed03d8df6a5c6f71f390ad76bd813cd0ac5` | 14,587,708 | 14,587,648 | 56,983 | 223 |
| repeat_oneway | `81ed5960c80cbc9ef0636a118a4d43017ccbf42c3eff03b22e3b902f5d158666` | 14,556,070 | 14,555,904 | 56,859 | 223 |
| compact_reciprocal | `94946452b42c0b3f100612436f7675bd00e3bba154d29c25a32dd9504b979f10` | 14,587,760 | 14,587,648 | 56,983 | 223 |
| repeat_reciprocal | `cfc708eb4d352a9892932c2ca441015fcc19956db110a63f1016e8940283de26` | 14,556,070 | 14,555,904 | 56,859 | 223 |

Token comparisons:

- compact minus repeat within oneway: +31,638 raw tokens, +31,744 active positions, +124 chunks per epoch.
- compact minus repeat within reciprocal: +31,690 raw tokens, +31,744 active positions, +124 chunks per epoch.
- reciprocal minus oneway within compact: +52 raw tokens, 0 active positions, 0 chunks per epoch.
- reciprocal minus oneway within repeat: 0 raw tokens, 0 active positions, 0 chunks per epoch.

The warning printed by the tokenizer (`270 > 256`) reflects row-level lengths under the neutral tokenizer. The causal trainer concatenates rows with EOS and chunks the stream into length-256 segments; it does not feed each row as one model input. Therefore this warning is not a training error for `causal_gpt_trainer.py`, but it would matter for any different row-as-example trainer.

## Future-only interaction comparator

Comparator: `scripts/compare_causal_topology_2x2.py`.

Smoke output: `data/topology_2x2_comparator_smoke/out/`.

The comparator consumes four completed selected-causal trajectories and computes the semantic-by-topology interaction:

\[
[(\mathrm{compact}_{reciprocal} - \mathrm{repeat}_{reciprocal}) - (\mathrm{compact}_{oneway} - \mathrm{repeat}_{oneway})]
\]

It reports the interaction for cheap7, cheap6(no GlobalPIQA), cheap5(no GlobalPIQA/Reading), syntax/surface, relation/state, volatile small columns, and individual columns. It is intentionally stricter than comparing reciprocal compact to reciprocal repeat alone.

The route remains live only if multiple endpoints show positive interaction in cheap7, cheap6(no GlobalPIQA), and cheap5(no GlobalPIQA/Reading), with at least four positive column interactions and without dominance by copied-token or GlobalPIQA/Reading movements. If repeat topology gains as much as compact topology, or the effect disappears without volatile columns, the result is recurrence/copy redistribution rather than reciprocal semantic learning.

## If a future short H100 screen is authorized

Do not jump to 100M. The minimal screen should use all four arms, same GPT2 architecture/neutral tokenizer/seed/batch size and selected checkpoints. Batch size 256 is preferred because all four cells have the same 223 steps/epoch at batch 256. A 20M two-epoch screen would cost four short runs and can be evaluated at `chck_10M` and `chck_20M` before any extension.

Candidate commands, only if authorized after DeBERTa/evidence:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=<gpu> python -B experiments/archive/frontier_consolidation/scripts/causal_gpt_trainer.py \
  --pool experiments/archive/frontier_consolidation/data/causal_topology_2x2_scaffold/causal_topology2x2_<ARM>_10M.jsonl \
  --tokenizer experiments/archive/frontier_consolidation/data/causal_transfer_scaffold/neutral_tokenizer \
  --run-dir experiments/archive/frontier_consolidation/training/runs/topology2x2_<ARM>_seed43022_20M \
  --gpu 0 --seed 43022 --total-words 20000000 --checkpoint-interval 10000000 --batch-size 256
```

Then evaluate each run with `selected_causal_checkpoint_eval.py` on `chck_10M chck_20M`, and compare with `compare_causal_topology_2x2.py`.

## Pending DeBERTa trajectory work

The two selected DeBERTa common-grid evaluations remain incomplete:

- protected scale1.75 seed43022 reference common 70M–100M grid, output `data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M`, skips cached 80M/100M.
- scale1.25 seed43022 dense common 70M–100M grid, output `data/selected_trajectory_eval_scale1p25_seed43022_dense_common2M`.

Dry launch plan for the remaining third trajectory succeeded (`data/seed43122_common_grid_launch_plan/selected_eval_plan.json`):

- run dir: `training/runs/adapter128_scale1p75_seed43122_dense100M`
- output plan dir: `data/seed43122_common_grid_launch_plan`
- endpoints: `chck_70M, chck_72M, ..., chck_100M`
- missing endpoints: none

When a GPU frees, launch scale1.75 seed43122 selected common-grid scoring to `data/selected_trajectory_eval_scale1p75_seed43122_dense_common2M`. After all three trajectories exist, run `scripts/selected_mlm_integrity_check.py` on each output and then `scripts/compare_selected_mlm_trajectories.py`.

Important: seed/scale trajectories test peak timing/width and adapter-scale interference, not reciprocal semantics. They must not be used as evidence that the causal topology scaffold should train. Reciprocal training should depend on triangle scores or a direct topology interaction rationale, not on the DeBERTa seed/scale grid.
