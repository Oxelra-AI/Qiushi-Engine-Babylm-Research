# clean d component ablation — Exact-coherent86 continuation as the next practical test

## Why this route replaces further replay tuning

causal intervention/026 showed that raw carrier-error credit on a fresh 82M→86M replay is not a trustworthy improvement over the inherited `coherent86` reference. The corrected deterministic carrier-residual run trails its replay standard by only `0.023` cheap7 points, while both replay arms trail exact `coherent86_alpha0.75` much more strongly and share the same Supplement deficit. The scientifically important separation is therefore:

- the raw weight rule may be too crude, but
- the larger score loss also reflects failure to reproduce the inherited successful 82M→86M training state under the gradient-accumulated replay conditions.

The next high-value practical question is not another local alpha tweak. It is whether the actual useful private correction already present in `coherent86_alpha0.75` can absorb additional legal experience and move beyond the 42.12 reference.

## Experimental question

Starting from the exact scored `coherent86_alpha0.75` endpoint:

1. Does ordinary continuation on the remaining compact-view-reinvest stream improve, preserve, or damage the common cheap7 score?
2. Does deterministic carrier-error weighting behave differently when applied to an already useful private correction rather than a freshly initialized private path?
3. Are gains/losses concentrated in the same columns as causal interface trajectory (EWoK/Entity/COMPS vs Supplement/GlobalPIQA), or does retaining the successful state change the tradeoff?
4. Where along the checkpoint ladder does the private correction peak under the remaining legal budget?

This distinguishes a limitation of the raw credit rule from a limitation of the replay trajectory. It also creates a stronger platform for later relation-retargeted data if the state-update stream yields a column-specific improvement.

## Legal and architectural accounting

Parent checkpoint:

- `models/frontier`
- inherited common screen in causal interface trajectory: cheap7 `44.5643`
- SHA256 in causal interface trajectory note: `e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8`

Continuation stream:

- `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl`
- next row after the 86M coherent tail: `skip_rows = 556791`
- remaining strict-small budget from earlier analysis/153 accounting: `100000000 - 86005295 = 13994705` words
- loading from row 556791 with this cap consumes exactly `13994705` words through row `647399`, reaching exactly `100000000` total words.

Trainable set and objective:

- train only existing `.private_adapter.*` parameters (`995584` trainable params);
- keep slow path frozen;
- keep private scale at the exact scored alpha, `0.75`, during continuation and saving;
- use the cumulative earlier analysis schedule continuation: `schedule_total=455`, `schedule_offset=101`, so the remaining 354 macro-updates occupy the tail of the original cosine schedule;
- retain deterministic private-on vs private-off neutral KL as in earlier analysis;
- save checkpoints every 1M tail words plus final.

## Arms launched/planned

Script: `experiments/archive/functional_learning/scripts/coherent86_continuation_trainer.py`.

The script first passed a 2-update smoke test for the standard arm, loading the exact coherent86 alpha0.75 parent, using row 556791, consuming legal words, and writing valid checkpoints/metrics.

Full arms:

- `standard`: ordinary WWM MLM continuation from the exact parent.
- `carrier_residual`: deterministic weight `1 - p(correct)` with the private adapters temporarily disabled for carrier scoring, eval-mode carrier, and RNG isolation. This is still a crude confidence-based rule; the point is to test whether its failure in causal interface trajectory came from the fresh replay trajectory rather than the rule alone.

## Interpretation to preserve

If ordinary continuation improves the parent, the immediate practical result is that the inherited coherent86 peak was not the end of useful legal learning; the checkpoint ladder then matters for selecting a new candidate. If ordinary continuation damages the score, that supports a narrow-tail/overtraining account and focuses attention on selective or data-composition interventions.

If carrier_residual beats ordinary continuation from the same parent and does not reproduce the Supplement/GlobalPIQA tradeoff, then confidence-weighted residual credit becomes a conditional lever whose earlier failure was replay-dependent. If it again trades EWoK/Entity/COMPS against Supplement/GlobalPIQA or trails standard, then raw uncertainty weighting should remain de-emphasized and the research should move to relation-resolvable credit or state-update data composition.

No full official evaluation is justified until a checkpoint survives the common cheap7 screen against exact `coherent86_alpha0.75` or shows a strong mechanistically interpretable column movement with a realistic path to preserve lost columns.
