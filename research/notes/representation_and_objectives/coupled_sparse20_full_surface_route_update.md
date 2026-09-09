# aligned shuffled full ewok residual slices — coupled sparse20 full-surface route update

## Evidence now resolved

The full ewok coupled turnover full-EWoK jobs completed and were aggregated with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B experiments/archive/representation_and_objectives/scripts/full_ewok_coupled_turnover.py --aggregate
```

Artifacts:

- `experiments/archive/representation_and_objectives/data/full_ewok_coupled_turnover/full_ewok_coupled_turnover_summary.json`
- `research/notes/representation_and_objectives/full_ewok_coupled_turnover.md`
- residual slices: `experiments/archive/representation_and_objectives/data/aligned_shuffled_full_ewok_residual/aligned_shuffled_full_ewok_residual_slices.json`
- residual slice note: `research/notes/representation_and_objectives/aligned_shuffled_full_ewok_residual_slices.md`

This is no-training inference on existing checkpoints over all 7,618 official EWoK rows using the validated fw ewok interaction reader four-cell pseudo-likelihood reader.

## Main full-surface result

Target full-surface summaries:

| target | n | accuracy | stable failures | positive interaction | mean interaction | median interaction |
|---|---:|---:|---:|---:|---:|---:|
| `mlm_only_20M` | 7618 | 0.4992 | 2614 | 3736 | -0.0086 | -0.0087 |
| `coupled_aligned_20M` | 7618 | 0.4982 | 2366 | 3620 | -0.0232 | -0.0055 |
| `coupled_shuffled_20M` | 7618 | 0.5018 | 2237 | 3784 | -0.0097 | -0.0006 |

Against `mlm_only_20M`:

- `coupled_aligned_20M`: repaired 1,891 baseline-wrong rows but broke 1,899 baseline-correct rows; net accuracy -0.105 points. Mean all-row interaction delta -0.0147 and median -0.0030. Stable failures fall by 248, but this is offset by almost equal row turnover and by weaker margins on baseline-correct rows.
- `coupled_shuffled_20M`: repaired 1,846 baseline-wrong rows and broke 1,826 baseline-correct rows; net accuracy +0.263 points. Mean all-row interaction delta -0.0011 and median +0.0111. Stable failures fall by 377.

Aligned versus shuffled on all EWoK rows:

- aligned-correct/shuffled-wrong: 1,649.
- shuffled-correct/aligned-wrong: 1,677.
- both-correct: 2,146; both-wrong: 2,146.
- aligned-minus-shuffled stable failures: +129, meaning aligned has more stable failures than shuffled.
- aligned-minus-shuffled interaction mean -0.0136, median -0.0038, positive fraction 0.4888.

This is stronger than the earlier fixed 1,471-row result: it shows that the apparent hard-row repair is not a net learned EWoK interaction improvement. It mostly compresses extreme baseline margins toward a near-zero surface and turns over which rows are correct.

## Residual slices

A separate existing-CSV script `experiments/archive/representation_and_objectives/scripts/aligned_shuffled_full_ewok_residual_slices.py` confirms no large aligned-specific slice remains:

- Domain material-dynamics strongly favors shuffled: aligned accuracy 0.4286 vs shuffled 0.5156, aligned has 123 more stable failures and interaction mean -0.1594 relative to shuffled.
- ContextDiff material also strongly favors shuffled: aligned accuracy 0.4345 vs shuffled 0.5095, aligned has 119 more stable failures and interaction mean -0.1534.
- TargetDiff concept swap favors shuffled: aligned has 44 fewer correct rows and 142 more stable failures than shuffled.
- Variable-swap rows are nearly tied: ContextDiff variable swap has aligned +15 correct but +13 stable failures and interaction mean +0.0011; TargetDiff variable swap has aligned +16 correct, -13 stable failures, interaction mean +0.0086.
- Small aligned-favorable domains such as physical-interactions, spatial-relations, and physical-dynamics are not clean: they have many aligned-only and shuffled-only row swaps, with weak or mixed interaction medians.

## Relation to earlier analysis adapter hook and broad score

earlier analysis already showed:

- matched shuffled reproduced almost all fixed-stable-row repair: true correspondence accounted for only about 4.47% of the fixed-subset stable-failure reduction.
- disabling live adapters did not remove hard-surface movement and often strengthened it; live adapters mildly harmed EWoK stable-failure repair.
- inference-disabled broad cheap7 collapsed further: disabled aligned 37.2071 and disabled shuffled 37.3679 versus `mlm_only_20M` 39.7864.

Together with full ewok coupled turnover/182, the coupled sparse20 route should no longer be treated as a promising source-rewrite alignment mechanism. The movement is a correspondence-free trajectory disturbance that reduces extreme four-cell margins and changes row identities. It damages broad language competence and does not produce a net all-EWoK interaction gain.

## Scientific consequence for the active goal

The practical reproduced endpoint `chck_82M` remains protected and untouched as the above-frontier candidate. The deeper research goal remains open: find a generalizable, compliant, data-efficient learning principle that forms context-sensitive alternatives while preserving the compact-view/scale1.75 broad substrate.

Do not launch endpoint-scale or 82M-scale dual-view work from sparse20. Do not weaken the same auxiliary until broad collapse disappears, because the full-surface evidence says the key movement is not the intended correspondence mechanism.

The next route should be constructed around a different representation-forming object. A promising low-cost direction is not training yet: build a fixed legal-pool contrastive margin measurement from corpus-derived coherent-vs-perturbed frames, then test whether it ranks known late trajectories without reading official labels during construction. role switch packet screen synthesis scout found ample heuristic frame yield, but those frames still require validation into clean transformed pairs before any objective is designed.
