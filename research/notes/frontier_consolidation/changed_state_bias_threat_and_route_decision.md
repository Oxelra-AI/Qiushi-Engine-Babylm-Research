# changed state bias threat and route decision changed-state bias threat to the MAX Entity carrier

This note records CPU-only and file-only evidence gathered before spending any H100 on the MAX permuted-companion arm.

## Why this matters

The current MAX-dose story had been: exact duplicate recurrence deteriorates under a fixed word budget, while source-conditioned compact re-expression may preserve or improve entity/state tracking. permuted companion correspondence control built the sharper aligned-versus-permuted control, but that H100 run is only valuable if the Entity carrier is not already explained by a simpler answer tendency.

The serious alternative raised in changed state bias threat and route decision is: compact views may shift the model toward predicting changed/updated state. In official Entity Tracking, this can raise nonzero-operation rows while hurting zero-operation rows. In the counterbalanced binding substrate, it appears as affected-row gains paired with unaffected-row losses, leaving the balanced composite nearly unchanged. Such a signed shift is not cross-surface record re-identification by itself.

## CPU/file evidence produced

### Official Entity predictions split by numops, first-basin MAX view-repeat

Script: `scripts/entity_numops_bias_readout.py`  
Output: `data/entity_numops_bias_readout/entity_numops_bias_summary.md`

First-basin MAX view-minus-repeat from official Entity prediction JSONs is strongly operation-skewed once learning is mature:

| checkpoint | all Entity Δ pp | zero-op Δ pp | nonzero-op Δ pp | nonzero minus zero Δ pp |
|---:|---:|---:|---:|---:|
| 30M | +1.802 | -3.051 | +2.773 | +5.824 |
| 40M | +1.941 | -11.036 | +4.536 | +15.572 |
| 50M | +2.433 | -8.052 | +4.530 | +12.582 |
| 60M | +6.491 | -12.442 | +10.278 | +22.720 |
| 70M | +2.094 | -12.445 | +5.002 | +17.446 |
| 80M | +4.501 | -9.000 | +7.201 | +16.201 |
| 90M | +3.661 | -10.414 | +6.476 | +16.890 |

The late 80/90M mean is: all +4.08 pp, zero-op -9.71 pp, nonzero +6.84 pp. This makes the first-basin official Entity carrier unsafe as standalone evidence for addressable source-view records. It is consistent with an update/operation allocation shift.

The 100M row is missing because CPU Entity scoring for MAX-repeat 100M timed out. Do not infer the 100M split until an Entity prediction file exists.

### Sampled continuous Entity margins split by numops

Script: `scripts/entity_margin_numops_readout.py`  
Output: `data/entity_margin_numops_readout/entity_margin_numops_summary.md`

The 80M sampled margin pilot shows the same operation skew, and the nonzero-minus-zero margin gap grows with restructured dose:

| dose | all margin Δ | zero-op margin Δ | nonzero margin Δ | nonzero minus zero |
|---|---:|---:|---:|---:|
| 1x | +0.8673 | -0.1729 | +1.0754 | +1.2483 |
| 1.82x | +0.2796 | -1.4723 | +0.6300 | +2.1022 |
| MAX | +0.9792 | -2.6744 | +1.7100 | +4.3844 |

This is stronger than a binary-threshold artifact: the graded score direction itself tilts away from zero-operation retention as dose grows.

### Counterbalanced binding affected/unaffected probe at MAX in both basins

Scripts and outputs:

- `scripts/max_binding_change_bias_probe.py`
- `data/max_binding_change_bias_probe/binding_change_bias_summary.md` for chck80
- `data/max_binding_change_bias_probe_late90_100/binding_change_bias_summary.md` for chck90/100

First-basin MAX view-repeat on the earlier analysis binding substrate is a near-perfect signed shift:

| checkpoint | affected Δ pp | unaffected Δ pp | EEBF Δ pp | signed margin shift |
|---:|---:|---:|---:|---:|
| 80M | +23.438 | -20.312 | +1.563 | +1.7320 |
| 90M | +19.792 | -19.792 | +0.000 | +1.6067 |
| 100M | +20.833 | -19.271 | +0.781 | +1.6281 |

Second-basin MAX view-repeat does not show the same direction:

| checkpoint | affected Δ pp | unaffected Δ pp | EEBF Δ pp | signed margin shift |
|---:|---:|---:|---:|---:|
| 80M | +0.521 | +7.812 | +4.167 | -1.2450 |
| 90M | -2.083 | +8.333 | +3.125 | -0.8698 |
| 100M | -1.042 | +6.771 | +2.865 | -0.7645 |

Thus the first basin clearly has an update-bias-shaped movement, while the second basin improves the balanced binding composite mostly through stronger unaffected preservation. This basin dependence means the current evidence cannot yet decide between record-addressability and answer-tendency explanations. The pending second-basin official Entity numops split is essential.

## What this changes

1. The entity leg reconstruction and second basin plan first-basin Entity dose slope should no longer be described as evidence for source-conditioned record addressability by itself. It is now evidence for a fixed-budget operation/state allocation shift that may include an update bias.
2. The MAX permuted-companion H100 run should not be launched merely because the control exists. It becomes valuable only if pending results show an aligned Entity carrier that is not mostly nonzero-op gain plus zero-op loss.
3. Breadth becomes even more important: if same-population sentence breadth matches the nonzero-op Entity gain while preserving zero-op rows better, the compact-view advantage is not a clean correspondence mechanism. If MAX view beats breadth specifically on binding-sensitive rows without zero-op loss, source-conditioned correspondence remains plausible.
4. permuted companion correspondence control bridge is compatible with this refinement: their result supports path-dependent formation of state records under sparse interference, but the semantic selector from role to key remains unresolved. Our BabyLM data result must similarly separate record formation from selector or answer-prior shifts.

## Immediate next reads

- After the second-basin view and repeat evaluations finish, run `scripts/second_basin_entity_ewok_readout.py` and then rerun `scripts/entity_numops_bias_readout.py` including `basin2_max_view` and `basin2_max_repeat`. The important question is whether official Entity in the second basin has zero-op loss and nonzero-op gain or a different profile.
- After breadth scoring finishes, run `scripts/breadth_specificity_readout.py`, then split breadth Entity predictions by numops. If needed, patch `entity_numops_bias_readout.py` or write a triangle reader for V-R, V-B, and B-R by zero/nonzero operations.
- Do not launch `scripts/train_deberta_permuted_companion.py` before these reads. If launched later, report V-R, V-P, and P-R separately by Entity, zero/nonzero operations, and binding affected/unaffected rows, and read EWoK in the same window.
