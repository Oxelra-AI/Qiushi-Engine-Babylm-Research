# post40k route assets post-legal40k route assets

This CPU-only asset was built while the two legal-40k trainings were still managed asynchronously. It does not poll them, train a model, or evaluate a model.

## Relation-cue density in the allowed 10M pool
- Pool SHA: `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`; rows `64740`, words `10000000`.
- Relation-cue words under the relation frame retention audit coarse lexicon: `1091528` (10.9153% of whitespace words).
- Mean/median/p90 row relation-cue fraction: `0.1090` / `0.1064` / `0.1625`.
- Top relation categories by word count: `{'negation_modality': 222287, 'causal_temporal': 205624, 'mental_state_social': 178087, 'quantity_measure': 164013, 'spatial_state': 112124, 'comparison_order': 78578, 'social_relation': 48018, 'physical_interaction': 41553, 'material_property': 22815, 'physical_dynamics': 19532}`.

## Tokenizer interface
| tokenizer | vocab | raw tok/word | visible tok/word | over256 rows | visible relation-group frac | visible relation-token frac | unk |
|---|---:|---:|---:|---:|---:|---:|---:|
| legal16k | 16384 | 1.4669 | 1.4299 | 15143 | 0.1096 | 0.0926 | 0 |
| legal40k | 40000 | 1.3943 | 1.3706 | 11407 | 0.1094 | 0.0960 | 0 |
| legal40k_minfreq25 | 29529 | 1.4139 | 1.3871 | 12335 | 0.1094 | 0.0950 | 0 |
| legal40k_minfreq50 | 19609 | 1.4484 | 1.4153 | 14096 | 0.1095 | 0.0934 | 0 |

## Relation-boost masking budget
A relation-focused WWM route can increase relation-cue group probability while lowering non-relation probability so the expected visible selected word-group count stays at 0.15 per group. This keeps total group-level target exposure comparable while testing whether Supplement/EWoK need denser relation/predicate prediction pressure.

Selected budget rows for legal40k:
| boost | p_relation | p_nonrelation | relation selected group frac | target token multiplier |
|---:|---:|---:|---:|---:|
| 2.0 | 0.300 | 0.132 | 0.2188 | 0.9850 |
| 3.0 | 0.450 | 0.113 | 0.3282 | 0.9700 |

## Architecture parameter counts
| architecture | tokenizer | params | embedding | non-embedding | hidden/layers/heads/intermediate |
|---|---|---:|---:|---:|---|
| 8x480_current_legal16k | legal16k | 34467424 | 7864320 | 26603104 | 480/8/8/1920 |
| 8x480_current_legal40k | legal40k | 45826720 | 19200000 | 26626720 | 480/8/8/1920 |
| 8x480_minfreq50 | legal40k_minfreq50 | 36018649 | 9412320 | 26606329 | 480/8/8/1920 |
| 12x384_leader_shape_legal16k | legal16k | 29329792 | 6291456 | 23038336 | 384/12/12/1280 |
| 12x384_leader_shape_legal40k | legal40k | 38421952 | 15360000 | 23061952 | 384/12/12/1280 |
| 12x384_leader_shape_minfreq50 | legal40k_minfreq50 | 30571417 | 7529856 | 23041561 | 384/12/12/1280 |
| 12x384_ffn4_legal40k_existing_trainer | legal40k | 40784320 | 15360000 | 25424320 | 384/12/12/1536 |
| 12x384_leader_pos1024_legal40k | legal40k | 38618560 | 15360000 | 23258560 | 384/12/12/1280 |

## Scientific use
- If legal40k improves Supplement/EWoK while preserving GlobalPIQA/Entity/COMPS, this asset helps characterize how much of the gain comes with low-support representation breadth and reduced target-token burden.
- If legal40k fails or trades away compact-view gains, the relation-boost budget and leader-shape parameter table are ready for a distinct objective or depth route; support-floored tokenizers remain interpretation assets, not automatic next GPU work.
- The full legal40k official vectors remain decisive; no route is selected by this CPU asset alone.

JSON: `experiments/archive/representation_and_objectives/data/post40k_route_assets/post40k_route_assets.json`
