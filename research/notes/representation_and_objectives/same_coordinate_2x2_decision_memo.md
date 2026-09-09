# ewok item flip audit — same-coordinate 2×2 decision memo

## Scientific question

The seed43122 compact_view_reinvest endpoint is below the visible Strict-Small leader, but that alone does not tell whether compact-view reinvestment failed. The decisive comparison is treatment effect within the same pretraining seed, on a shared metric coordinate.

## Completed 2×2

Using the same leaderboard arithmetic (mean of nine columns), pristine 7,618-row official EWoK, official-min-context AoA, and SuperGLUE primary-metric convention:

| cell | Overall | margin vs 41.8 |
|---|---:|---:|
| clean_43022 | 41.347633 | -0.452367 |
| clean_43122 | 40.767775 | -1.032225 |
| reinvest_43022 | 42.033135 | +0.233135 |
| reinvest_43122 | 41.248240 | -0.551760 |

Treatment effects:

- seed43022: reinvest - clean = **+0.685502 Overall**
- seed43122: reinvest - clean = **+0.480465 Overall**
- two-seed average treatment effect = **+0.582983 Overall**
- treatment×seed interaction = **-0.205037 Overall**

Thus seed43122 below-leader is not a verdict against the compact-view reinvestment principle. The treatment is positive in both tested seeds. The absolute seed43122 weakness mostly reflects the backbone recipe already being weaker at seed43122; compact-view adds a smaller but still positive increment.

## Coordinate caveat

This is same-coordinate at the metric-definition level, not a single fresh end-to-end clean-collation reproduction. The reinvest cells are from official collator coordinate seed43022/changed block overlap ancestry current official pristine official collation. The clean cells reuse COMPACT_EXPERIENCE compact density subtask delta zero-shot/Reading columns for BLiMP, Supplement, Entity, COMPS, GlobalPIQA, and Reading, while replacing EWoK with companion analysis earlier analysis pristine 7,618-row EWoK and recomputing clean SuperGLUE from saved per-task `results.txt` files using the official primary-metric convention. This is appropriate for the current decision, but before using the clean 2×2 as final public evidence, verify clean tokenizer/model/evaluator identities and, if needed, run a pristine-collator reconstruction for the clean cells.

## What is stable and what is unstable

Treatment effects by column:

- seed43022: EWoK +2.873497, Entity +1.985741, SuperGLUE +1.170431, Supplement +0.435764, Reading +0.481572, GlobalPIQA -0.998641, BLiMP +0.032323, COMPS +0.188828.
- seed43122: SuperGLUE +1.145582, Reading +1.660017, Entity +1.025339, Supplement +0.442353, EWoK +0.411374, GlobalPIQA +0.515922, BLiMP -0.355600, COMPS -0.520806.

Treatment×seed interaction is concentrated in:

- EWoK: **-2.462123**
- Entity: -0.960402
- COMPS: -0.709634
- BLiMP: -0.387923
- SuperGLUE: -0.024849
- GlobalPIQA: +1.514563
- Reading: +1.178445
- Supplement: +0.006589

This supports a narrowed route: the compact-view route should not be replaced; the research problem is stabilizing relation-sensitive knowledge formation, especially EWoK and its Entity/COMPS neighbors.

## EWoK domain localization

The strongest negative official-domain interactions are descriptive but important:

- material-dynamics: DiD -17.532 (TE43022 +9.870, TE43122 -7.662, n=770)
- physical-dynamics: DiD -10.833 (TE43022 +7.500, TE43122 -3.333, n=120)
- spatial-relations: DiD -8.367 (TE43022 +2.653, TE43122 -5.714, n=490)
- physical-interactions: DiD -5.935 (TE43022 +4.317, TE43122 -1.619, n=556)
- social-relations: DiD -2.132 (TE43022 +0.065, TE43122 -2.067, n=1548)

The strongest positive interactions are material-properties, social-interactions, social-properties, and agent-properties. This means the route should not use a generic semantic-quality explanation. The instability is more directional/dynamic/relational than simply factual or lexical.

## independent_review-supported next research program before any new 100M run

Independent review checked the arithmetic and supported the conclusions, with the caveat above, and proposed the following low-cost ordering of tests:

1. **EWoK margin and trajectory audit on existing checkpoints.** Re-score or parse all four cells, and several late reinvest checkpoints if available, with per-item log-likelihood margins. Determine whether seed43122 errors are near-zero flips, stable confidently wrong relations, or late-checkpoint wandering. Focus first on material/physical/spatial/direct-context domains.
2. **Separate downstream adaptation randomness from pretrained-weight differences.** Zero-shot EWoK is the immediate target; SuperGLUE/Entity/COMPS may require repeated finetuning seeds only after the zero-shot relation picture is clearer.
3. **Cheap consolidation probes using existing weights.** Test same-run late-checkpoint tail averaging or checkpoint choice first if the margin/trajectory audit shows late instability. Cross-seed weight interpolation is only a probe; output ensembling can show complementary relational evidence without parameter alignment assumptions.
4. **Build, not train, relation-preserving compact-row counterfactuals.** Annotate compact pairs for directional predicates, material nouns/properties, state transitions, role order, inverse-pole balance, source/rewrite distance, and assertion/negation/modality. Compare accepted vs unused rows before constructing any new 423.5k-word alternative.

The adjacency-broken control should wait. It tests whether source-own compact adjacency causes average treatment gain; it does not explain why the treatment effect is smaller mainly in EWoK at seed43122. It becomes useful after relation-stability analysis clarifies whether pair adjacency is still the most plausible mechanism.

## Immediate route implication

Protect seed43022 in parallel for submission packaging because it is a real above-leader official-coordinate endpoint with compliance ledger and sparse-overlap evidence. Continue research on stabilizing relation formation/consolidation using existing artifacts first. Do not launch a new expensive training branch until the margin/trajectory and relation-selection analyses say which uncertainty the run would settle.

Primary JSON evidence:

- `experiments/archive/representation_and_objectives/data/consistent_official_2x2/consistent_official_2x2.json`
- `experiments/archive/representation_and_objectives/data/relation_instability_synthesis/relation_instability_synthesis.json`

independent_review evidence:

- `data/external/independent_review01_generator1_integration.md`
- `data/external/independent_review01_verifier1_integration.md`

## Prediction-file schema check

A quick inspection of current official EWoK `predictions.json` files for clean43022, clean43122, and reinvest43022 showed domain-keyed prediction records with `id` and selected `pred` strings, not stored per-alternative log-likelihood margins. The reinvest43122 path is recorded in `experiments/archive/representation_and_objectives/data/seed43122_official_coordinate_repairs/seed43122_official_coordinate_repairs_summary.json` as `experiments/archive/representation_and_objectives/data/official_ewok_reeval_seed43122/official_outputs/EWoK/chck_100M/official_ewok_seed43122/zero_shot/mlm/ewok/ewok_filtered/predictions.json`. Therefore a true margin audit will likely need to rerun the zero-shot scorer or a custom forward-pass script that records both alternative scores; existing prediction files can support flip/count analysis but not confidence/margin analysis.
