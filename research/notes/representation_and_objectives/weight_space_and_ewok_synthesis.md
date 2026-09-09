# fw weight space sweep — FW same-initialization weight-space branch-consolidation test

## Question
The FW compact, interleaved-breadth, and row-block-breadth endpoints share initialization
(seed 43/43022/43023), architecture (DeBERTa-v2 8x480, 34,467,424 params), shared legal
16k tokenizer (SHA `e70d167f...`), and the COMPACT_EXPERIENCE full-batch training coordinate; they differ
only in the FineWeb companion data view. The strategist asked whether these aligned branches
encode complementary behavior in compatible weight-space directions, so that a single
linear-combination checkpoint could keep compact broad capability (Supplement/Entity) while
inheriting breadth relation movement (EWoK / GlobalPIQA hard-parallel). This is scientifically
different from the failed decision framework and clean collation plan cross-seed soup because these branches are same-initialization.

## Geometry (fw weight space sweep sweep manifest)
- ||interleaved − compact|| / ||compact|| = 0.7822; ||rowblock − compact|| / ||compact|| = 0.7892.
- cosine(interleaved−compact, rowblock−compact) = 0.5375 globally, and 0.51–0.57 per layer group.
- The two breadth deltas are large (about 78% of the compact norm) and share a substantial common
  direction (positive cosine ~0.54) rather than being orthogonal capability axes.

## Fixed sweep readout (no training, no SuperGLUE/AoA, no full collation)
Endpoints: compact broad-pair(Supp,Ent)=43.61, relation-pair(EWoK,GP)=44.44; interleaved 41.01 / 46.48;
row-block 41.79 / 43.78; row-block hard52 margin 1.4333 (lowest of the endpoints).

| model | Supplement | Entity | EWoK | GP-parallel | GP-nonpar | GP | broad pair | relation pair | hard52 acc | hard52 margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| compact (endpoint) | 58.86 | 28.36 | 50.25 | 24.27 | 53.0 | 38.635 | 43.61 | 44.44 | 3.85 | 1.717 |
| interleaved (endpoint) | 56.42 | 25.60 | 51.85 | 26.21 | 56.0 | 41.105 | 41.01 | 46.48 | 3.85 | 1.660 |
| rowblock (endpoint) | 59.69 | 23.88 | 50.50 | 29.13 | 45.0 | 37.065 | 41.79 | 43.78 | 5.77 | 1.433 |
| ci 0.75/0.25 | 57.44 | 25.71 | 50.59 | 27.18 | 48.0 | 37.59 | 41.58 | 44.09 | 0.00 | 1.521 |
| ci 0.50/0.50 | 56.66 | 20.65 | 51.40 | 21.36 | 50.0 | 35.68 | 38.66 | 43.54 | 1.92 | 1.501 |
| ci 0.25/0.75 | 56.64 | 22.11 | 50.04 | 27.18 | 50.0 | 38.59 | 39.38 | 44.32 | 0.00 | 1.560 |
| cr 0.75/0.25 | 57.53 | 23.03 | (not run) | 23.30 | 48.0 | — | — | — | 0.00 | 1.467 |
| cir 0.50/0.25/0.25 | 55.25 | 18.94 | (not run) | 19.42 | 51.0 | — | — | — | 3.85 | 1.384 |

(EWoK aggregate was read for the three compact→interleaved mixtures: ci 0.75/0.25 = 50.59,
ci 0.50/0.50 = 51.40, ci 0.25/0.75 = 50.04. It was not run for the two row-block mixtures because
their broad sentinels already collapsed.)

## What this establishes
- No mixture keeps compact-like Entity. Even the mildest interleaved injection (25%) drops Entity
  28.36 → 25.71 and Supplement 58.86 → 57.44; adding any row-block component drops Entity further
  (23.03, 18.94). Entity behaves as the most interpolation-fragile compact capability.
- No mixture reaches a broad-pair above the compact endpoint while also lifting the relation-pair
  above the compact endpoint; the mixtures sit on or below the endpoint tradeoff frontier.
- Relation movement from interpolation is small and non-monotone: EWoK stays 50.0–51.4, GlobalPIQA
  aggregate never exceeds the interleaved endpoint, and the hard52 deep-rank margin never drops below
  the row-block endpoint (1.433). The one mixture that lowers hard52 margin to 1.384 (cir) does so with
  Entity 18.94 and GlobalPIQA_parallel 19.42 — a broad and parallel-accuracy collapse, not a genuine
  combined gain.
- The positive-cosine, large-magnitude branch deltas explain this: compact→interleaved and
  compact→row-block move mostly along a shared subspace, so linear combination traces the existing
  broad↔relation tradeoff rather than reaching an off-frontier joint optimum.

## Conclusion for the route
Same-initialization linear weight-space consolidation of the FW data-view branches does not
produce a checkpoint that unites compact broad capability with breadth relation movement. The
branch-and-consolidate-by-averaging idea is closed for this endpoint family. This does not
reopen cross-seed soups (decision framework and clean collation plan) and does not motivate a new 100M FW allocation branch. The
next evidence-producing route should change the relational objective or representation directly,
using the transition structure match 12,418-word compact-contained matched transition/control subset only as a
mechanism probe to distinguish source/style from transition structure — not as a SOTA training route.

## Files
- mixtures + geometry: `experiments/archive/representation_and_objectives/data/fw_weight_space_sweep/weight_space_sweep_manifest.json`
- compact→interleaved screen: `experiments/archive/representation_and_objectives/data/weight_sweep_first_screen/weight_sweep_first_screen_summary.json`
- row-block screen: `experiments/archive/representation_and_objectives/data/weight_sweep_rowblock_screen/weight_sweep_first_screen_summary.json`
- EWoK four-cell (compact + row-block, from s112_t16_tool1): `experiments/archive/representation_and_objectives/data/fw_ewok_interaction_reader/fw_ewok_interaction_reader_summary.json`
