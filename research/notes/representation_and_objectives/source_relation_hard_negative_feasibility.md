# source relation hard negative feasibility — source-grounded relation hard-negative feasibility

This CPU-only work used the existing compact-view-reinvest 10M corpus and already trained checkpoints. It did not train, run the BabyLM suite, or touch the managed SGCR tasks.

## Source generator

The relation-alternative inventory was frozen before reading any new evaluation rows. It excludes the high-polysemy pairs `in/out`, `on/off`, `left/right`, `like/dislike`, `know/ignore`, and `can/cannot`. Sentences are kept only when exactly one safe relation term appears and the opposite term is absent.

Rows scanned `819`, words scanned `114478`, sentence fragments seen `6314`, kept candidates `165`.

Kept candidates by group:

| group | kept | raw matches |
|---|---:|---:|
| access_state | 15 | 31 |
| affordance_block | 15 | 75 |
| containment_membership | 15 | 95 |
| identity_state | 15 | 102 |
| possibility | 15 | 30 |
| quantity_comparison | 15 | 395 |
| spatial_containment | 15 | 35 |
| spatial_vertical | 15 | 177 |
| temporal_order | 15 | 209 |
| thermal_state | 15 | 28 |
| truth_state | 15 | 16 |

## Existing-model margins

Margin is log p(original relation word | masked source sentence) minus log p(replacement word | the same masked sentence), with only equal-token-span cases scored. Weak or negative margins mean the future objective would have a real prediction target; large positive margins mean the signal is probably redundant.

| model | scoreable | positive | nonpositive | near zero | weak <=1 | median | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| legal40_depth_12x384_43022 | 158/165 | 0.7784810126582279 | 0.22151898734177214 | 0.05063291139240506 | 0.34177215189873417 | 2.194729804992676 | 2.159100797734683 |
| legal40_8x480_43022 | 158/165 | 0.810126582278481 | 0.189873417721519 | 0.08227848101265822 | 0.37341772151898733 | 1.884347915649414 | 2.018228507588936 |

Group-level margin summaries are in the JSON.

## Scientific reading

A future source-grounded hard-negative relation objective should use these examples only after human or teacher-assisted quality reading of sampled edits, because regex precision is uneven even after strict filters. The single-input hard-negative form is the lower-accounting form: encode the original corpus sentence once, mask the edited span, and compare the original label against the frozen opposite label. Full corrupted-sequence ranking would be a different and more expensive exposure design.

JSON: `experiments/archive/representation_and_objectives/data/source_relation_hard_negative_feasibility/source_relation_hard_negative_feasibility.json`

Candidates: `experiments/archive/representation_and_objectives/data/source_relation_hard_negative_feasibility/source_relation_hard_negative_candidates.csv`

Margins: `experiments/archive/representation_and_objectives/data/source_relation_hard_negative_feasibility/source_relation_hard_negative_model_margins.csv`
