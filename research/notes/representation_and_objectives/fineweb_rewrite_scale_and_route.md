# fineweb rewrite scale and route FineWeb rewrite scale and route

## What the peer high-anchor pilot actually shows

frontier_consolidation generated 1024 high-anchor FineWeb simplification prompts and accepted 952 (0.930). The accepted rows preserve entities and numbers well, but many accepted rows are not a new expression in a strong sense: exact copies among accepted = 136 (0.143); near copies by flag or high string similarity = 523 (0.549); accepted rows with a visibly changed but still source-faithful second view by the current text-similarity rule = 416 (0.437).

The pilot's accepted pair mass is 48,655 words from 26,563 prompt-source words, so a direct scale-up gives 1.832 accepted source+rewrite corpus words per input source word. The more substantive changed-view subset gives 21,322 pair words, or 0.803 per input source word. This means rewrite quality is safe enough to use, but the first prompt is often too conservative to be the main learning mechanism by itself.

## Scale estimates

| source set | source words | estimated accepted pair words | corpus fraction | relative to clean-Qwen 1.6568M pair block | estimated substantive pair words |
|---|---:|---:|---:|---:|---:|
| peer_high_anchor_full | 97,605 | 178,781 | 1.79% | 0.108x | 78,347 |
| peer_high_precision_full | 214,771 | 393,392 | 3.93% | 0.237x | 172,396 |
| peer_medium_repaired_full | 469,887 | 860,684 | 8.61% | 0.519x | 377,176 |
| a01a02_priority_dedup_all_available | 725,008 | 1,327,985 | 13.28% | 0.802x | 581,961 |
| a01_seqsafe96_cached_source_block | 1,753,280 | 3,211,453 | 32.11% | 1.938x | 1,407,350 |

The frontier_consolidation factorized corpus built from the first pilot uses only 16,640 changed-block words (0.166% of the 10M pool), which is 0.010x the inherited clean-Qwen pair block. It is useful as a construction smoke test but should not consume a full 100M-word training slot as a main SOTA probe unless deliberately testing an ultra-small perturbation.

## Inherited score context

The inherited clean-Qwen pair block moved official Overall by 0.353 versus the selected-original-duplicate control with a 1.6568M-word paired block. Its task movement was Ksum=-0.930, Supplement+Reading+SuperGLUE sum=2.005, and GlobalPIQA=-4.460. Thus same-window official-source rewriting is a real sample-efficiency signal but not enough for the current 41.8 target, partly because GlobalPIQA moved down.

The inherited mix25 endpoint remains numerically closer to the leader (Overall 41.4803) but lacks the strict checkpoint ladder needed for a direct submit-ready AoA package. Relative to clean-Qwen it has Overall delta 0.136, EWoK 2.070, GlobalPIQA 1.500, Reading 0.645, and Entity -3.060; it improves some leader-gap columns while harming Entity and SuperGLUE. It is a useful signal about mixture direction, not the next artifact to submit.

## Route implication

The next large H100 work should not be the tiny 16.6k-word peer contrast. It should either (a) use the existing seqsafe96 1.75M-word FineWeb source-only replacement to test broad source breadth at a scale comparable to the clean-Qwen intervention, or (b) first expand/repair the high-anchor/high-precision rewrite pool so the source+rewrite arm reaches a nontrivial unique-word scale. The existing seqsafe96 block is a source-breadth candidate, not a clean rewrite substrate; its cached 96-word chunks include residual web/document artifacts and would need sentence-level or paragraph-level repair before simplification. The cleanest scientific path is still three arms: protected slot control, FineWeb source repetition, and FineWeb source plus accepted faithful rewrite. The generation prompt should be improved toward information-dense faithful compression because the first high-anchor prompt produced many near-copy views.

The pending semantic-view comparison remains decisive for whether source-conservative second views deserve further investment. Once it returns, read the component trajectories before any new H100 launch. If same-source views do not move EWoK/Entity/COMPS/GlobalPIQA, prioritize the scaled source-breadth contrast. If they do move those columns without damaging Supplement/Reading, construct a scaled three-arm FineWeb design using accepted high-anchor/high-precision rewrites plus a source-repeat arm.

JSON: `experiments/archive/representation_and_objectives/data/fineweb_rewrite_scale/fineweb_rewrite_scale_analysis.json`
