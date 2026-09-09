# route reopen conditional innovation — conditional innovation in compact source/rewrite pairs

CPU-only; no model update, no official evaluation, no corpus/tokenizer change, no H100 work.

## What was measured
For masked rewrite word groups, compare three contexts: true source span visible, source span masked, and source span replaced by a same-row decoy source from another pair. Groups are split into `copyable` if the normalized rewrite group occurs in the source span and `innovation` otherwise.

- changed rows loaded: `512` of `3005`
- selected targets: `600`; counts: `{'copyable': 300, 'innovation': 300}`; cue counts: `{'copyable:content_or_other': 276, 'innovation:relation_cue': 41, 'innovation:content_or_other': 259, 'copyable:relation_cue': 24}`
- train SHA: `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`
- tokenizer SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`

## Compact table
| checkpoint | group | n | true loss | source help | true over decoy | masked over decoy |
|---|---|---:|---:|---:|---:|---:|
| tokenmean_20M | all | 600 | 5.9075 | 0.2785 | 0.3437 | -0.0652 |
| tokenmean_20M | innovation | 300 | 5.9770 | 0.2175 | 0.2695 | -0.0520 |
| tokenmean_20M | copyable | 300 | 5.8380 | 0.3394 | 0.4178 | -0.0784 |
| tokenmean_20M | innovation:relation_cue | 41 | 4.4161 | 0.0887 | 0.2065 | -0.1177 |
| tokenmean_20M | innovation:content_or_other | 259 | 6.2241 | 0.2379 | 0.2795 | -0.0416 |
| tokenmean_80M | all | 600 | 2.6874 | 2.8932 | 3.2563 | -0.3631 |
| tokenmean_80M | innovation | 300 | 4.4440 | 0.9717 | 1.3171 | -0.3454 |
| tokenmean_80M | copyable | 300 | 0.9308 | 4.8146 | 5.1955 | -0.3808 |
| tokenmean_80M | innovation:relation_cue | 41 | 3.0280 | 0.5826 | 0.7722 | -0.1896 |
| tokenmean_80M | innovation:content_or_other | 259 | 4.6681 | 1.0333 | 1.4034 | -0.3700 |
| tokenmean_100M | all | 600 | 2.6254 | 2.8902 | 3.2721 | -0.3818 |
| tokenmean_100M | innovation | 300 | 4.3550 | 1.0067 | 1.3737 | -0.3670 |
| tokenmean_100M | copyable | 300 | 0.8959 | 4.7738 | 5.1705 | -0.3967 |
| tokenmean_100M | innovation:relation_cue | 41 | 2.9604 | 0.6022 | 0.8544 | -0.2522 |
| tokenmean_100M | innovation:content_or_other | 259 | 4.5758 | 1.0707 | 1.4559 | -0.3852 |
| clean_80M | all | 600 | 3.6586 | 2.4274 | 2.7355 | -0.3080 |
| clean_80M | innovation | 300 | 5.3856 | 0.6206 | 0.8634 | -0.2428 |
| clean_80M | copyable | 300 | 1.9317 | 4.2342 | 4.6075 | -0.3733 |
| clean_80M | innovation:relation_cue | 41 | 4.2681 | 0.1330 | 0.4013 | -0.2683 |
| clean_80M | innovation:content_or_other | 259 | 5.5624 | 0.6978 | 0.9366 | -0.2388 |

## Reading
- source_help = loss(source masked) - loss(true source); positive means the real source span helps predict the masked rewrite group.
- true_over_decoy = loss(same-row decoy source) - loss(true source); positive means the help is source-specific rather than generic same-row pair identity.
- If innovation groups have positive true_over_decoy and remain high-loss at mature checkpoints, conditional innovation is a credible unsaturated route; if only copyable groups show source help, a training objective would mostly teach copying.
- This is not an official score result and should only determine whether to construct a small single-variable objective or whether route choice should shift to contextual computation evidence.

Full JSON: `experiments/archive/frontier_consolidation/data/conditional_innovation_probe/conditional_innovation_probe.json`
Per-target rows: `experiments/archive/frontier_consolidation/data/conditional_innovation_probe/per_target_<checkpoint>.jsonl`
