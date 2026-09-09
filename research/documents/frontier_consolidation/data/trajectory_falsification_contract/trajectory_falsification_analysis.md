# generalization experiment contract trajectory falsification analysis

This file treats the dense runs as local contrasts. They can support or weaken specific seed/scale accounts, but they do not by themselves prove a transferable data-efficient learning law.

## scale1p75_seed43022_reference
- best: chck_82M at 82M, cheap7=43.9594
- last: chck_100M cheap7=43.5432; last-minus-best=-0.4163
- band within 0.200 of best spans 3.0M words: chck_80M, chck_82M, chck_83M
- local peak excess over adjacent-neighbor mean: +0.2309

## Fixed interpretation

- Similar scale1.75 peak timing supports seed robustness only if the late decline and family profile also resemble the reference.
- Lower scale supports amplitude-controlled interference only if it shifts later and/or broadens the high-score band without becoming a different family tradeoff.
- Semantic-overlap by surface-diversity interaction remains untested here and needs a matched semantic-view contrast under a different architecture or scale.

JSON: `experiments/archive/frontier_consolidation/data/trajectory_falsification_contract/trajectory_falsification_analysis.json`
