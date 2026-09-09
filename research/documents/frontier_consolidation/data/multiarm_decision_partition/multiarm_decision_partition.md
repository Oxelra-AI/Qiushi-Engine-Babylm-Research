# coherent86 multiarm mechanism reading multi-arm decision partition

Status: **COMPLETE**

Saved official-compatible prediction payloads are compared at common item level. This is a scientific reading of endpoint diversity, not a training or submission procedure.

## Payload scores

| arm | cheap7 | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading | payload |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| anchor82 | 43.95944987645173 | 68.49128403651986 | 62.9378112562002 | 50.05545332553276 | 28.314041930298774 | 52.19117509443596 | 37.57766990291262 | 8.148713589261902 | `experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/staged_full_eval/per_target/scale1p75_chck82_independent.json` |
| ordinary86 | 43.770714285714284 | 68.47 | 62.68 | 50.14 | 28.58 | 52.29 | 36.135 | 8.1 | `experiments/archive/representation_and_objectives/data/scale1p75_chck86_cheap7_eval/per_target/scale1p75_chck86_cheap7.json` |
| shuffled86 | 44.01285714285714 | 69.23 | 59.81 | 51.44 | 26.77 | 52.65 | 39.565 | 8.625 | `experiments/archive/frontier_consolidation/data/frozen82_tail4M_shuffled_eval/per_target/frozen82_tail4M_shuffled.json` |
| coherent86 | 44.10642857142857 | 68.52 | 63.65 | 49.91 | 28.44 | 51.99 | 38.065 | 8.17 | `experiments/archive/frontier_consolidation/data/fastpath4M_coherent_eval/per_target/fastpath4M_coherent.json` |
| spanbreak86 | 43.121428571428574 | 68.18 | 62.36 | 49.37 | 24.24 | 52.04 | 37.565 | 8.095 | `experiments/archive/frontier_consolidation/data/fastpath4M_spanbreak_eval/per_target/fastpath4M_spanbreak.json` |

## Aggregate decision partition

- coherent unique gains vs anchor and all other arms: **366**
- coherent unique losses vs anchor while all other arms preserve: **425**
- coherent losses vs anchor that ordinary86 preserves: **1454**
- anchor-tie hard majority discrete mean: **49.878647** (Δ vs anchor -0.049259; Δ vs coherent -0.217697)

## By-column compact view

| column | n common | coherent unique gains | coherent unique losses | coherent losses ordinary preserves | majority score | majority Δ vs anchor | majority Δ vs coherent | top anchor-wrong patterns | top anchor-correct damage patterns |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| BLiMP | 59875 | 74 | 102 | 331 | 68.5799 | +0.0886 | +0.0506 | none:14080; shuffled86:2235; spanbreak86:594; shuffled86+spanbreak86:510; ordinary86:299 | none:36095; shuffled86:2052; spanbreak86:849; shuffled86+spanbreak86:426; ordinary86:346 |
| Supplement | 5218 | 8 | 10 | 34 | 62.9119 | -0.0259 | -0.7340 | none:1099; shuffled86:99; spanbreak86:35; shuffled86+spanbreak86:26; ordinary86:24 | none:3538; shuffled86:140; spanbreak86:58; shuffled86+spanbreak86:36; ordinary86+shuffled86+spanbreak86:16 |
| EWoK | 7618 | 23 | 10 | 58 | 49.7416 | -0.3139 | -0.1645 | none:2737; shuffled86:452; spanbreak86:176; shuffled86+spanbreak86:115; ordinary86:64 | none:2774; shuffled86:430; spanbreak86:199; shuffled86+spanbreak86:93; ordinary86:58 |
| Entity | 6780 | 14 | 5 | 38 | 28.2339 | -0.0802 | -0.2061 | none:4177; spanbreak86:370; shuffled86:112; ordinary86:56; shuffled86+spanbreak86:50 | none:1096; spanbreak86:433; shuffled86+spanbreak86:105; shuffled86:95; ordinary86+shuffled86+spanbreak86:28 |
| COMPS | 91028 | 246 | 298 | 992 | 52.2270 | +0.0358 | +0.2332 | none:27822; shuffled86:7007; spanbreak86:2013; shuffled86+spanbreak86:1724; ordinary86:919 | none:32842; shuffled86:6582; spanbreak86:2237; shuffled86+spanbreak86:1630; ordinary86:1031 |
| GlobalPIQA | 203 | 1 | 0 | 1 | 37.5777 | +0.0000 | -0.4854 | none:108; shuffled86:9; shuffled86+spanbreak86:2; ordinary86:2; ordinary86+shuffled86+spanbreak86:1 | none:60; shuffled86:5; ordinary86:4; ordinary86+shuffled86:2; ordinary86+shuffled86+spanbreak86:1 |

## Scientific reading

- Across the six discrete columns, coherent86 has 366 unique gains and 425 unique losses relative to the anchor when compared with ordinary86, shuffled86, and spanbreak86.
- Coherent losses that ordinary86 preserves total 1454; these are the decisions a retention-oriented route would need to protect without using benchmark labels.
- Anchor-tie hard prediction majority has discrete mean 49.878647 versus anchor 49.927906, coherent 50.096344, and ordinary86 49.716742. This is analysis of endpoint diversity, not a submission procedure or a training signal.

JSON: `experiments/archive/frontier_consolidation/data/multiarm_decision_partition/multiarm_decision_partition.json`
