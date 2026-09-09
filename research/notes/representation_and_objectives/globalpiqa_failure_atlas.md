# compact density subtask delta GlobalPIQA failure atlas

This CPU-only pass links the finished compact-density fast GlobalPIQA predictions to the local evaluation examples and to the selected compact source-view pairs. It was done while the compact reinvest full eval summary full evaluation and seed43122 run continued asynchronously, without polling or modifying them.

## Accuracy reconstruction

| arm | parallel | nonparallel | mean |
|---|---:|---:|---:|
| compact_repeat_core | 26.21 | 42.00 | 34.11 |
| compact_view_core | 24.27 | 46.00 | 35.14 |
| compact_view_reinvest | 25.24 | 46.00 | 35.62 |

The reconstructed accuracies match the recorded results: reinvest is 25.24 parallel and 46.00 nonparallel, mean 35.62. This confirms that the weakness is not a parsing artifact.

## Flips between arms

### parallel
- `view_core_minus_repeat_core`: fixed 9, broken 11, net -2, both wrong 67.
- `reinvest_minus_view_core`: fixed 7, broken 6, net 1, both wrong 71.
- `reinvest_minus_repeat_core`: fixed 5, broken 6, net -1, both wrong 71.
- all three compact arms wrong: 66; only reinvest correct: 1.
### nonparallel
- `view_core_minus_repeat_core`: fixed 13, broken 9, net 4, both wrong 45.
- `reinvest_minus_view_core`: fixed 13, broken 13, net 0, both wrong 41.
- `reinvest_minus_repeat_core`: fixed 13, broken 9, net 4, both wrong 45.
- all three compact arms wrong: 38; only reinvest correct: 7.

## Category movement

### parallel: reinvest minus repeat by category
| category | n | repeat | reinvest | delta |
|---|---:|---:|---:|---:|
| time | 11 | 9.09 | 18.18 | +9.09 |
| object_properties_interactions | 60 | 31.67 | 31.67 | +0.00 |
| spatial | 26 | 15.38 | 15.38 | +0.00 |
| counting | 15 | 20.00 | 20.00 | +0.00 |
| affordances | 21 | 28.57 | 23.81 | -4.76 |
| object_properties | 2 | 50.00 | 0.00 | -50.00 |

### nonparallel: reinvest minus repeat by category
| category | n | repeat | reinvest | delta |
|---|---:|---:|---:|---:|
| uncategorized | 100 | 42.00 | 46.00 | +4.00 |

## Source overlap for unresolved examples

- parallel: 66 examples are wrong for repeat, core-view, and reinvest. For these examples, compact-pair lexical overlap is only a rough source-coverage signal, not evidence of learned ability.
  - affordances: n=14, mean max content-Jaccard to any selected compact pair=0.119
  - counting: n=11, mean max content-Jaccard to any selected compact pair=0.137
  - object_properties: n=1, mean max content-Jaccard to any selected compact pair=0.121
  - object_properties_interactions: n=34, mean max content-Jaccard to any selected compact pair=0.120
  - spatial: n=20, mean max content-Jaccard to any selected compact pair=0.135
  - time: n=8, mean max content-Jaccard to any selected compact pair=0.134
- nonparallel: 38 examples are wrong for repeat, core-view, and reinvest. For these examples, compact-pair lexical overlap is only a rough source-coverage signal, not evidence of learned ability.
  - uncategorized: n=38, mean max content-Jaccard to any selected compact pair=0.114

## Mechanistic reading for next work

- Compact views repair many nonparallel practical examples relative to literal repetition, but they do not repair the harder parallel side; reinvest adds a small parallel gain and no nonparallel gain over core-view.
- The remaining gap to the public leader is dominated by practical affordance / causal-procedural GlobalPIQA examples, especially cases where all three compact arms choose the same wrong option. If the full reinvest endpoint misses only narrowly, the next data repair should add compact source-view packets describing physical affordances, everyday tool use, material response, time/counting, and action outcome contrasts, rather than adding more generic encyclopedic statements.
- Any such repair should preserve the successful compact-view properties already measured: short jointly visible source+rewrite packets, high entity/number retention, exact 10M/100M accounting, and no broad loss in Supplement/Entity/EWoK.

JSON: `experiments/archive/representation_and_objectives/data/globalpiqa_failure_atlas/globalpiqa_failure_atlas.json`
