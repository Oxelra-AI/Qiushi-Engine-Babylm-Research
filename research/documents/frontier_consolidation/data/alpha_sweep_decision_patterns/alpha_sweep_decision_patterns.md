# coherent86 multiarm mechanism reading alpha-sweep decision patterns

Status: **COMPLETE**

Bits are ordered `[alpha0, alpha0.5, alpha0.75, alpha1]`, where alpha0 is the protected anchor.

## Aggregate

- total common discrete items: `170722`
- adjacent changes: `{'a0->a0p5': 3141, 'a0p5->a0p75': 1613, 'a0p75->a1': 1625}`
- monotonic activation gain candidates: `3116`; nonmonotonic gain candidates: `9`
- monotonic damage candidates: `3231`; nonmonotonic damage candidates: `7`

Top aggregate patterns:
- `1111`: 95240
- `0000`: 69119
- `1000`: 1575
- `0111`: 1553
- `1100`: 837
- `1110`: 819
- `0001`: 792
- `0011`: 771
- `0110`: 6
- `1001`: 5
- `0010`: 2
- `0100`: 1

## By column

| column | n | adjacent changes | all stable correct 1111 | all stable wrong 0000 | gain all scaled 0111 | gain only 0.5 0100 | gain 0.5/0.75 not 1 0110 | lost only at 1 1110 | lost at 0.75/1 1100 | lost all scaled 1000 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 59875 | `{'a0->a0p5': 737, 'a0p5->a0p75': 375, 'a0p75->a1': 390}` | 40204 | 18174 | 383 | 1 | 1 | 188 | 193 | 350 |
| Supplement | 5218 | `{'a0p75->a1': 24, 'a0p5->a0p75': 22, 'a0->a0p5': 42}` | 3819 | 1311 | 15 | 0 | 0 | 14 | 10 | 27 |
| EWoK | 7618 | `{'a0->a0p5': 156, 'a0p5->a0p75': 74, 'a0p75->a1': 68}` | 3654 | 3667 | 83 | 0 | 0 | 37 | 40 | 72 |
| Entity | 6780 | `{'a0p5->a0p75': 33, 'a0->a0p5': 71, 'a0p75->a1': 37}` | 1825 | 4815 | 33 | 0 | 0 | 12 | 16 | 38 |
| COMPS | 91028 | `{'a0p75->a1': 1105, 'a0->a0p5': 2132, 'a0p5->a0p75': 1108}` | 45664 | 41028 | 1036 | 0 | 5 | 567 | 577 | 1088 |
| GlobalPIQA | 203 | `{'a0->a0p5': 3, 'a0p75->a1': 1, 'a0p5->a0p75': 1}` | 74 | 124 | 3 | 0 | 0 | 1 | 1 | 0 |

## Scientific reading

- Across 170722 common discrete items, adjacent alpha changes are {'a0->a0p5': 3141, 'a0p5->a0p75': 1613, 'a0p75->a1': 1625}; most decisions are stable, but the private residual has thousands of threshold-sensitive decisions.
- Anchor-wrong gains include 3116 monotonic activation candidates and 9 nonmonotonic candidates; anchor-correct damage includes 3231 monotonic larger-alpha damage candidates and 7 nonmonotonic damage candidates.
- Alpha0.5/0.75 improving cheap7 while reducing changed items is consistent with amplitude moderation, but binary item patterns remain mixed rather than a clean capability threshold.

JSON: `experiments/archive/frontier_consolidation/data/alpha_sweep_decision_patterns/alpha_sweep_decision_patterns.json`
