# research synthesis correction: Corrected binding interpretation and practical route

## Correction to the binding coordinate

The earlier analysis independence-excess coordinate (`joint - A·B/n`) is structurally wrong
for the recombination design. Each pair shares identical context and differs only in
the queried entity. A non-gating model gives both halves the same answer and gets
exactly one right per pair. The no-gating null is **joint = 0**, not `A·B/n`.

Under the corrected coordinate, answer_clean is a positive signal:

| arm | joint/200 | gated% | both_wrong | still rising? |
|---|---:|---:|---:|---|
| base (epoch 0) | 6 | 3.0% | 0 | - |
| answer_clean (ep20) | 44 | 22.0% | 0 | YES (35→44 from ep15) |
| uniform_wwm (ep20) | 7 | 3.5% | 1 | no (flat) |
| answer_corrupt (ep20) | 30 | 15.0% | 1 | no (plateaued ep15) |

Concentrated answer credit raised gating ~7x from base. Uniform WWM installed
nothing (state update intervention prestate reproduced at adapter scale). Corrupted support reached 15%
then plateaued — visible update evidence matters but is not sufficient alone.

both_wrong ≈ 0 for all arms: every joint success is a genuine flip, not noise.

## Position audit

~80% of held pairs have unchanged entity first in context. If position were the 
mechanism, gating should vanish when the position is reversed.

answer_clean at epoch 20:
- unchanged_first (n=160): 38/160 = 23.8% gated
- updated_first (n=29): 5/29 = 17.2% gated
- ambiguous (n=11): 1/11 = 9.1% gated

Gating appears in BOTH position classes. uniform_wwm gets 0/29 in updated_first.
Position enrichment exists but does not explain the gating signal. The 5/29
reversed-position successes require entity identity, not position.

## Seed43122 cheap7

Two-seed dose comparison:

| column | seed43022 base/d21/d25 | seed43122 base/d21/d25 |
|---|---|---|
| BLiMP | 68.62/67.61/66.00 | 67.07/67.22/67.16 |
| Entity | 27.46/27.12/26.83 | 25.77/27.45/27.99 |
| EWoK | 49.08/50.14/52.52 | 52.27/52.08/51.35 |
| Reading | 8.32/7.87/7.30 | 8.31/7.77/8.39 |

Seed43022 monotone patterns (BLiMP down, EWoK up, Reading down) do NOT replicate
at seed43122. Entity rises at both seeds. The monotone pattern is seed-specific,
not a general dose law.

## Practical route

The corrected binding result supports the practical candidate:
1. answer_clean raises gating ~7x, both-wrong = 0, position doesn't explain it
2. Still rising at epoch 20 → likely undertrained
3. coherent86 mechanism is identical: frozen slow + 995,584 private parameters
4. Direct test: same balanced answer-credit on chck_82M private branch

Build and validate the chck_82M binding candidate (engineering). If the binding
measurement shows genuine entity-conditioned improvement, compare against coherent86.

Before any transfer or general claim:
- Run no-context scorer on held queries 
- Run operation-preserving transfer checks
- Check that improvement is not dominated by 160/200 position-favorable pairs
