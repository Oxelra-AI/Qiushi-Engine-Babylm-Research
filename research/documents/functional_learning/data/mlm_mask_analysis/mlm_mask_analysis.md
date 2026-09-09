# credit allocation binding pilot MLM mask analysis for contrastive entity-binding packets
## Purpose
Quantify how often 15% whole-word masking (WWM) actually presents binding-sensitive
supervision: the answer is masked while the relational evidence (which entity was
updated, what the new state is) remains visible.

## Per-type aggregates

| type | n | P(ans masked) | P(ev visible) | P(useful) | P(useful no shortcut) | mean ans groups | mean ev groups |
|---|---:|---:|---:|---:|---:|---:|---:|
| UPDATE | 3 | 0.2359 | 0.4487 | 0.1060 | 0.0257 | 1.7 | 5.0 |
| RETAIN | 3 | 0.2349 | 0.4679 | 0.1111 | 0.0278 | 1.7 | 4.7 |

## Per-pair joint supervision

| pair_id | UPDATE P(useful) | RETAIN P(useful) | P(both) | Expected both/10ep |
|---|---:|---:|---:|---:|
| hand_001 | 0.0669 | 0.0664 | 0.0044 | 0.04 |
| hand_002 | 0.1053 | 0.1244 | 0.0131 | 0.13 |
| hand_003 | 0.1458 | 0.1425 | 0.0208 | 0.21 |

## Per-packet detail

### hand_001 — UPDATE
- Answer: `green` (2 occurrences, 1 word groups at primary location)
- Critical evidence: 5 word groups
- Total word groups: 32
- **P(answer masked)**: 0.1502 (analytic: 0.1500)
- **P(evidence visible)**: 0.4446 (analytic: 0.4437)
- **P(useful supervision)**: 0.0669 (analytic: 0.0666)
- **P(shortcut|useful)**: 0.8601
- **P(useful, no shortcut)**: 0.0094

### hand_001 — RETAIN
- Answer: `red` (2 occurrences, 1 word groups at primary location)
- Critical evidence: 5 word groups
- Total word groups: 32
- **P(answer masked)**: 0.1494 (analytic: 0.1500)
- **P(evidence visible)**: 0.4437 (analytic: 0.4437)
- **P(useful supervision)**: 0.0664 (analytic: 0.0666)
- **P(shortcut|useful)**: 0.8521
- **P(useful, no shortcut)**: 0.0098

### hand_002 — UPDATE
- Answer: `digital tablets` (2 occurrences, 2 word groups at primary location)
- Critical evidence: 6 word groups
- Total word groups: 60
- **P(answer masked)**: 0.2790 (analytic: 0.2775)
- **P(evidence visible)**: 0.3773 (analytic: 0.3771)
- **P(useful supervision)**: 0.1053 (analytic: 0.1047)
- **P(shortcut|useful)**: 0.7256
- **P(useful, no shortcut)**: 0.0289

### hand_002 — RETAIN
- Answer: `leather-bound journals` (2 occurrences, 2 word groups at primary location)
- Critical evidence: 5 word groups
- Total word groups: 62
- **P(answer masked)**: 0.2808 (analytic: 0.2775)
- **P(evidence visible)**: 0.4403 (analytic: 0.4437)
- **P(useful supervision)**: 0.1244 (analytic: 0.1231)
- **P(shortcut|useful)**: 0.7206
- **P(useful, no shortcut)**: 0.0347

### hand_003 — UPDATE
- Answer: `bronze compass` (2 occurrences, 2 word groups at primary location)
- Critical evidence: 4 word groups
- Total word groups: 31
- **P(answer masked)**: 0.2784 (analytic: 0.2775)
- **P(evidence visible)**: 0.5243 (analytic: 0.5220)
- **P(useful supervision)**: 0.1458 (analytic: 0.1449)
- **P(shortcut|useful)**: 0.7346
- **P(useful, no shortcut)**: 0.0387

### hand_003 — RETAIN
- Answer: `silver watch` (2 occurrences, 2 word groups at primary location)
- Critical evidence: 4 word groups
- Total word groups: 31
- **P(answer masked)**: 0.2745 (analytic: 0.2775)
- **P(evidence visible)**: 0.5198 (analytic: 0.5220)
- **P(useful supervision)**: 0.1425 (analytic: 0.1449)
- **P(shortcut|useful)**: 0.7284
- **P(useful, no shortcut)**: 0.0387

## Interpretation

With standard 15% WWM, the probability of useful relational supervision per packet
per epoch is the product P(answer masked) × P(evidence visible). For a typical
1-word-group answer and ~5-10 evidence groups:
- P(answer masked) ≈ 0.15
- P(evidence visible) ≈ 0.85^n_evidence ≈ 0.44-0.72
- P(useful) ≈ 0.07-0.11

Over 10 training epochs, each packet provides ~0.7-1.1 useful supervision events.
For paired contrasts, P(both UPDATE and RETAIN useful in same epoch) ≈ 0.005-0.012,
meaning the learner rarely sees both sides of the same pair in the same pass.
This quantifies why volume matters: the model must accumulate binding signal
across different pairs with different entities, not from repeated same-pair exposure.

If useful_no_shortcut is much lower than useful_supervision, the packet design has
a leakage problem: the answer text appears at another visible location.
