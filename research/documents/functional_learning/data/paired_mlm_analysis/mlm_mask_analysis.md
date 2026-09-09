# credit allocation binding pilot MLM mask analysis for contrastive entity-binding packets
## Purpose
Quantify how often 15% whole-word masking (WWM) actually presents binding-sensitive
supervision: the answer is masked while the relational evidence (which entity was
updated, what the new state is) remains visible.

## Per-type aggregates

| type | n | P(ans masked) | P(ev visible) | P(useful) | P(useful no shortcut) | mean ans groups | mean ev groups |
|---|---:|---:|---:|---:|---:|---:|---:|
| UPDATE | 8 | 0.2615 | 0.4972 | 0.1278 | 0.0338 | 1.9 | 4.4 |
| RETAIN | 8 | 0.2610 | 0.4975 | 0.1277 | 0.0346 | 1.9 | 4.4 |

## Per-pair joint supervision

| pair_id | UPDATE P(useful) | RETAIN P(useful) | P(both) | Expected both/10ep |
|---|---:|---:|---:|---:|
| tp_001 | 0.0924 | 0.0918 | 0.0085 | 0.08 |
| tp_002 | 0.1039 | 0.1043 | 0.0108 | 0.11 |
| tp_003 | 0.1432 | 0.1431 | 0.0205 | 0.21 |
| tp_004 | 0.1437 | 0.1434 | 0.0206 | 0.21 |
| tp_005 | 0.1453 | 0.1449 | 0.0211 | 0.21 |
| tp_006 | 0.1064 | 0.1060 | 0.0113 | 0.11 |
| tp_007 | 0.1440 | 0.1431 | 0.0206 | 0.21 |
| tp_008 | 0.1434 | 0.1446 | 0.0207 | 0.21 |

## Per-packet detail

### tp_001 — UPDATE
- Answer: `green` (2 occurrences, 1 word groups at primary location)
- Critical evidence: 3 word groups
- Total word groups: 32
- **P(answer masked)**: 0.1502 (analytic: 0.1500)
- **P(evidence visible)**: 0.6163 (analytic: 0.6141)
- **P(useful supervision)**: 0.0924 (analytic: 0.0921)
- **P(shortcut|useful)**: 0.8560
- **P(useful, no shortcut)**: 0.0133

### tp_001 — RETAIN
- Answer: `red` (2 occurrences, 1 word groups at primary location)
- Critical evidence: 3 word groups
- Total word groups: 32
- **P(answer masked)**: 0.1494 (analytic: 0.1500)
- **P(evidence visible)**: 0.6145 (analytic: 0.6141)
- **P(useful supervision)**: 0.0918 (analytic: 0.0921)
- **P(shortcut|useful)**: 0.8500
- **P(useful, no shortcut)**: 0.0138

### tp_002 — UPDATE
- Answer: `digital tablets` (2 occurrences, 2 word groups at primary location)
- Critical evidence: 6 word groups
- Total word groups: 41
- **P(answer masked)**: 0.2764 (analytic: 0.2775)
- **P(evidence visible)**: 0.3752 (analytic: 0.3771)
- **P(useful supervision)**: 0.1039 (analytic: 0.1047)
- **P(shortcut|useful)**: 0.7185
- **P(useful, no shortcut)**: 0.0292

### tp_002 — RETAIN
- Answer: `leather-bound journals` (2 occurrences, 2 word groups at primary location)
- Critical evidence: 6 word groups
- Total word groups: 41
- **P(answer masked)**: 0.2769 (analytic: 0.2775)
- **P(evidence visible)**: 0.3771 (analytic: 0.3771)
- **P(useful supervision)**: 0.1043 (analytic: 0.1047)
- **P(shortcut|useful)**: 0.7031
- **P(useful, no shortcut)**: 0.0310

### tp_003 — UPDATE
- Answer: `white truck` (2 occurrences, 2 word groups at primary location)
- Critical evidence: 4 word groups
- Total word groups: 37
- **P(answer masked)**: 0.2772 (analytic: 0.2775)
- **P(evidence visible)**: 0.5240 (analytic: 0.5220)
- **P(useful supervision)**: 0.1432 (analytic: 0.1449)
- **P(shortcut|useful)**: 0.7309
- **P(useful, no shortcut)**: 0.0385

### tp_003 — RETAIN
- Answer: `silver sedan` (2 occurrences, 2 word groups at primary location)
- Critical evidence: 4 word groups
- Total word groups: 37
- **P(answer masked)**: 0.2736 (analytic: 0.2775)
- **P(evidence visible)**: 0.5230 (analytic: 0.5220)
- **P(useful supervision)**: 0.1431 (analytic: 0.1449)
- **P(shortcut|useful)**: 0.7099
- **P(useful, no shortcut)**: 0.0415

### tp_004 — UPDATE
- Answer: `converted warehouse` (2 occurrences, 2 word groups at primary location)
- Critical evidence: 4 word groups
- Total word groups: 42
- **P(answer masked)**: 0.2784 (analytic: 0.2775)
- **P(evidence visible)**: 0.5228 (analytic: 0.5220)
- **P(useful supervision)**: 0.1437 (analytic: 0.1449)
- **P(shortcut|useful)**: 0.7168
- **P(useful, no shortcut)**: 0.0407

### tp_004 — RETAIN
- Answer: `small apartment` (2 occurrences, 2 word groups at primary location)
- Critical evidence: 4 word groups
- Total word groups: 42
- **P(answer masked)**: 0.2768 (analytic: 0.2775)
- **P(evidence visible)**: 0.5235 (analytic: 0.5220)
- **P(useful supervision)**: 0.1434 (analytic: 0.1449)
- **P(shortcut|useful)**: 0.7241
- **P(useful, no shortcut)**: 0.0396

### tp_005 — UPDATE
- Answer: `spotted rabbit` (2 occurrences, 2 word groups at primary location)
- Critical evidence: 4 word groups
- Total word groups: 41
- **P(answer masked)**: 0.2784 (analytic: 0.2775)
- **P(evidence visible)**: 0.5204 (analytic: 0.5220)
- **P(useful supervision)**: 0.1453 (analytic: 0.1449)
- **P(shortcut|useful)**: 0.7294
- **P(useful, no shortcut)**: 0.0393

### tp_005 — RETAIN
- Answer: `tabby cat` (2 occurrences, 2 word groups at primary location)
- Critical evidence: 4 word groups
- Total word groups: 41
- **P(answer masked)**: 0.2769 (analytic: 0.2775)
- **P(evidence visible)**: 0.5202 (analytic: 0.5220)
- **P(useful supervision)**: 0.1449 (analytic: 0.1449)
- **P(shortcut|useful)**: 0.7102
- **P(useful, no shortcut)**: 0.0420

### tp_006 — UPDATE
- Answer: `mushroom risotto` (2 occurrences, 2 word groups at primary location)
- Critical evidence: 6 word groups
- Total word groups: 43
- **P(answer masked)**: 0.2785 (analytic: 0.2775)
- **P(evidence visible)**: 0.3773 (analytic: 0.3771)
- **P(useful supervision)**: 0.1064 (analytic: 0.1047)
- **P(shortcut|useful)**: 0.7222
- **P(useful, no shortcut)**: 0.0296

### tp_006 — RETAIN
- Answer: `spicy tofu` (2 occurrences, 2 word groups at primary location)
- Critical evidence: 6 word groups
- Total word groups: 43
- **P(answer masked)**: 0.2785 (analytic: 0.2775)
- **P(evidence visible)**: 0.3781 (analytic: 0.3771)
- **P(useful supervision)**: 0.1060 (analytic: 0.1047)
- **P(shortcut|useful)**: 0.7172
- **P(useful, no shortcut)**: 0.0300

### tp_007 — UPDATE
- Answer: `electric violin` (2 occurrences, 2 word groups at primary location)
- Critical evidence: 4 word groups
- Total word groups: 38
- **P(answer masked)**: 0.2762 (analytic: 0.2775)
- **P(evidence visible)**: 0.5230 (analytic: 0.5220)
- **P(useful supervision)**: 0.1440 (analytic: 0.1449)
- **P(shortcut|useful)**: 0.7233
- **P(useful, no shortcut)**: 0.0398

### tp_007 — RETAIN
- Answer: `acoustic guitar` (2 occurrences, 2 word groups at primary location)
- Critical evidence: 4 word groups
- Total word groups: 38
- **P(answer masked)**: 0.2758 (analytic: 0.2775)
- **P(evidence visible)**: 0.5203 (analytic: 0.5220)
- **P(useful supervision)**: 0.1431 (analytic: 0.1449)
- **P(shortcut|useful)**: 0.7253
- **P(useful, no shortcut)**: 0.0393

### tp_008 — UPDATE
- Answer: `denim vest` (2 occurrences, 2 word groups at primary location)
- Critical evidence: 4 word groups
- Total word groups: 44
- **P(answer masked)**: 0.2769 (analytic: 0.2775)
- **P(evidence visible)**: 0.5187 (analytic: 0.5220)
- **P(useful supervision)**: 0.1434 (analytic: 0.1449)
- **P(shortcut|useful)**: 0.7220
- **P(useful, no shortcut)**: 0.0399

### tp_008 — RETAIN
- Answer: `wool coat` (2 occurrences, 2 word groups at primary location)
- Critical evidence: 4 word groups
- Total word groups: 44
- **P(answer masked)**: 0.2805 (analytic: 0.2775)
- **P(evidence visible)**: 0.5230 (analytic: 0.5220)
- **P(useful supervision)**: 0.1446 (analytic: 0.1449)
- **P(shortcut|useful)**: 0.7262
- **P(useful, no shortcut)**: 0.0396

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
