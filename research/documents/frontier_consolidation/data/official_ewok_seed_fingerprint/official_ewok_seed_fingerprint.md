# earlier analysis — official EWoK seed-spread fingerprint for compact_view_reinvest

This compares seed43022 and seed43122 on the same current official 7618-row EWoK coordinate. It does not require the still-locked full seed43122 vector.

## Overall official EWoK seed spread
- official macro-domain seed43022: 53.537
- official macro-domain seed43122: 51.892
- official macro-domain seed43122 - seed43022: -1.645
- micro item-weighted seed43022: 51.720
- micro item-weighted seed43122: 50.499
- micro agreement rate: 66.99%
- micro seed43022-only correct: 1304; seed43122-only correct: 1211

## Domain deltas (seed43122 - seed43022)
- material-dynamics: -11.56 (n=770, 43022=58.18, 43122=46.62, agree=49.7)
- spatial-relations: -4.29 (n=490, 43022=47.55, 43122=43.27, agree=72.0)
- physical-interactions: -3.24 (n=556, 43022=51.26, 43122=48.02, agree=62.9)
- material-properties: -1.76 (n=170, 43022=56.47, 43122=54.71, agree=64.1)
- physical-relations: -1.34 (n=818, 43022=51.10, 43122=49.76, agree=69.6)
- social-properties: -0.91 (n=328, 43022=55.49, 43122=54.57, agree=64.9)
- physical-dynamics: -0.83 (n=120, 43022=60.00, 43122=59.17, agree=69.2)
- social-relations: +0.13 (n=1548, 43022=49.74, 43122=49.87, agree=62.8)
- agent-properties: +1.76 (n=2210, 43022=49.95, 43122=51.72, agree=75.9)
- quantitative-properties: +1.91 (n=314, 43022=55.41, 43122=57.32, agree=66.2)
- social-interactions: +2.04 (n=294, 43022=53.74, 43122=55.78, agree=63.3)

## ContextDiff deltas
- variable_swap: -26.67 (n=30, 43022=63.33, 43122=36.67)
- material: -10.00 (n=840, 43022=57.86, 43122=47.86)
- active-passive: -3.33 (n=30, 43022=56.67, 43122=53.33)
- other: -2.04 (n=490, 43022=54.90, 43122=52.86)
- variable swap: -1.21 (n=2564, 43022=50.82, 43122=49.61)
- game: +0.00 (n=20, 43022=50.00, 43122=50.00)
- antonym: +0.71 (n=3374, 43022=49.85, 43122=50.56)
- negation: +3.16 (n=190, 43022=57.89, 43122=61.05)
- number: +13.75 (n=80, 43022=55.00, 43122=68.75)

## TargetDiff deltas
- concept swap: -1.77 (n=4746, 43022=52.61, 43122=50.84)
- variable swap: -0.31 (n=2872, 43022=50.24, 43122=49.93)

## EWoK treatment-effect read
- TE43022 (reinvest-clean EWoK): +3.347
- TE43122 (reinvest-clean EWoK): +1.462
- EWoK DiD: -1.885

The official-coordinate EWoK repair shows seed43122 still gains over its matched clean-Qwen EWoK baseline, but much less than seed43022; material-dynamics and spatial-relations dominate the remaining seed spread.

Machine-readable output: `experiments/archive/frontier_consolidation/data/official_ewok_seed_fingerprint/official_ewok_seed_fingerprint.json`
