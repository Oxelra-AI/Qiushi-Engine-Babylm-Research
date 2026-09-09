# cf smoke counterfactual history probe

Evidence JSON: `experiments/archive/initial_model_studies/data/counterfactual_history_probe.json`

Pairs: 500; mean gap 41.3 words; 293 unique target words; 427 unique swap words

## Results

| model | L2 cos | L4 cos | L6 cos | L8 cos | L2 disc | L4 disc | L6 disc | L8 disc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wwm42_40M | 0.81080 | 0.81635 | 0.82177 | 0.84156 | 63.7 | 61.0 | 63.0 | 62.3 |
| wwm42_100M | 0.81522 | 0.81034 | 0.82254 | 0.83479 | 60.3 | 65.7 | 64.7 | 65.3 |
| wwm43_40M | 0.80679 | 0.81464 | 0.82117 | 0.84218 | 59.3 | 58.7 | 57.3 | 56.0 |
| wwm43_100M | 0.81507 | 0.81556 | 0.82642 | 0.84112 | 61.0 | 63.3 | 63.0 | 65.0 |
| amlm42_40M | 0.81817 | 0.82550 | 0.82728 | 0.84799 | 64.0 | 62.0 | 60.7 | 59.0 |
| amlm42_100M | 0.80946 | 0.82120 | 0.82144 | 0.83789 | 68.0 | 65.7 | 64.3 | 65.7 |

## Interpretation

If mean cosine ≈ 1.0 and discrimination ≈ 50%: model ignores history entity identity → official remention probe gain was from topic/lexical cues, not entity-state tracking.
If cosine < 0.99 and discrimination > 60%: genuine entity-identity signal propagates across sentences → intermediate-state supervision has a real target.
