# cf smoke counterfactual history probe

Evidence JSON: `experiments/archive/initial_model_studies/data/cf_smoke.json`

Pairs: 50; mean gap 32.7 words; 36 unique target words; 43 unique swap words

## Results

| model | L2 cos | L4 cos | L6 cos | L8 cos | L2 disc | L4 disc | L6 disc | L8 disc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wwm43_40M | 0.83024 | 0.83692 | 0.84602 | 0.86736 | 63.3 | 66.7 | 66.7 | 63.3 |

## Interpretation

If mean cosine ≈ 1.0 and discrimination ≈ 50%: model ignores history entity identity → official remention probe gain was from topic/lexical cues, not entity-state tracking.
If cosine < 0.99 and discrimination > 60%: genuine entity-identity signal propagates across sentences → intermediate-state supervision has a real target.
