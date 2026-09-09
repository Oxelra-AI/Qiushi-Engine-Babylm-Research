# Endpoint-Selector Validation Is Not a New FineWeb Result

Status: historical selector validation using an existing semantic-view trajectory; projected scores, not new FineWeb evaluations.

The selector was exercised on semantic-view treatment versus packet-local repetition data. Its output labels referred to a different intended comparison, so the validation must not be interpreted as measured FineWeb source-breadth evidence.

| Exposure | Treatment equal7 | Control equal7 | Difference |
|---|---:|---:|---:|
| 10M | 38.4157 | 38.0121 | 0.4036 |
| 20M | 39.5857 | 39.9379 | -0.3521 |
| 30M | 40.6814 | 40.1950 | 0.4864 |
| 40M | 41.5229 | 39.8029 | 1.7200 |
| 50M | 41.0071 | 40.5914 | 0.4157 |
| 60M | 41.7486 | 41.6121 | 0.1364 |
| 70M | 41.2414 | 41.3614 | -0.1200 |
| 80M | 41.8429 | 41.5800 | 0.2629 |
| 90M | 41.7086 | 41.4971 | 0.2114 |
| 100M | 41.6029 | 41.4307 | 0.1721 |

The unrestricted fixture selected 80M and 40M for matched full-evaluation follow-up. At 80M, required SuperGLUE to reach Overall 41.8 with AoA zero was 83.3000; the reference-SuperGLUE projection was only 40.2989. At 40M those figures were 85.5400 and 40.0500. Such projections determine the value of a missing measurement; they are not measurements themselves and do not turn a selector test into a new training contrast.
