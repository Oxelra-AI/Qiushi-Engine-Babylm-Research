# packed targetselect static load and readout patch packed static label-load profile

JSON: `experiments/archive/representation_and_objectives/data/packed_static_label_load_profile/packed_static_label_load_profile.json`

This profile was run while the earlier analysis 100M arms were still managed asynchronously; it does not read their outputs.

## Key static totals

- stream rows used: 647400
- word exposure used: 100000000
- total candidate BPE pieces: 143915480
- source-absent-content candidate BPE pieces: 319580
- selected copied-content candidate BPE pieces: 322370
- selected minus absent candidate pieces: 2790

## Batch-level denominator signal

```json
{
  "n": 2529,
  "mean": 0.99998071,
  "median": 0.99998221,
  "p01": 0.99902897,
  "p05": 0.99930948,
  "p10": 0.9994716,
  "p25": 0.99970176,
  "p75": 1.00024757,
  "p90": 1.00050394,
  "p95": 1.00065415,
  "p99": 1.00095741,
  "min": 0.99856297,
  "max": 1.00182084
}
```

A ratio near 1 says the per-batch mean-loss denominator difference is tiny. Endpoint interpretation still depends on the completed arm comparison and local denoising readout.
