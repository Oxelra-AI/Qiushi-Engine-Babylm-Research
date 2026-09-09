# depth decision calibration — Depth training health snapshot

CPU-only log analysis. This does not infer downstream score and does not replace official evaluation.

Depth current: step `1608`, words `63585068`, loss `2.7636`, lr `0.000326647`, mask rate `0.1495`.

## Matched 10M comparisons against legal40k 8x480 seed43022
- 10M: depth mean loss(last2M) minus 8x480 = `+0.1009` (ratio `1.0228`).
- 20M: depth mean loss(last2M) minus 8x480 = `+0.0512` (ratio `1.0138`).
- 30M: depth mean loss(last2M) minus 8x480 = `+0.1014` (ratio `1.0303`).
- 40M: depth mean loss(last2M) minus 8x480 = `+0.0691` (ratio `1.0227`).
- 50M: depth mean loss(last2M) minus 8x480 = `+0.0455` (ratio `1.0161`).
- 60M: depth mean loss(last2M) minus 8x480 = `+0.0360` (ratio `1.0133`).

## Anomalies
- No severe execution/optimization anomaly detected in the current log snapshot.

JSON: `experiments/archive/representation_and_objectives/data/depth_training_health_snapshot/depth_training_health_snapshot.json`
