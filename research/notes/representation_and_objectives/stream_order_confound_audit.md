# mask budget and pad contract — Stream-order confound audit for experience-utilization trainer

This CPU-only audit compares the current 10M pool used by the experience utilization trainer verified trainer with the materialized 100M stream used by the legal40k/depth baselines.

## Result

10M pool: 64740 rows, 10000000 words, SHA `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`.
100M stream: 647400 rows, 100000000 words, SHA `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`.
Every 64,740-row block of the 100M stream is exactly the same raw-line multiset as the 10M pool: `True`; every block has exactly 10M words: `True`.

However, the row order is not the canonical 10M-file order repeated by experience utilization trainer verified. Fixed-position matches are near zero and adjacent canonical edges are nearly absent. Launching the experience utilization trainer verified pool-order trainer unchanged would therefore test visibility plus a substantial data-order change relative to the existing legal40k/depth baselines.

| epoch | rows | words | multiset match | same-position fraction | adjacent-forward fraction | mean abs displacement | position corr | first 12 base positions |
|---:|---:|---:|:---:|---:|---:|---:|---:|---|
| 0 | 64740 | 10000000 | True | 0.000015 | 0.000000 | 21581.3 | 0.0006 | [10913, 55330, 59795, 26130, 16586, 18894, 39304, 13279, 5486, 62354, 42746, 36649] |
| 1 | 64740 | 10000000 | True | 0.000000 | 0.000000 | 21534.4 | 0.0020 | [15887, 7166, 32650, 41992, 23715, 47099, 38929, 53758, 14258, 46616, 45489, 2960] |
| 2 | 64740 | 10000000 | True | 0.000015 | 0.000015 | 21589.3 | 0.0002 | [10599, 50722, 13653, 40158, 38800, 41180, 21250, 15678, 19447, 46603, 10201, 40093] |
| 3 | 64740 | 10000000 | True | 0.000031 | 0.000031 | 21587.7 | -0.0000 | [15495, 19428, 52830, 24451, 52199, 46115, 17203, 52490, 8328, 53123, 12368, 40084] |
| 4 | 64740 | 10000000 | True | 0.000015 | 0.000015 | 21604.7 | -0.0023 | [51205, 35939, 15373, 26291, 4623, 28613, 31467, 44359, 13436, 37506, 52803, 55507] |
| 5 | 64740 | 10000000 | True | 0.000031 | 0.000000 | 21640.8 | -0.0060 | [46350, 53264, 59835, 25427, 18683, 59429, 53691, 52833, 8231, 26964, 24039, 31468] |
| 6 | 64740 | 10000000 | True | 0.000000 | 0.000031 | 21565.4 | 0.0018 | [12633, 14078, 55869, 14679, 33898, 6543, 21988, 46194, 16253, 24760, 44074, 18863] |
| 7 | 64740 | 10000000 | True | 0.000015 | 0.000000 | 21549.6 | 0.0026 | [41885, 9533, 54856, 48023, 17500, 18732, 433, 43820, 27223, 27134, 24782, 53585] |
| 8 | 64740 | 10000000 | True | 0.000015 | 0.000015 | 21459.0 | 0.0094 | [12657, 13654, 40844, 61570, 6438, 40010, 29270, 32814, 44056, 20095, 48239, 11130] |
| 9 | 64740 | 10000000 | True | 0.000062 | 0.000000 | 21691.4 | -0.0088 | [27209, 31564, 20687, 23468, 7681, 64322, 36667, 9163, 60977, 38370, 43706, 12573] |

## Consequence for the route

The first expensive experience-utilization run should preserve the materialized 100M stream order while replacing prefix-hidden row slices by word-boundary chunks. A stream-order trainer can still keep the chunk stream preflight invariants (10M charged words per exposure epoch, 253 updates per epoch, 2,530 total updates, full-effective-batch WWM, masked-token weighting) but removes this avoidable data-order confound.

JSON: `experiments/archive/representation_and_objectives/data/stream_order_confound_audit/stream_order_confound_audit.json`
