# factorial construction report full BPE/supervised-mass audit (all 3,006 changed rows)

JSON: `experiments/archive/representation_and_objectives/data/full_bpe_audit/full_bpe_audit.json`

## Totals per arm (one pass / changed block only)

| arm | BPE pieces | candidate tokens | word groups |
|---|---:|---:|---:|
| HS | 607,262 | 607,262 | 423,170 |
| LS | 583,262 | 583,262 | 423,366 |
| HD | 607,783 | 607,783 | 423,449 |
| LD | 583,467 | 583,467 | 423,488 |

## Deltas (all changed rows)

| contrast | BPE delta | word group delta | BPE % | WG % |
|---|---:|---:|---:|---:|
| HS_minus_LS | +24,000 | -196 | +3.952% | -0.046% |
| HD_minus_LD | +24,316 | -39 | +4.004% | -0.009% |
| HS_minus_HD | -521 | -279 | -0.086% | -0.066% |
| LS_minus_LD | -205 | -122 | -0.034% | -0.029% |
| interaction | -316 | -157 | -0.052% | -0.037% |

## Per-row interaction BPE distribution
- mean: -0.1051, median: 0.0, stdev: 8.0831
- range: [-38, 32]

## Expected WWM target mass over 100M
- HS: ~634,755 targets
- LS: ~635,049 targets
- HD: ~635,174 targets
- LD: ~635,232 targets
- interaction target difference: ~236 targets over 100M
