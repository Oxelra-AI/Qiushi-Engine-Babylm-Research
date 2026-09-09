# equivariant symmetry repair and macro context equivariant learned pilot

Small mechanism pilot on the repaired active/passive symmetry surface. Scores are against the true assignment; inverted orientation should therefore appear as low changed/mixed accuracy rather than high accuracy.

## Per-run critical metrics

| init | arm | seed | train acc | hh | mixed | state chg | state unchg | pair both | pair chg-only | pair unchg-only |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pretrained | heldheld_only | 27800 | 0.492 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 |

## Cross-seed means

| init | arm | n | mean train | mean hh | mean mixed | std mixed | mean chg | mean unchg | mean pair both |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pretrained | heldheld_only | 1 | 0.492 | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 | 0.000 |

## Reading

Aligned-vs-inverted evidence is mechanism-relevant only if both arms fit training and preserve unaffected facts, while changing mixed held-seen and changed-state orientation in opposite directions. If all arms collapse to the same mixed score, the run is task calibration or residual surface/pretraining bias rather than sparse coordinate identification.
