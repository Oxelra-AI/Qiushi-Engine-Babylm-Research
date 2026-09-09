# fw 70m ewok prediction overlap — FW 70M EWoK prediction overlap

This CPU-only readout uses the completed 70M EWoK prediction files from FW compact and row-block breadth arms. It reproduces official EWoK as the mean of domain accuracies, then locates breadth-vs-compact movement with respect to full ewok interaction synthesis stable conditional-failure sets.

## Score reproduction

- compact_view_70M: official-domain mean 49.4248; micro accuracy 50.1838; 3823/7618 row matches.
- source_breadth_rowblock_70M: official-domain mean 51.4416; micro accuracy 49.8687; 3799/7618 row matches.

## Domain movement, breadth minus compact

- physical-dynamics: +20.000 points (+24 rows, n=120).
- quantitative-properties: +6.051 points (+19 rows, n=314).
- social-interactions: +2.041 points (+6 rows, n=294).
- social-properties: +0.915 points (+3 rows, n=328).
- material-dynamics: +0.779 points (+6 rows, n=770).
- material-properties: +0.000 points (+0 rows, n=170).
- spatial-relations: -0.816 points (-4 rows, n=490).
- agent-properties: -1.041 points (-23 rows, n=2210).
- physical-interactions: -1.259 points (-7 rows, n=556).
- social-relations: -1.550 points (-24 rows, n=1548).
- physical-relations: -2.934 points (-24 rows, n=818).

## Stable-failure subset movement

- depth_stable: n=2494, compact 29.952%, breadth 30.914%, breadth-compact +0.962 points (+24 rows).
- legal40_8x480_stable: n=2550, compact 30.314%, breadth 30.824%, breadth-compact +0.510 points (+13 rows).
- both_stable: n=1471, compact 20.938%, breadth 20.462%, breadth-compact -0.476 points (-7 rows).
- either_stable: n=3573, compact 33.921%, breadth 35.153%, breadth-compact +1.231 points (+44 rows).
- all_rows: n=7618, compact 50.184%, breadth 49.869%, breadth-compact -0.315 points (-24 rows).
- not_both_stable: n=6147, compact 57.182%, breadth 56.906%, breadth-compact -0.277 points (-17 rows).

## Scientific reading

The 70M EWoK row movement is early and column-only. It can indicate whether the independent-source breadth arm touches the same robust relational-error subset, but it cannot settle the FW mechanism or Overall SOTA route without complete paired cheap7/100M results and the fw ewok interaction reader four-cell/margin readouts.

## Evidence files

- JSON: `experiments/archive/representation_and_objectives/data/fw_70m_ewok_prediction_overlap/fw_70m_ewok_prediction_overlap.json`
- Domain delta CSV: `experiments/archive/representation_and_objectives/data/fw_70m_ewok_prediction_overlap/ewok_70m_domain_deltas.csv`
- Stable subset delta CSV: `experiments/archive/representation_and_objectives/data/fw_70m_ewok_prediction_overlap/ewok_70m_stable_subset_deltas.csv`
