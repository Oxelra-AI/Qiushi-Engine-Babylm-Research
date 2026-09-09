# strict innovation target readout before scores — strict innovation target readout before earlier analysis scores

This prepares a strict-innovation target readout before the pending probability-reallocation results are available.

The earlier analysis 70M/80M official-compatible vector alone is not enough to decide whether the research exact-swap variant deserves another long run. A mechanism readout must ask whether the **same repaired row-unique conditional-innovation targets** become easier under the earlier analysis intervention. If broad transfer weakens and this target object does not improve, the innovation-masking family should not continue merely because exact-swap is mechanically cleaner.

## New reproducible artifacts

- Target probe script: `experiments/archive/frontier_consolidation/scripts/strict_innovation_target_probe.py`
- Decision reader: `experiments/archive/frontier_consolidation/scripts/read_innovation_family_decision.py`
- Paired-delta analyzer: `experiments/archive/frontier_consolidation/scripts/analyze_target_probe_deltas.py`
- Current target-probe outputs: `experiments/archive/frontier_consolidation/data/strict_innovation_target_probe/strict_innovation_target_probe.{json,md}`
- Fixed target manifest: `experiments/archive/frontier_consolidation/data/strict_innovation_target_probe/strict_innovation_target_manifest.jsonl`
- Delta anatomy: `experiments/archive/frontier_consolidation/data/strict_innovation_target_delta_anatomy/strict_innovation_target_delta_anatomy.{json,md}`
- Decision reader output, currently waiting for broad scores: `experiments/archive/frontier_consolidation/data/innovation_family_decision_reading/innovation_family_decision_reading.{json,md}`

The target population matches earlier analysis strict map / repaired route reopen conditional innovation logic: changed-row compact source/rewrite pair, both spans visible, rewrite word group absent from paired source, absent from every other word group in the row, and content-like under the earlier analysis stopword/relation filter. Current sample: 768 rows requested, 320 targets selected from 263 rows, 493 target token labels. The SHA checks are the legal frozen stream SHA `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691` and spatial repair route status tokenizer SHA `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`.

## Reference target-prediction anchors, no new GPU

On the fixed 320-target sample:

| checkpoint | true loss | source help | same-row decoy advantage | cross-row decoy advantage |
|---|---:|---:|---:|---:|
| tokenmean_70M | 5.5737 | 1.0553 | 1.6693 | 1.0895 |
| tokenmean_80M | 5.3727 | 1.2219 | 1.8640 | 1.2051 |
| tokenmean_100M | 5.2916 | 1.2792 | 1.9444 | 1.2474 |
| clean_80M | 6.3903 | 0.7800 | 1.2511 | 0.7123 |

Paired row-bootstrap deltas:

- tokenmean_80M - tokenmean_70M: true loss −0.2010 [−0.2713, −0.1251], source help +0.1666 [+0.0886, +0.2464], same-row decoy advantage +0.1947 [+0.1141, +0.2688].
- tokenmean_100M - tokenmean_80M: true loss −0.0811 [−0.1203, −0.0447], source help +0.0573 [+0.0166, +0.0934], same-row decoy advantage +0.0804 [+0.0400, +0.1221].
- tokenmean_80M - clean_80M: true loss −1.0175 [−1.2063, −0.8448], source help +0.4418 [+0.2678, +0.6065], same-row decoy advantage +0.6129 [+0.4275, +0.8141].

Reading: compact-view reinvestment already strongly improves this exact object over the clean control, and ordinary extra exposure continues to improve it from 70→80 and 80→100M. Therefore a earlier analysis mechanism signal should be compared with these natural increments, not just declared from a raw target loss number.

## Exact post-delivery commands

After probability-reallocation training and evaluation complete and the required files exist, run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/strict_innovation_target_probe.py \
  --checkpoints tokenmean_70M,tokenmean_80M,tokenmean_100M,clean_80M,reference_70M,reference_80M \
  --sample-rows 768 --max-targets 320 --batch-size 16 --torch-threads 8 --bootstrap-iters 200

PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/analyze_target_probe_deltas.py

PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/read_innovation_family_decision.py
```

The decision reader combines:

1. `experiments/archive/frontier_consolidation/data/strict_innovation_trajectory/strict_innovation_vs_trajectory.json`
2. `experiments/archive/frontier_consolidation/data/strict_innovation_target_probe/strict_innovation_target_probe.json`

## Interpretation rule now preserved

- If earlier analysis broad 80M is strongly positive across the cheap surface, especially BLiMP/Supplement/EWoK, continue earlier analysis before launching exact-swap.
- If earlier analysis broad 80M is weak/negative/redistributive but the strict target readout improves, close earlier analysis yet keep research exact-swap as a distinct lower-perturbation hypothesis because the failure may be excessive pressure or copyable suppression.
- If earlier analysis broad 80M is weak/negative/redistributive and the strict target readout does **not** improve, close the innovation-masking family for now. Do not launch exact-swap simply because it is mechanically cleaner.

No new GPU work was launched.
