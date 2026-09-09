# intrarow calibration synthesis — opposition/shared-slot active relation pair funnel

This pool rebuilds the muon50 gp and relation pair synthesis/128 relation object by requiring both a hand-coded opposite relation pivot and a more similar target-side local slot. It is built from the same active-token 256-token view used by the PVDM trainers; no model weights are loaded.

Segment: 20000 rows / 3115669 words.
Active events before opposition filtering: 15,834.
Opposition-pivot events kept: 6,801.
Pairs constructed: **90** using 180 events.

## Pairs by opposition axis

| axis | pairs |
|---|---:|
| cause_concession | 39 |
| temporal_before_after | 25 |
| spatial_inclusion | 14 |
| condition_exception | 6 |
| spatial_vertical | 5 |
| build_destroy | 1 |

## Pair quality summaries

Match levels: `{'2': 46, '0': 31, '1': 13}`
Pair cost: `{'n': 90, 'mean': 3.726150258398699, 'std': 5.8424019823418, 'median': 5.337, 'p05': -4.896346153846155, 'p25': -1.5863636363636364, 'p75': 8.502445652173913, 'p95': 12.645617391304349, 'min': -8.155882352941177, 'max': 12.930777777777779}`
Slot overlap: `{'content_jaccard': {'n': 90, 'mean': 0.04054217387550721, 'std': 0.07292805845208737, 'median': 0.0, 'p05': 0.0, 'p25': 0.0, 'p75': 0.08035714285714285, 'p95': 0.17499999999999996, 'min': 0.0, 'max': 0.3333333333333333}, 'surface_jaccard': {'n': 90, 'mean': 0.09973732419591737, 'std': 0.036235948187401605, 'median': 0.09090909090909091, 'p05': 0.04, 'p25': 0.08333333333333333, 'p75': 0.125, 'p95': 0.17065217391304344, 'min': 0.037037037037037035, 'max': 0.2}, 'shared_content_count': {'n': 90, 'mean': 0.45555555555555555, 'std': 0.791076778281225, 'median': 0.0, 'p05': 0.0, 'p25': 0.0, 'p75': 1.0, 'p95': 2.0, 'min': 0.0, 'max': 2.0}, 'shared_surface_count': {'n': 90, 'mean': 2.2444444444444445, 'std': 0.7499794235860535, 'median': 2.0, 'p05': 1.0, 'p25': 2.0, 'p75': 3.0, 'p95': 3.549999999999997, 'min': 1.0, 'max': 4.0}}`
Repeated target-pair count (>=2 contexts): 1

Top pivot pairs:
- ['because', 'though']: 30
- ['after', 'before']: 25
- ['into', 'outside']: 10
- ['although', 'because']: 7
- ['if', 'unless']: 6
- ['above', 'under']: 5
- ['outside', 'within']: 3
- ['broke', 'built']: 1
- ['though', 'thus']: 1
- ['although', 'thus']: 1
- ['inside', 'outside']: 1

Top repeated target pairs:
- ['death', 'start']: 2

Files:
- summary: `experiments/archive/representation_and_objectives/data/relation_opposition_slot_funnel_strict_pilot20k/opposition_slot_pair_funnel_summary.json`
- pairs: `experiments/archive/representation_and_objectives/data/relation_opposition_slot_funnel_strict_pilot20k/opposition_slot_pair_pool.jsonl`
- csv: `experiments/archive/representation_and_objectives/data/relation_opposition_slot_funnel_strict_pilot20k/opposition_slot_pair_pool_summary.csv`
