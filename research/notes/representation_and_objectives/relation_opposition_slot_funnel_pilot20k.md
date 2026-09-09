# intrarow calibration synthesis — opposition/shared-slot active relation pair funnel

This pool rebuilds the muon50 gp and relation pair synthesis/128 relation object by requiring both a hand-coded opposite relation pivot and a more similar target-side local slot. It is built from the same active-token 256-token view used by the PVDM trainers; no model weights are loaded.

Segment: 20000 rows / 3115669 words.
Active events before opposition filtering: 15,834.
Opposition-pivot events kept: 6,864.
Pairs constructed: **602** using 1,204 events.

## Pairs by opposition axis

| axis | pairs |
|---|---:|
| cause_concession | 261 |
| temporal_before_after | 190 |
| spatial_inclusion | 73 |
| condition_exception | 35 |
| spatial_vertical | 23 |
| build_destroy | 8 |
| open_close | 4 |
| enter_leave | 3 |
| comparative_amount | 2 |
| change_amount | 1 |
| push_pull | 1 |
| size_change | 1 |

## Pair quality summaries

Match levels: `{'2': 296, '0': 158, '1': 148}`
Pair cost: `{'n': 602, 'mean': 4.410611525862504, 'std': 5.452492119031206, 'median': 3.210090909090909, 'p05': -3.231767734553776, 'p25': -1.0, 'p75': 8.938340909090911, 'p95': 12.168316943187525, 'min': -8.155882352941177, 'max': 13.399999999999999}`
Slot overlap: `{'content_jaccard': {'n': 602, 'mean': 0.06512221119727832, 'std': 0.04294381764311267, 'median': 0.07417582417582418, 'p05': 0.0, 'p25': 0.04887218045112782, 'p75': 0.09090909090909091, 'p95': 0.125, 'min': 0.0, 'max': 0.3333333333333333}, 'surface_jaccard': {'n': 602, 'mean': 0.10647911984968567, 'std': 0.05528830852708455, 'median': 0.08695652173913043, 'p05': 0.038461538461538464, 'p25': 0.05065789473684211, 'p75': 0.15, 'p95': 0.2, 'min': 0.037037037037037035, 'max': 0.25}, 'shared_content_count': {'n': 602, 'mean': 0.760797342192691, 'std': 0.4493532030847273, 'median': 1.0, 'p05': 0.0, 'p25': 1.0, 'p75': 1.0, 'p95': 1.0, 'min': 0.0, 'max': 2.0}, 'shared_surface_count': {'n': 602, 'mean': 2.377076411960133, 'std': 1.1216708989079165, 'median': 2.0, 'p05': 1.0, 'p25': 1.0, 'p75': 3.0, 'p95': 4.0, 'min': 1.0, 'max': 5.0}}`
Repeated target-pair count (>=2 contexts): 4

Top pivot pairs:
- ['after', 'before']: 190
- ['because', 'though']: 172
- ['although', 'because']: 65
- ['into', 'outside']: 55
- ['if', 'unless']: 35
- ['above', 'under']: 20
- ['inside', 'outside']: 11
- ['therefore', 'though']: 8
- ['though', 'thus']: 7
- ['outside', 'within']: 7
- ['although', 'thus']: 5
- ['although', 'therefore']: 4

Top repeated target pairs:
- ['boxes', 'stairs']: 2
- ['felt', 'start']: 2
- ['gets', 'likes']: 2
- ['going', 'living']: 2

Files:
- summary: `experiments/archive/representation_and_objectives/data/relation_opposition_slot_funnel_pilot20k/opposition_slot_pair_funnel_summary.json`
- pairs: `experiments/archive/representation_and_objectives/data/relation_opposition_slot_funnel_pilot20k/opposition_slot_pair_pool.jsonl`
- csv: `experiments/archive/representation_and_objectives/data/relation_opposition_slot_funnel_pilot20k/opposition_slot_pair_pool_summary.csv`
