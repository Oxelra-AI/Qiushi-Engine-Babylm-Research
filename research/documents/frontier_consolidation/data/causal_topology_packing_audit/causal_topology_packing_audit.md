# deberta grid tooling and topology packing confound causal topology packing audit

Manifest: `experiments/archive/frontier_consolidation/data/causal_topology_2x2_scaffold/manifest.json`. Epochs audited: [0, 1, 2]. Seq length: 256.

## Arm/epoch summary

| arm | epoch | active tokens | chunks | dropped tail | pair rows | pair row cross-chunk frac | pair units share chunk frac | pair units adjacent-doc frac | pair row token mean | pair doc distance median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| compact_oneway | 0 | 14587648 | 56983 | 60 | 12142 | 0.202108 | 0.000000 | 0.000000 | 51.698 | 6.000 |
| compact_oneway | 1 | 14587648 | 56983 | 60 | 12142 | 0.202767 | 0.000000 | 0.000000 | 51.698 | 21293.000 |
| compact_oneway | 2 | 14587648 | 56983 | 60 | 12142 | 0.200708 | 0.000000 | 0.000000 | 51.698 | 21534.000 |
| repeat_oneway | 0 | 14555904 | 56859 | 166 | 12142 | 0.186954 | 0.000000 | 0.000000 | 49.092 | 6.000 |
| repeat_oneway | 1 | 14555904 | 56859 | 166 | 12142 | 0.188849 | 0.000000 | 0.000000 | 49.092 | 21293.000 |
| repeat_oneway | 2 | 14555904 | 56859 | 166 | 12142 | 0.187613 | 0.000000 | 0.000000 | 49.092 | 21534.000 |
| compact_reciprocal | 0 | 14587648 | 56983 | 112 | 12142 | 0.195767 | 0.000000 | 0.000000 | 51.702 | 6.000 |
| compact_reciprocal | 1 | 14587648 | 56983 | 112 | 12142 | 0.201779 | 0.000000 | 0.000000 | 51.702 | 21293.000 |
| compact_reciprocal | 2 | 14587648 | 56983 | 112 | 12142 | 0.194037 | 0.000000 | 0.000000 | 51.702 | 21534.000 |
| repeat_reciprocal | 0 | 14555904 | 56859 | 166 | 12142 | 0.186954 | 0.000000 | 0.000000 | 49.092 | 6.000 |
| repeat_reciprocal | 1 | 14555904 | 56859 | 166 | 12142 | 0.188849 | 0.000000 | 0.000000 | 49.092 | 21293.000 |
| repeat_reciprocal | 2 | 14555904 | 56859 | 166 | 12142 | 0.187613 | 0.000000 | 0.000000 | 49.092 | 21534.000 |

## Key comparisons

| epoch | C-R active tokens oneway | C-R active tokens reciprocal | recip-oneway active compact | recip-oneway active repeat | recip-oneway pair share compact | recip-oneway pair share repeat |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 31744.0 | 31744.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 1 | 31744.0 | 31744.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2 | 31744.0 | 31744.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Interpretation

- The GPT trainer concatenates full tokenized rows plus EOS and chunks the stream; rows longer than 256 are legal for this trainer but can span multiple chunks.
- Epoch 0 preserves JSONL document order; later epochs shuffle whole rows with random.Random(seed+epoch), so two units of the same pair usually stop being adjacent even though each unit still contains source and view/repeat within one row.
- The intended causal conditioning is inside each pair-unit row. Pair-unit rows that cross chunk boundaries provide only partial within-row context for tokens after the boundary; the fractions here quantify that exposure geometry.
- A future topology interaction must be read with these packing quantities, copied-token overlap, and compact-vs-repeat token asymmetry; this audit does not authorize training.

JSON: `experiments/archive/frontier_consolidation/data/causal_topology_packing_audit/causal_topology_packing_audit.json`
