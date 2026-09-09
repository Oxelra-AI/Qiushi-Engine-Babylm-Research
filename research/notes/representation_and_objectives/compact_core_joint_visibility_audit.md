# compact core joint visibility audit compact_view_core actual joint visibility

JSON: `experiments/archive/representation_and_objectives/data/compact_core_joint_visibility_audit/compact_core_joint_visibility_audit.json`

## Seq256 visibility for the mechanism anchor

- Pair rows/non-pair rows: 2514 / 435 across 2949 changed rows; exact string reconstruction from selected pairs: 2514 rows.
- Full source+rewrite visible pairs: 10012 / 10094 (0.991876); source visible pairs: 10078 (0.998415).
- Partially visible / hidden pairs: 79 / 3; view rows over seq256 with special tokens: 240.
- Non-full-visible pair positions: {3: 41, 4: 23, 2: 16, 1: 1, 5: 1}.

## Token-geometry difference against repeat

- Row-paired view-minus-repeat token delta (no special) mean/median/p95/sum: 7.0905 / 7.0000 / 18.0000 / 20910.0.
- Rows with view token length greater/less/equal than repeat: 2306 / 139 / 504.

## Scientific reading

The compact_view_core mechanism anchor is physically visible in the seq256 interface: nearly all source+compact rewrite pairs survive actual row packing. The remaining interpretation must still account for view-vs-repeat token fragmentation and for whether compact rewrites act through two-view consolidation, cleaner wording, lexical diversity, or a mixture of these.
