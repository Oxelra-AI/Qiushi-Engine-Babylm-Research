# changed block neardup scan changed-block near-duplicate scan

This CPU-only scan indexes score-bearing official evaluation strings by rare 5-grams and compares each individual compact source/rewrite packet in the reinvest changed block against retrieved candidates. It is a provenance check only; no data or model artifact is changed.

## Counts

- Score-bearing eval text records indexed: 278499.
- Unique rare 5-grams in index (df <= 40): 938181.
- Packet sides scanned: 23892.
- Retrieved candidate packet/eval comparisons scored: 2309.
- Threshold-passing near-duplicate hits: 0 across 0 pair-sides.
- Hit distribution by top dir: {}.
- Maximum contiguous token span observed among scored candidates: 8.
- Maximum 5-gram Jaccard observed among scored candidates: 0.2222.

## Top threshold-passing hits

## Reading

No packet/eval pair passed the high-recall near-duplicate thresholds. This strengthens the exact-overlap reading: the changed block has sparse generic exact overlaps and no obvious near-copy of score-bearing eval text at packet level.

Machine-readable JSON: `experiments/archive/frontier_consolidation/data/changed_block_neardup_scan/changed_block_neardup_scan.json`
Hit JSONL: `experiments/archive/frontier_consolidation/data/changed_block_neardup_scan/changed_block_neardup_hits.jsonl`
