# fw breadth boundary audit — interleaved whole-sentence source-breadth arm

This variant reuses the same repaired independent FineWeb sentences, but places them between the same common source segments to approximate the compact arm's source/rewrite alternation. It avoids cross-row sentence splitting and preserves exact row word totals.

## Layout movement

- Mean absolute source-start displacement vs compact: interleaved 6.496 words; source-block 21.386 words.
- Per-row max absolute displacement median: interleaved 13.0; source-block 42.0.
- Slot length absolute error mean: 11.262; empty slots 11,046, nonempty slots 11,774.

## Token geometry

- Compact visible tokens/word: 1.4449; interleaved breadth visible tokens/word: 1.4412.
- Compact rows over 256: 16,294; interleaved breadth rows over 256: 15,907.
- Compact WWM groups/pass: 9,790,415; interleaved breadth WWM groups/pass: 9,794,364.

## Use

This arm is a layout-closer coherent-breadth comparator. It is useful if the first H100 pair should minimize common-source position movement relative to the compact arm. The row-block whole-sentence breadth arm remains a cleaner coherent-breadth block comparator but has larger source-position movement.

Manifest: `experiments/archive/representation_and_objectives/data/fw_source_breadth_interleaved_wholesentence_arm/fw_source_breadth_interleaved_wholesentence_manifest.json`
10M arm: `experiments/archive/representation_and_objectives/data/fw_source_breadth_interleaved_wholesentence_arm/fw_preserved_source_breadth_interleaved_wholesentence_10M.jsonl`
100M stream: `experiments/archive/representation_and_objectives/data/fw_source_breadth_interleaved_wholesentence_arm/fw_preserved_source_breadth_interleaved_wholesentence_100M.jsonl`
