# fw breadth boundary audit — repaired FineWeb source-breadth arm with whole-sentence companions

The fw comparison mechanical audit source-breadth arm preserved the compact arm's row word sequence by slicing a global stream of independent FineWeb words. The boundary check found that this made almost every breadth row contain a source-sentence fragment. The repaired arm keeps the same comparison but fills each row's companion budget with whole independent FineWeb sentences.

## Preserved structure

- Total words: 10,000,000; rows: 64,183; FineWeb rows: 5,785.
- Common selected FineWeb source words: 494,154; repaired independent companion words: 318,851; FineWeb block: 813,005.
- Row word sequence matches compact arm: yes; source-sentence splits across breadth rows: 0.
- Selected independent sources: 11,942 sentences / 318,851 words from 6,021 docs, max doc use 6 (cap 6).
- Per-row companion source count: mean 2.064, median 2, max 4.
- Hash overlap with compact-pair selected sources: 0.
- 100M stream written: True; pass order matches existing compact stream: True.

## Use for training

Use this repaired whole-sentence source-breadth arm rather than the fw comparison mechanical audit sliced source-breadth stream if the compact-vs-breadth H100 comparison is launched. The compact arm and shared tokenizer are unchanged.

Manifest: `experiments/archive/representation_and_objectives/data/fw_source_breadth_wholesentence_arm/fw_source_breadth_wholesentence_manifest.json`
10M arm: `experiments/archive/representation_and_objectives/data/fw_source_breadth_wholesentence_arm/fw_preserved_source_breadth_wholesentence_10M.jsonl`
100M stream: `experiments/archive/representation_and_objectives/data/fw_source_breadth_wholesentence_arm/fw_preserved_source_breadth_wholesentence_100M.jsonl`
