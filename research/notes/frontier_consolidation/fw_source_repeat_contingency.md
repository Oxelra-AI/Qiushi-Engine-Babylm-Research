# fw source repeat contingency FW source-repeat contingency

This note records a CPU-only preparation for the literal source-repeat arm. It does not authorize or launch training.

- Status: SOURCE_REPEAT_CONTINGENCY_READY
- Compact/source-repeat rows: 64,183 / 64,183
- Compact/source-repeat words: 10,000,000 / 10,000,000
- Row word sequence matches: True
- Example-id sequence matches: True
- Generated pass orders match adapter scale sweep plan streams: True
- Source-repeat 10M SHA: bf02c71fab2010cd977c9cf4a1267fd18147bd045700c738f92bb9543ee0d6bb
- Manifest: `experiments/archive/frontier_consolidation/data/fw_source_repeat_contingency/source_repeat_stream_dryrun.json`

Use only if the compact-vs-breadth result requires an attribution arm. The first authorized action would be `--write-stream`, then the same earlier analysis training/eval recipe with this source-repeat 100M stream.
