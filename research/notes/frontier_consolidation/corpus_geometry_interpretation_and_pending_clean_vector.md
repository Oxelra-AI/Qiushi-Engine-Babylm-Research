# corpus geometry interpretation and pending clean vector — Training-substrate geometry for the pending legal-tokenizer treatment vector

This note records the CPU-only repair and execution of `compare_corpus_treatment_geometry.py`. It uses only training-side files and the fixed spatial repair route status same-pool tokenizer. It does not use official evaluation text, does not launch GPU work, and does not choose the next intervention without the missing clean-control vector.

## Why this was useful now

The decisive comparison remains clean-Qwen versus compact-view reinvest at 70M/80M under the same tokenizer. While the clean results are pending, training-substrate measurements can establish the actual corpus changes needed to interpret the later reinvest-minus-clean vector.

## Script repair

`experiments/archive/frontier_consolidation/scripts/compare_corpus_treatment_geometry.py` previously failed because it derived the wrong root with `parents[3]`, causing a write attempt at a read-only location. The repair used the following original code identifiers:

- `SESSION_ROOT = Path(__file__).resolve().parents[2]`
- `SESSIONS_ROOT = SESSION_ROOT.parent`

Then the script was parsed and run successfully.

## Integrity anchors

Output: `experiments/archive/frontier_consolidation/data/corpus_treatment_geometry/corpus_treatment_geometry.json` and `.md`.

The script verified all expected hashes:

- clean 10M SHA matched: `e0a3cdc20e39f2715fbdb0cfbc6c4aff61d51924c480a982049878272c5690b3`
- clean 100M SHA matched: `728192f8e8c5855aaa52a6b6940ff3a4fd6aa8f87c98aca4ff018dbc04cb0345`
- reinvest 10M SHA matched: `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`
- reinvest 100M SHA matched: `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`
- spatial repair route status tokenizer SHA matched: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`

## What changed in the training substrate

Clean-Qwen and compact-view-reinvest both contain exactly 10,000,000 declared words, but the treatment changes the substrate in a very specific way:

- Rows: clean 64,381; reinvest 64,740; delta +359.
- Reinvest inserts 3,005 FineWeb source+compact-view pair rows carrying 423,511 words.
- It removes 423,520 words of clean-Qwen official-source material, mostly CHILDES (-137,760), Gutenberg (-99,520), OpenSubtitles (-97,760), SimpleWiki (-56,640), BNC spoken (-30,720), and a small Switchboard slice (-1,120), plus 9 neutral top-up words.
- The inherited Qwen paired block remains unchanged at 1,656,800 words.

Under the fixed spatial repair route status tokenizer, reinvest is not simply a larger target-volume treatment:

- Raw tokens per word: clean 1.468864; reinvest 1.466452; delta -0.002412.
- Visible tokens per word at seq256: clean 1.430053; reinvest 1.429489; delta -0.000564.
- Visible WWM groups per word: clean 0.979209; reinvest 0.980219; delta +0.001010.
- Rows over seq256: clean 15,872; reinvest 15,117; delta -755.
- Truncated tokens: clean 388,105; reinvest 369,626; delta -18,479.

Thus the treatment slightly reduces subword token volume and truncation while very slightly increasing visible whole-word group opportunities. Any mature behavioral difference should be interpreted as a source/diversity/second-view effect, not as a large token-count or truncation artifact.

## Coarse corpus cues

The reinvest corpus is not relation-enriched in the broad cue sense used by this script. Relative to clean-Qwen:

- Any measured cue fraction decreases from 0.259732 to 0.255834, delta -0.003898.
- Mental/social/dialogue cues decrease by -0.003564 of declared words.
- Negation/modality decreases by -0.000608.
- Spatial, causal/temporal, and physical-dynamics cues are essentially flat or slightly lower.
- Material/property and quantity cues increase only very slightly (+0.000234 and +0.000208).

This aligns with Steps56-57: relation-only masking is not well matched to the actual legal-coordinate difficulty. The compact-view treatment is better represented as budget-reallocated source diversity plus semantic second views under lower redundancy, not as a simple relation-token pressure increase.

## How to use this when clean results land

After clean 20M/70M/80M cheap-column results become available, rerun:

- `python experiments/archive/frontier_consolidation/scripts/compare_legal_treatment_trajectory.py`
- `python experiments/archive/frontier_consolidation/scripts/mature_treatment_subtask_interpreter.py`

Interpret reinvest-minus-clean together with this substrate measurement:

- If reinvest is broadly positive at 70M/80M, the compact-view source-diversity mechanism survives under the legal tokenizer; the next expensive work should focus on legal representation/optimization rather than relation-only masking.
- If reinvest helps knowledge/source-diversity columns but loses syntax/dialogue-heavy columns, the next mechanism should explain the tradeoff between source breadth and lexical/syntactic consolidation.
- If reinvest is broadly below clean, the old 42.033 mechanism likely depended on the inherited representation coordinate and the broader route should be reopened.

No final result or submission is supported yet. The best fully legal endpoint remains 41.2578, below the current 41.8 target, and the clean fixed-tokenizer mature vector is still missing.
