# fw 100m cheap7 synthesis — FW shared-anchor endpoint and three-arm readout synthesis

## Interleaved endpoint verification

The interleaved whole-sentence breadth arm is a valid completed 100M endpoint for the shared-anchor FineWeb comparison.

- Run dir: `experiments/archive/representation_and_objectives/training/runs/fw_source_breadth_interleaved_wholesentence_fullbatch_shared16k_seed43022`
- Stream SHA: `1758508e3bfae8bda8d4d66409ebd695499552a495e93d8327be8aa0d9182c44`
- Shared tokenizer SHA: `e70d167f620668130f88744998bd1bacbe05f6d0735012144fe501b271b04366`
- Trainer: `experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py`
- Recipe: DeBERTa-v2 MLM, 8 layers, hidden 480, 8 heads, FFN x4, batch 256, sequence length 256, AdamW LR 0.001, warmup 0.06, weight decay 0.01, fixed WWM 0.15
- Params: 34,467,424
- Training: 2,508 steps, exact 100,000,000 word exposure, 100 checkpoint dirs, final loss 2.508267402648926
- Parent `hf_model` and `hf_model/chck_100M` are SHA-identical for config, model weights, tokenizer files, and special-token map. Parent model SHA: `b1ad655aaac92c1c3eb53d2b822f7e6cee5d38e651b40a47b2858ede206a0dcd`.

Verification used a direct file-hash check and the run files `scientific_metrics.json`, `example_order_manifest.json` and `train_result.json`.

## 100M cheap7 comparison

cheap7 = mean(BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, Reading). SuperGLUE and AoA were not run here; this is a route readout, not a submission Overall.

| arm | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading | cheap7 | gap to leader cheap7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| compact same-proposition recurrence | 66.87 | 58.86 | 50.25 | 28.36 | 51.84 | 38.635 | 7.455 | 43.1814 | -0.5886 |
| row-block independent breadth | 67.90 | 59.69 | 50.50 | 23.88 | 51.33 | 37.065 | 8.105 | 42.6386 | -1.1314 |
| interleaved independent breadth | 66.89 | 56.42 | 51.85 | 25.60 | 51.75 | 41.105 | 7.940 | 43.0793 | -0.6907 |

Visible leader cheap7 is about 43.77. To reach Overall 41.80, the three arms would need SuperGLUE+AoA around 73.93 (compact), 77.73 (row-block), and 74.64 (interleaved), versus the visible leader's SuperGLUE+AoA about 69.79 and the best inherited noncompliant SuperGLUE about 71.04. This does not make a full official evaluation a good use of compute at this point.

Evidence:
- `experiments/archive/representation_and_objectives/data/fw_shared_anchor_full_eval/per_target/*.json`
- `experiments/archive/representation_and_objectives/data/fw_shared_anchor_official_ewok/*/official_ewok_reeval_*.json`
- `experiments/archive/representation_and_objectives/data/fw_100m_cheap7_synthesis/fw_100m_cheap7_synthesis.json`
- `research/notes/representation_and_objectives/fw_100m_cheap7_synthesis.md`

## GlobalPIQA row and margin readout

The GlobalPIQA three-arm readout shows that the useful movements remain split across layout and subset.

Parallel:
- Compact: 24.27 accuracy, hard52 2/52, hard52 mean top-minus-correct 1.717 nats.
- Row-block breadth: 29.13 accuracy, hard52 3/52, hard52 mean top-minus-correct 1.433 nats.
- Interleaved breadth: 26.21 accuracy, hard52 2/52, hard52 mean top-minus-correct 1.660 nats.

Nonparallel:
- Compact: 53.0 accuracy.
- Row-block breadth: 45.0 accuracy.
- Interleaved breadth: 56.0 accuracy.

Across 103 parallel rows, three-arm oracle-union accuracy is 40.78, but 61 rows remain wrong for all three arms and only 17 are correct for all three. Correctness patterns in compact/row-block/interleaved order: `000=61`, `111=17`, `010=8`, `001=6`, `100=5`, `011=3`, `110=2`, `101=1`. Row-block has the clearest parallel shift; interleaved does not preserve that shift strongly, but gains nonparallel and EWoK aggregate.

Evidence:
- `experiments/archive/representation_and_objectives/data/fw_globalpiqa_margin_reader/a02_fw_compact_fullbatch_seed43022_margins.json`
- `experiments/archive/representation_and_objectives/data/fw_globalpiqa_margin_reader/a02_fw_breadth_rowblock_fullbatch_seed43022_margins.json`
- `experiments/archive/representation_and_objectives/data/fw_globalpiqa_margin_reader/a01_fw_breadth_interleaved_fullbatch_seed43022_margins.json`
- `experiments/archive/representation_and_objectives/data/fw_globalpiqa_threearm_synthesis/fw_globalpiqa_threearm_synthesis.json`
- `research/notes/representation_and_objectives/fw_globalpiqa_threearm_synthesis.md`

## Scientific interpretation

The three FW arms do not contain the desired combination in a single checkpoint. Compact remains the strongest broad cheap7 arm. Row-block breadth supplies the clearest GlobalPIQA_parallel rank/margin movement, but pays heavily in Entity and nonparallel GlobalPIQA. Interleaved breadth repairs some of that broad damage and improves EWoK and nonparallel GlobalPIQA relative to compact, but it still trails compact cheap7 and does not reproduce row-block's stronger hard-parallel movement. The shared GlobalPIQA hard-row weakness remains mostly intact.

Therefore the FW allocation family is not a current SOTA route. It has taught a useful mechanism-level lesson: independent coherent FineWeb coverage can move different world-relation rows, but without same-proposition recurrence it does not consolidate into a broad-capability gain under the shared legal 16k representation. Local layout changes alter which relation signal appears, which weakens further FW allocation variants as a path to 41.8+ unless the pending EWoK four-cell readers reveal a much stronger conditional-compatibility change than the aggregate scores imply.

Pending readouts:
- compact + row-block EWoK four-cell reader from execution synthesis.
- interleaved EWoK four-cell reader submitted in fw 100m cheap7 synthesis to separate output paths.

If those readouts do not show a strong, broad relation-compatibility improvement coupled with compact-like broad capability, the next evidence-producing route should stop extending FW allocation variants and move to a more direct general relational/causal transition representation test, starting from the transition structure match 12,418-word compact-contained matched transition/control extra subset only as a modest mechanism probe.
