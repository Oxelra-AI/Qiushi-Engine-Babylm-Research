# fw 70m execution synthesis — FW 70M readout and next-use synthesis

## What was read

frontier_consolidation had completed companion analysis sgcr zero sharing parity FW compact-view and row-block whole-sentence breadth training to 100M, but only paired 70M cheap-eval files were complete on disk during this step. companion analysis managed interleaved breadth task `s106_t10_tool1` remains under runtime management; it was not queried.

The useful CPU work was to read the complete 70M evidence, repair an intermediate-checkpoint loading pitfall, and decide how much weight the 70M result should carry before any full endpoint evaluation.

## 70M cheap7 vector

`scripts/fw_peer_status_and_cheap_synthesis.py` was rerun and found only 70M complete as a paired cheap7 comparison:

| arm | BLiMP | Supp | EWoK | Entity | COMPS | GlobalPIQA | Reading | cheap7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact-view 70M | 65.62 | 57.94 | 49.42 | 27.74 | 51.76 | 38.605 | 7.515 | 42.657 |
| row-block breadth 70M | 65.98 | 58.39 | 51.44 | 22.44 | 51.31 | 36.605 | 8.055 | 42.031 |

Compact is +0.626 cheap7 over row-block breadth. Breadth gains EWoK (+2.02) but loses Entity (-5.30), GlobalPIQA (-2.00), COMPS (-0.45), and Reading is only +0.54. Compact is still 1.113 cheap7 below the visible leader cheap7 43.77; breadth is 1.739 below.

## GlobalPIQA label readout

`scripts/fw_70m_globalpiqa_row_overlap.py` reads existing official predictions only. It reproduces the 70M split scores:

| arm | GlobalPIQA_parallel | GlobalPIQA_nonparallel | aggregate |
|---|---:|---:|---:|
| compact-view 70M | 26.21 | 51.00 | 38.607 |
| row-block breadth 70M | 26.21 | 47.00 | 36.607 |

The full 2-point aggregate GlobalPIQA gap is nonparallel. On the fw globalpiqa relevant substrate 52-row parallel hard set, compact gets 2/52 and row-block breadth gets 1/52; 49 rows are wrong for both. On the 82 rows wrong for at least half of prior endpoints, compact gets 13/82 and breadth gets 11/82. Thus the row-block breadth EWoK gain is not accompanied by an accuracy repair of the load-bearing parallel GlobalPIQA rows.

Files:
- `experiments/archive/representation_and_objectives/data/fw_70m_globalpiqa_row_overlap/fw_70m_globalpiqa_row_overlap.json`
- `research/notes/representation_and_objectives/fw_70m_globalpiqa_row_overlap.md`

## Intermediate-checkpoint loading repair

The first all-option margin wrapper passed the local `hf_model` parent with `revision=chck_70M`. Because `hf_model/` itself also contains final weights, that first run did not faithfully load the 70M subdirectory and did not reproduce the official 70M parallel scores. The repaired wrapper points directly at `hf_model/chck_70M` with `revision=main`.

Use this convention for later 80M/100M intermediate readouts too: when local checkpoint directories exist under an `hf_model` parent that also has weights, pass the checkpoint directory as `model_root`; do not assume Hugging Face `revision` chooses the local subdirectory.

Repaired wrapper:
- `experiments/archive/representation_and_objectives/scripts/fw_70m_globalpiqa_margin_wrapper.py`

## GlobalPIQA margin readout

The repaired margin readout reproduces official 70M GlobalPIQA_parallel accuracy 26.2136 for both arms. It then shows a softer but insufficient movement from breadth:

| arm | parallel acc | rank1/2/3/4 | all mean top-correct | hard52 acc | hard52 rank1/2/3/4 | hard52 mean top-correct | hard52 <=0.50 nats |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact-view 70M | 26.21 | 27/26/24/26 | 1.036 | 3.85 | 2/12/17/21 | 1.703 | 6 |
| row-block breadth 70M | 26.21 | 27/27/33/16 | 0.919 | 1.92 | 1/15/25/11 | 1.476 | 9 |

Compact versus breadth on all 103 parallel rows: same exact accuracy, but breadth has better rank on 32 rows, same rank on 49, worse rank on 22, and lower top-minus-correct margin on 48 rows. On hard52: breadth has better rank on 18 rows, same on 24, worse on 10, and lowers mean top-minus-correct by 0.227 nats.

Scientific reading: row-block breadth nudges probability mass toward correct options on hard parallel rows, especially by reducing rank4 cases, but it does not yet convert that into accuracy; hard52 remains 1/52 correct for breadth. This is not enough to support a new GlobalPIQA-specific expensive training line, especially because the same arm has worse 70M cheap7 through Entity and nonparallel GlobalPIQA losses.

Files:
- `experiments/archive/representation_and_objectives/data/fw_70m_globalpiqa_margin_synthesis/fw_70m_globalpiqa_margin_synthesis.json`
- `research/notes/representation_and_objectives/fw_70m_globalpiqa_margin_synthesis.md`

## Late cheap7 recovery context

`scripts/late_cheap7_recovery_envelope.py` mined available local full-zero-shot/Reading trajectories. The extracted 70M→100M sample is small (two trajectories), so it is only context, not a law. In those records, 70M→100M cheap7 changes were +0.069 and +0.361; 80M→100M changes were -0.149, -0.240, and -0.008.

compact 70M would need +1.113 cheap7 to match the visible leader cheap7, and row-block breadth would need +1.739. Such a jump is larger than the small local late-training range found here. This does not close the 100M FW endpoints, but it supports withholding full official endpoint evaluation until complete 100M cheap7 or other strong endpoint evidence appears.

Files:
- `experiments/archive/representation_and_objectives/data/late_cheap7_recovery_envelope/late_cheap7_recovery_envelope.json`
- `research/notes/representation_and_objectives/late_cheap7_recovery_envelope.md`

## Current research consequence

The FW compact-vs-breadth route remains unresolved until complete 100M cheap7/full-vector evidence is available, but the 70M evidence is not SOTA-plausible by itself. Compact is currently the better endpoint candidate because it preserves Entity and nonparallel GlobalPIQA. Row-block breadth has a real EWoK movement and a weak hidden-rank/margin improvement on GlobalPIQA_parallel hard rows, but these movements are too small and too costly elsewhere to justify a new expensive line.

Next use:
1. When complete 80M/100M cheap files appear, rerun the fw 70m ewok prediction overlap status and plausibility scripts.
2. If a 100M arm is near the 41.80 arithmetic range, run the prepared full official-compatible evaluation on the minimum endpoint set and then run direct-checkpoint GlobalPIQA margin and EWoK four-cell readouts.
3. If 100M cheap7 remains far below range and does not repair the shared relation weaknesses, keep the anchor matched controls ultra-clean 30k transition/control assets as the next small matched probe option rather than scaling a noisy relation corpus.

## Checkpoint-loading provenance addendum

A direct SHA check showed that for sgcr zero sharing parity 100M endpoints, the parent `hf_model/model.safetensors`, `config.json`, and `tokenizer.json` equal `hf_model/chck_100M/` for both compact and row-block breadth. Thus the existing fw ewok interaction reader 100M wrapper that targets the parent `hf_model` is safe for final 100M readouts. The intermediate 70M case is different because the parent points to final weights; direct checkpoint subdirectories are necessary for 70M/80M margin/readout scripts.

100M parent/chck equality:
- compact model SHA `637a84ea3d595b4dd0d587cbed686f4f3f858cd906ac35bc43d6d761a8bde8d1`, tokenizer SHA `e70d167f620668130f88744998bd1bacbe05f6d0735012144fe501b271b04366`
- row-block breadth model SHA `8c0b305cbc5d7c838781ded60bd64c1335d970c5b4de0a4a6a7316e173de396b`, tokenizer SHA `e70d167f620668130f88744998bd1bacbe05f6d0735012144fe501b271b04366`
