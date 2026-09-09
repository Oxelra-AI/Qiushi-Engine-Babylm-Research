# fw 70m ewok prediction overlap — FW peer endpoints and early EWoK readout synthesis

## Active question

The active expensive evidence remains the FineWeb compact-versus-breadth family. companion analysis reported that its sgcr zero sharing parity paired 100M trainings completed, and companion analysis's interleaved-breadth managed task `s106_t10_tool1` remains under runtime supervision. This step did not launch new H100 training or full evaluation; it inspected completed artifacts and built CPU-only decision/readout summaries.

## training artifacts are real and ready for endpoint readouts

Evidence files:

- note: `research/notes/frontier_consolidation/fw_training_complete_verification.md`
- preflight: `experiments/archive/frontier_consolidation/data/fw_compact_vs_breadth_train/preflight.json`
- status summary: `experiments/archive/representation_and_objectives/data/fw_peer_status/fw_peer_status_and_cheap_synthesis.json`
- note: `research/notes/representation_and_objectives/fw_peer_status_and_cheap_synthesis.md`

Verified properties:

- `compact_view` run: `experiments/archive/frontier_consolidation/training/runs/fw_compact_view_shared16k_seed43022`
  - `chck_100M` exists, 100 checkpoint directories, word exposure 100,000,000, 2,508 training steps.
  - DeBERTa-v2 8×480, 34,467,424 parameters, fixed WWM 0.15, seq 256, seed 43 / init 43022 / train RNG 43023.
  - final loss 2.4668.
  - saved tokenizer SHA `e70d167f620668130f88744998bd1bacbe05f6d0735012144fe501b271b04366`.
- `source_breadth_rowblock` run: `experiments/archive/frontier_consolidation/training/runs/fw_source_breadth_shared16k_seed43022`
  - `chck_100M` exists, 100 checkpoint directories, same recipe/seeds/checkpoint ladder.
  - final loss 2.4749.
  - saved tokenizer SHA `e70d167f620668130f88744998bd1bacbe05f6d0735012144fe501b271b04366`.
- preflight records the intended streams:
  - compact stream SHA `c8d7f24b5edd2dad21f589c8cde72671d0b76038d9632fcd3d6c79178a24be68`;
  - row-block breadth stream SHA `1b98269fb210cc9494885ec47ee26d9fd1b6ead60a1c308f5d9c47d9b385731d`;
  - shared tokenizer SHA `e70d167f620668130f88744998bd1bacbe05f6d0735012144fe501b271b04366`.

dry-ran the posttraining official-compatible controller for both endpoints and both fw ewok interaction reader relational readout wrappers; all reported `READY`:

- `scripts/fw_shared_anchor_posttrain_eval_controller.py --target fw_compact_fullbatch_seed43022 --dry-run`
- `scripts/fw_shared_anchor_posttrain_eval_controller.py --target fw_breadth_rowblock_fullbatch_seed43022 --dry-run`
- `scripts/fw_globalpiqa_margin_wrapper.py --targets fw_compact_fullbatch_seed43022 fw_breadth_rowblock_fullbatch_seed43022 --dry-run`
- `scripts/fw_ewok_interaction_reader.py --targets fw_compact_fullbatch_seed43022 fw_breadth_rowblock_fullbatch_seed43022 --dry-run`

## Available cheap evidence is incomplete and should not drive full evaluation alone

visible cheap eval files at the time of the fw 70m ewok prediction overlap status synthesis:

- `70M compact_view`: complete 7/7 cheap columns, cheap7 = 42.6571.
- `70M source_breadth_rowblock`: only BLiMP/Supplement/EWoK complete, no cheap7 yet.
- `80M` and `100M`: no visible per-target JSON yet.

The analysis uses `scripts/fw_sota_plausibility_calculator.py` and produced:

- JSON: `data/fw_sota_plausibility/fw_sota_plausibility_calculator.json`
- note: `notes/fw_sota_plausibility_calculator.md`

Arithmetic reading:

- Visible 41.80 leader cheap7 = 43.7700 and SuperGLUE+AoA = 69.79.
- To reach Overall 41.80 with SuperGLUE+AoA 69.79, a model needs cheap7 ≈ 43.7729.
- To reach Overall 41.80 with SuperGLUE+AoA 71.04, it still needs cheap7 ≈ 43.5943.
- The compact 70M cheap7 = 42.6571 would require SuperGLUE+AoA = 77.60 to reach 41.80 at that same cheap7, far outside observed local complete vectors.

This does not close the FW family because 80M/100M cheap7 may differ, and breadth 70M is incomplete. It does block treating compact 70M alone as a reason to spend full official evaluation before the cheap screens mature.

## 70M EWoK row-level readout: breadth's EWoK gain is not a direct repair of the strongest shared stable failures

70M EWoK cheap predictions were available for both compact and row-block breadth. The comparison uses `scripts/fw_70m_ewok_prediction_overlap.py` and produced:

- JSON: `data/fw_70m_ewok_prediction_overlap/fw_70m_ewok_prediction_overlap.json`
- note: `notes/fw_70m_ewok_prediction_overlap.md`
- domain deltas: `data/fw_70m_ewok_prediction_overlap/ewok_70m_domain_deltas.csv`
- stable-subset deltas: `data/fw_70m_ewok_prediction_overlap/ewok_70m_stable_subset_deltas.csv`

The script first reproduced official EWoK scoring as mean of domain accuracies:

- compact_view_70M: official-domain mean 49.4248; micro accuracy 50.1838.
- source_breadth_rowblock_70M: official-domain mean 51.4416; micro accuracy 49.8687.

So the official +2.0168 EWoK advantage for breadth coexists with a micro row-match disadvantage of −0.315 points. The gain is strongly domain-weighted:

- physical-dynamics +20.000 points (+24/120 rows);
- quantitative-properties +6.051 points (+19/314 rows);
- social-interactions +2.041 points (+6/294 rows);
- but physical-relations −2.934, social-relations −1.550, agent-properties −1.041.

Against the full ewok interaction synthesis stable conditional-failure sets:

- `depth_stable`: breadth +0.962 points (+24 rows).
- `legal40_8x480_stable`: breadth +0.510 points (+13 rows).
- `both_stable`: breadth −0.476 points (−7 rows).
- `either_stable`: breadth +1.231 points (+44 rows).
- all rows: breadth −0.315 points (−24 rows).

Scientific reading: row-block breadth's early 70M EWoK advantage touches some architecture-specific stable-failure subsets and small physical/quantitative domains, but it does **not** repair the strongest shared stable-failure intersection. This supports the fw globalpiqa relevant substrate/anchor matched controls caution: aggregate EWoK movement can be domain-weighted or subset-specific and must be read through four-cell conditional interaction and GlobalPIQA margins before route decisions.

## Consequence for next execution

1. Let cheap 70M/80M evidence finish; do not duplicate it.
2. When complete paired 80M or 100M cheap7 exists, rerun:
   - `scripts/fw_peer_status_and_cheap_synthesis.py`
   - `scripts/fw_sota_plausibility_calculator.py`
3. If a 100M arm is SOTA-plausible by cheap7 arithmetic, run the prepared full official-compatible controller and fw ewok interaction reader relational readouts on the minimum necessary endpoints.
4. If FW arms do not improve both Overall plausibility and the relational readouts, do not scale the anchor matched controls transition substrate; use the ultra-clean 30k treatment/control only as a small matched probe after the FW result is read.

No compliant endpoint above 41.80 exists yet.
