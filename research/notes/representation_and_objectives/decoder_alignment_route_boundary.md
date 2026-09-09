# decoder alignment route boundary decoder-robust alignment route boundary

## Purpose

This analysis tested the midlayer base80 proposal without launching training: whether an interior conditional-ordering signal survives decoder/basis controls strongly enough to motivate a protected late path. Depth/band and decoder rule were fixed using legal-corpus no-label alignment and the non-official earlier analysis naturalistic bridge, then EWoK/GlobalPIQA were used only as readouts.

## Execution artifacts

- Script: `experiments/archive/representation_and_objectives/scripts/decoder_robust_alignment_probe.py`
- Per-target outputs:
  - `experiments/archive/representation_and_objectives/data/decoder_alignment_base80/matched_base_80M/decoder_alignment_target_summary.json`
  - `experiments/archive/representation_and_objectives/data/decoder_alignment_scale80/scale1p75_live_80M/decoder_alignment_target_summary.json`
  - `experiments/archive/representation_and_objectives/data/decoder_alignment_disabled80/scale1p75_disabled_80M/decoder_alignment_target_summary.json`
- Collation: `experiments/archive/representation_and_objectives/scripts/collate_decoder_alignment_probe.py`
- Synthesis: `experiments/archive/representation_and_objectives/data/decoder_alignment_synthesis/decoder_alignment_synthesis.json`
- Per-target bridge-only selector check: `experiments/archive/representation_and_objectives/data/decoder_alignment_synthesis/bridge_target_selector_anatomy.json`

## No-label alignment and endpoint validation

Each target fitted layerwise statistic and ridge maps on 3,072 masked states from the legal compact-view pretraining corpus (`frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`) only. No official labels or score rows were used to fit the maps.

Final-head equality passed exactly in the full runs:

- matched base: max |`model.cls(h_final)` - `out.logits`| = 0.0; target log-prob diff = 0.0.
- scale1.75 live: max logit diff = 0.0; target log-prob diff = 0.0.
- scale1.75 disabled: max logit diff = 0.0; target log-prob diff = 0.0.

The scale1.75 custom model source confirms hidden states after each adapter addition (`layer_output = layer_output + self.adapter(layer_output)`), so the layer readout is reading the residual state actually passed onward.

## Bridge-fixed selection result

The non-official bridge supplied no robust positive interaction coordinate. The pooled selector over matched base + scale1.75 live chose `raw` L1-L2 only because it had **one** persistent actual recovery and zero persistent swap recovery across 1,152 target-pair cases:

- selected rule: decoder=`raw`, adjacent band L1-L2
- band both-correct = 3/1152, band swap-both-correct = 3/1152
- persistent both-correct = 1/1152, persistent swap = 0/1152
- mean four-cell M = -0.0504
- min correct margins for the three band-correct cases are tiny (mean 0.0108 nats)

The per-target bridge anatomy makes this even weaker:

- matched base alone: best raw L1-L2 has 1/576 persistent actual and 0/576 persistent swap.
- scale1.75 live alone: best score is raw L0-L1 with 0 actual and 0 swap; no positive persistent bridge recovery. Late bands strongly favor the swap null (e.g. raw L7-L8 persistent actual 74 vs swap 123).
- scale1.75 disabled is similar to live.

Thus there is no held-out natural-interaction evidence for a protected late path. The official readouts below are not allowed to select a better depth.

## Official readout under the fixed weak rule

Using the pre-fixed raw L1-L2 equal blend:

EWoK transition-enriched rows (n=925):

- matched base: both=0.0832, persistent both=0.0173, persistent swapped null=0.0195, stable failure=0.3005.
- scale1.75 live: both=0.0822, persistent both=0.0249, persistent swapped null=0.0119, stable failure=0.3049.
- scale1.75 disabled: both=0.0886, persistent both=0.0205, persistent swapped null=0.0054, stable failure=0.3081.

GlobalPIQA hard52:

- matched base: true accuracy=0.1154, persistent true=0.0769, rotated-label persistent null=0.3077.
- scale1.75 live: true accuracy=0.1538, persistent true=0.0769, rotated-label persistent null=0.2692.
- scale1.75 disabled: true accuracy=0.1538, persistent true=0.0769, rotated-label persistent null=0.2692.

Paired fixed-rule deltas:

- scale1.75 live minus matched base: EWoK net persistent both +7 but net stable failures +4; GlobalPIQA hard52 persistent true delta 0 and mean top-minus-true delta +0.0671 nats (worse margin on average; negative would be better), with 20 improved-margin rows vs 27 worsened.
- scale1.75 disabled minus matched base: EWoK net persistent both +3 but net stable failures +7; GlobalPIQA hard52 persistent true delta 0 and mean top-minus-true delta +0.0712 nats.
- live minus disabled: EWoK interaction mean +0.0003, net persistent both +4, net stable -3; GlobalPIQA hard52 persistent true delta 0 and mean margin delta -0.0041 nats, essentially null.

## Scientific implication

The decoder-robust reanalysis closes the simple late-path/readout branch as tested. midlayer base80's apparent `any middle layer` recovery does not survive the required bridge-fixed, adjacent-depth, no-label-aligned, null-compared interpretation:

1. The non-official bridge does not identify any robust conditional-ordering depth or decoder mode. The selected rule is a one-row artifact with tiny margins.
2. Scale1.75, the model whose broad score made this branch tempting, has no positive target-specific bridge band; its late bridge bands favor the swapped null.
3. Under the fixed weak rule, EWoK and GlobalPIQA official rows show no coherent scale-specific recovered interior signal erased late. GlobalPIQA hard52 persistent correctness is identical for base/live/disabled and far below the rotated-label null; scale margins are slightly worse than base under the fixed band.
4. Live-vs-disabled effects are near-null under the fixed rule, consistent with related experiments: scale1.75's relation cost is embedded in the co-trained trajectory/representation, not an inference-time overlay.

The next high-value route should stop trying to protect a late readout and instead form the missing representation: legal, broad, context-sensitive experience or architecture that makes alternatives compete under context while preserving the scale1.75 broad gains. Any H100 training should be preceded by a minimum-cost screen that uses matched EWoK transition rows, GlobalPIQA hard ranks, held-text MLM loss, and a non-official held-out natural interaction object; packet/bridge replay and local-logit natural miners remain closed.
