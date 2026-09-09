# commoncopy and paired world design — common-copy readout harness and conditional paired-world mechanism design

## Active causal decision

The architecture-interaction experiment showed a strong selected-score interaction:

\[
I_{\text{plain nodis-full}}=(\text{compact}-\text{repeat})_{\text{plain no-disentangle}}-(\text{compact}-\text{repeat})_{\text{full DeBERTa}}
\]

was negative on stable families. Mean over 80M/100M:

- full DeBERTa compact-minus-repeat: cheap6_no_GlobalPIQA +0.2721; cheap5 +0.2440; EWoK_plus_Entity_sum +1.3400.
- plain no-disentangle compact-minus-repeat: cheap6_no_GlobalPIQA -0.4188; cheap5 -0.3640; EWoK_plus_Entity_sum -1.7800.
- interaction nodis-minus-full: cheap6_no_GlobalPIQA -0.6908; cheap5 -0.6080; EWoK_plus_Entity_sum -3.1200.

The result is important but confounded because initialization parity and interaction interpretation showed that plain no-disentangle changes common initialization (53/140 same-named tensors differ) and capacity (34.47M -> 30.77M). This motivated the common-copy disambiguation experiment:

- Common-copied no-disentangle compact: `experiments/archive/frontier_consolidation/training/runs/commoncopy_nodis_compact_deberta100M_seed43022`.
- Common-copied no-disentangle repeat: `experiments/archive/frontier_consolidation/training/runs/commoncopy_nodis_repeat_deberta100M_seed43022`.

These tasks must complete before solidifying any unified mechanism claim.

## Readout harness created

Two scripts were created and syntax-checked:

1. `experiments/archive/frontier_consolidation/scripts/commoncopy_integrity_reader.py`
   - File-only.
   - Checks `common_copy_initialization.json`: source `p2c,c2p`, target `[]`, target 30,773,344 params, after-copy 140/140 common tensors exact, only-left 32 positional projection tensors, training-build patch mode.
   - Checks terminal `scientific_metrics.json`: 100,000,000 words, 2,529 steps, legal tokenizer label, WWM 0.15, DeBERTa 8x480, batch256/seq256, finite losses, ten checkpoints.
   - Checks `chck_80M` and `chck_100M` model/config files.

2. `experiments/archive/frontier_consolidation/scripts/commoncopy_architecture_interaction_readout.py`
   - Thin wrapper around the hardened earlier analysis panel.
   - Emits the same canonical arm names expected by earlier analysis bootstrap: `full_compact`, `full_repeat`, `nodis_compact`, `nodis_repeat`.
   - Patches `nodis_compact`/`nodis_repeat` to point to the common-copied runs.
   - Reuses full compact and full repeat per-target selected payloads from architecture interaction result/legal tokenizer clean control trajectory design/compliant tokenizer shift and eval readiness.
   - After common-copy completion, the intended commands are:

```bash
python3 -B experiments/archive/frontier_consolidation/scripts/commoncopy_integrity_reader.py \
  --out-dir experiments/archive/frontier_consolidation/data/commoncopy_integrity_final

python3 -B experiments/archive/frontier_consolidation/scripts/commoncopy_architecture_interaction_readout.py \
  --out-dir experiments/archive/frontier_consolidation/data/commoncopy_architecture_interaction_selected_panel \
  --checkpoints chck_80M chck_100M

python3 -B experiments/archive/frontier_consolidation/scripts/architecture_interaction_interval_driver.py \
  --panel-dir experiments/archive/frontier_consolidation/data/commoncopy_architecture_interaction_selected_panel \
  --out-dir experiments/archive/frontier_consolidation/data/commoncopy_architecture_interaction_pair_intervals \
  --checkpoints chck_80M chck_100M

python3 -B experiments/archive/frontier_consolidation/scripts/architecture_interaction_bootstrap_analyzer.py \
  --panel-dir experiments/archive/frontier_consolidation/data/commoncopy_architecture_interaction_selected_panel \
  --out-dir experiments/archive/frontier_consolidation/data/commoncopy_architecture_interaction_four_cell_bootstrap \
  --checkpoints chck_80M chck_100M
```

`--plan-only` was run for the panel. The plan currently shows full compact/repeat ready and common-copy missing metrics/checkpoints, as expected while the H100 tasks are unresolved.

The current file-only integrity read is not terminal evidence. It shows both common-copy run dirs already contain valid `common_copy_initialization.json` records: training-build patch mode, source full `p2c,c2p`, target `[]`, target params 30,773,344, after-copy 140/140 exact, only-left 32 positional-projection tensors, initial full-vs-target logit mean_abs 0.0024677. It reports missing terminal metrics and 80M/100M checkpoints, so no selected readout should run yet.

## Conditional interpretation after common-copy

- If common-copy no-disentangle still has compact-minus-repeat collapse on stable families, the unified direction becomes stronger: structured semantic compression gives downstream sample efficiency only when the learner has an inductive path that can bind recurring shared content to changed relational/occurrence roles; DeBERTa's p2c/c2p positional-score machinery is one such path.
- If common-copy recovers full-like compact-minus-repeat, the plain architecture-interaction collapse was mostly initialization/optimization-basin dependence. The explanation should be rebuilt around how compact views exploit a fragile basin/credit-assignment coordinate rather than a necessary positional-score role-binding path.
- If common-copy is intermediate, both removed positional-score path and initialization/capacity/basin contribute; the next mechanism experiment should separate score path from parameter count only if it changes the broader sample-efficient-learning principle.

## inheritance and paired-world substrate

initialization parity and interaction interpretation showed raw-token memory variants failed while hard-coordinate memory works. Positive residue: the update/read mechanism works when occurrence roles are supplied; the bottleneck is latent role assignment from raw text.

commoncopy architecture interaction result found source-attested independent paired worlds at scale: tennis 35,715 reversed unordered player pairs, BWF 6,162, LaLiga 149 seed pairs, and football data likely thousands more. These provide independent contexts where an asymmetric role relation flips: `R(A,B)` in one event, `R(B,A)` in another, with source-attested labels. The risk is shortcuts: visible scores can solve sports outcomes by numeric comparison, and sports `defeated` is one narrow relation family.

## CPU paired-world construction probes

I wrote:

- `experiments/archive/frontier_consolidation/scripts/paired_world_shortcut_pilot.py`
- repaired version `experiments/archive/frontier_consolidation/scripts/paired_world_shortcut_pilot_v2.py`

The first pilot produced score-visible and score-ablated LaLiga packets, but direct inspection revealed a possible template-label proxy because each family/context used only one template.

The v2 pilot fixes that by fully crossing every family/context with all four relation templates:

- output dir `experiments/archive/frontier_consolidation/data/paired_world_shortcut_pilot_v2`
- 149 families, 2 contexts/family, 4 templates/context, 2 variants = 2,384 context packets.
- score_ablated: 1,192 packets, template counts exactly 298 each, each template appears with 149 `team_a_over_team_b` and 149 `team_b_over_team_a` entailed directions, max template-direction delta 0.
- winner_first_rate 0.5, team_a_entailed_rate 0.5.
- exact query verb `defeated` absent in contexts (rate 0.0).
- team_a/team_b count equality 1.0 and winner/loser count equality 1.0.
- directed hypotheses are bag-of-words identical (rate 1.0).
- score_digit_sentence_rate 0.0 for score_ablated; digits remain only in neutral date/season anchors (`any_digit_rate` 1.0), so a future version may remove dates too if date anchoring becomes a shortcut.
- same-template C1/C2 bag-of-words Jaccard after names/numbers normalization is exactly 1.0, meaning the two contexts differ only in which team occupies winner/loser positions in the relation sentence and source event anchors.

This is not a BabyLM training corpus. It is a shortcut-resistant test object for a later small model probe: the model must use order-sensitive predicate-argument binding, not word counts, query-word overlap, template identity, or score comparison. Remaining limits: one relation family, deterministic template language, sports domain, and dates/seasons still present. Future expansion must mix tennis/BWF/LaLiga/football and at least several non-sports asymmetric relation families before any serious pretraining claim.

## Next scientific sequence

1. Do not poll the common-copy H100 tasks. Wait only when the next decision truly depends on them or when the runtime delivers terminal results.
2. After delivery, run the integrity reader, selected panel, interval driver, and four-cell bootstrap above.
3. Only then choose the next route:
   - persistent common-copy attenuation -> design one shortcut-resistant paired-world model test of the joint structured-compression/role-binding mechanism across at least DeBERTa and one architecture that previously failed compact transfer, rather than further subdividing DeBERTa;
   - common-copy recovery -> redirect theory toward initialization/optimization-basin dependence and ask what data/architecture coordinates make compact views usable;
   - intermediate -> decide whether a clean parameter-count/score-path follow-up is worth more than the broader paired-world role-binding test.

No GPU work, selected scoring, SuperGLUE, AoA, upload, leaderboard submission, or final-facing deliverable was performed.
