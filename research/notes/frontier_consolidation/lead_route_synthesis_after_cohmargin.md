# lead route synthesis after cohmargin Lead synthesis after coherence-margin closure

## Current score-bearing state

The public protected endpoint remains scale1.75 `chck_82M`:

- local hardened Overall: `41.942481167385985`
- public displayed Overall: `41.94`, persistent rank 1
- trusted custom-code path required; native DeBERTa fallback is the wrong function
- HF repo: `leslie721007/babylm-strict-small-scale1p75-chck82` revision `f49775dc5eafbf3a14d6f2107ec5368188d2b1dd`

The strongest local endpoint carrier is now coherent86 alpha0.75:

- cheap7: `44.18142857142857`
- SuperGLUE: `69.81922238969935`
- projected Overall(AoA0): `42.1210247099666`
- delta vs chck82 local: `+0.17854354258061278`
- carrier: `data/truthful_private_scale_carriers/coherent86_alpha0p75/all_full_preds_truthful_coherent86_alpha0p75_mlm.json`
- carrier SHA256: `40181994810e21bc823474a3e4ac84c8eb42213e03904d36a60a4698477d1994`
- scalar AoA=0 and no fast history; no upload/submission.

Scientific reading of the alpha family remains unchanged by the stronger endpoint arithmetic: it is endpoint redistribution along an amplitude-controlled coherent86 residual, dominated by COMPS/BLiMP churn and a very small GlobalPIQA-example contribution, not a general data-efficient learning principle.

## coherence margin signal isolation closure absorbed

The reopen structured context margin route/159 coherent-context margin route is closed. Its 4M-charged pilot scored cheap7 `36.692142857142855`, `-7.267307` below protected chck82, with severe BLiMP/Supplement/Entity damage. The training trace and NLL probes showed the intended coherent-over-disrupted likelihood separation did not appear: margin loss stayed near `0.799` and bad-minus-coherent NLL stayed around zero. Do not spend on a 20M continuation, lambda-zero official comparison, retention-KL, or variants of the block-shuffle negative construction.

## research exact-swap innovation WWM: valuable construction, not next H100 route

The completed research loop is mechanically strong:

- target pool: 26,315 source-absent rewrite innovation groups / 36,899 tokens per 10M pool;
- begins with exact baseline 15% Bernoulli WWM;
- proposes at most one innovation group per changed row;
- if not already selected, swaps it with an already-selected ordinary non-pair donor of identical token length;
- preserves selected group and token mass exactly per batch;
- preserves source and copyable pair selections exactly in smoke;
- projected extra innovation selections over 100M: about 25.5k, only ~0.174% of global selected-group mass.

This is distinct from the earlier analysis probability-reallocation run, which upweighted strict content innovations to ~0.5 probability and reduced copyable rewrite supervision. earlier analysis did prove the local target is trainable: strict target true loss improved by `-0.6476` at 80M and source help improved `+0.1532`. But earlier analysis broad transfer was decisively negative at 80M: mean7 `-1.0529`, BLiMP `-0.460`, Supplement `-2.720`, EWoK `-1.850`, COMPS `-0.750`, GlobalPIQA `-2.445`. Therefore optimizing the source-absent innovation target through ordinary MLM masks did not create broad BabyLM competence.

The research exact-swap could still answer a retrospective attribution question: whether earlier analysis damage was driven by copyable suppression or excessive pressure. But independent_review lead route synthesis after cohmargin and the actual evidence agree that this is low-upside relative to the SOTA goal. It should not receive a 20M/40M H100 screen now. It may be useful later as a bounded causal control if a new edit-state transfer mechanism needs a mask-only comparator.

## Routes already weakened by zero-training discriminators

The route reconstruction plan named three zero-training discriminators. Their actual token value private readout synthesis outcomes matter now:

1. **Natural intra-row discourse:** raw true-vs-shuffled context helped NLL, but true-vs-reversed was only `+0.018` and detached private true-vs-shuffled advantage was only `+0.0145` with CI crossing zero. This does not support discourse-state private learning as the next route.
2. **MLM-gradient importance vs displacement:** element-level correlations between spatial repair route status squared-gradient importance and endpoint displacement were near zero or negative for scale1.75, U256, and normal spatial repair route status late movement. Simple static importance damping is unsupported.
3. **Directed edit-state:** this is the surviving substrate. Full changed-span targets had raw true-source advantage `+2.7419` and private true-vs-decoy residual advantage `+2.8311`. In the stricter source-absent subset, raw true source was actually worse than decoy (`-0.3122` NLL advantage), but a detached private residual readout still extracted a strong true-vs-decoy advantage: held-out `+0.8365`, high-error `+1.2397`. This says useful edit information exists in hidden states but is not converted into the public token-likelihood path.

## Why the old dual-view/frozen-tail route does not close all edit-state routes

The source free transfer synthesis shared-readout probe suggested true source-conditioned and source-free views can transfer when a small shared residual readout is trained offline. The later train-time realizations failed for specific reasons:

- broad dual-view at 20M displaced main-stream words and let source-conditioned auxiliary training co-adapt with the backbone; aligned scored cheap7 `38.7629`, far below exact `mlm_only` `39.7864`;
- sparse separated aligned at 20M did show correspondence value (`40.2929` vs shuffled `40.0636`, and +0.5057 vs mlm_only), so separation and sparsity mattered at early state;
- mature frozen-82M aligned tail learned to use true source when present but failed source-free transfer: aligned was worse than shuffled by `+0.0180` NLL on true source-free view and below the protected endpoint in Overall.

Thus broad source-input auxiliary learning is closed. The remaining scientific object is narrower: **information transfer from a source/edit-aware view into a source-free ordinary token predictor**, tested with true-vs-decoy controls and target-piece removal. The new construction must not merely train a source-present auxiliary head or select an endpoint residual scale.

## Leading next hypothesis: information-removal edit distillation

Mechanism: create a transient source/edit-aware teacher for source-absent rewrite targets and train a source-free student/public MLM path to match only information that survives source removal.

A conceptual loss for source-absent target groups `I_absent` is:

`L = L_MLM + lambda * KL(stopgrad p_T(y_i | true source, masked rewrite, edit state) || p_S(y_i | rewrite-only or source-dropped context))`.

The teacher must be legal and corpus-derived: no external text, no evaluation data, no external teacher weights. It can be a small residual readout or pathway trained only on legal pair rows. Every target piece must be masked/removed from both teacher and student contexts; decoy-source and shuffled-edit controls must be capacity-matched. The final deployed function must be the normal source-free MLM path, not a source-present inference model and not an endpoint alpha selector.

This differs from research exact-swap because it does not just increase label frequency through the same ordinary MLM masks. It asks whether edit-state information present in hidden states can be distilled across an explicit source-removal intervention into the public predictor. It differs from broad dual-view because success is measured in source-free held-out prediction and the teacher is stopped/controlled rather than allowed to win only when source is present.

## Immediate next work before any training

Do **not** launch 20M/40M H100 training. The minimal decisive next step is a zero-training current-anchor discriminator:

1. Parameterize or copy `scripts/token_value_private_readouts.py` so it accepts:
   - `--checkpoint` / `--model-label` instead of the hardcoded spatial repair route status checkpoint;
   - `--trust-remote-code` for scale1.75 adapter checkpoints;
   - isolated writable HF/cache roots;
   - source-absent edit mode with the same item-building seed and doc/key split.
2. Rerun the source-absent edit-state readout on the current relevant functions:
   - protected `chck_82M`: `training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M`;
   - optionally coherent86 alpha0.75 model path only as endpoint comparison, not mechanism proof: `training/runs/coherent86_private_scale_0p75/hf_model/final`.
3. Compare against the existing spatial repair route status source-absent result:
   - raw true-vs-decoy NLL advantage;
   - held-out private true-vs-decoy advantage;
   - high-error advantage;
   - target-kind and overlap/changed-fraction strata;
   - whether the signal is still present on anchor errors rather than only on already-easy/copyable-like items.

What this decides:

- If chck82 has no robust source-absent private true-vs-decoy advantage, stop the edit-distillation route before trainer construction.
- If the advantage persists but raw true-source logits remain weak or negative, the representation-to-public-readout mismatch is confirmed at the actual score-bearing model; proceed to a frozen offline teacher-student transfer test before pretraining.
- If raw true-source logits are already strong and private advantage is small, the model already uses the source when present and a source-removal distillation may have little endpoint value.

## Second zero-training test if the current-anchor signal survives

Build a frozen offline teacher-student transfer test on held-out pair keys, not a pretraining run:

- teacher contexts: true source + masked rewrite + optional deterministic edit topology;
- false teacher controls: matched decoy source and shuffled/permuted edit topology;
- student context: rewrite-only or source-dropped standard row context;
- train teacher/readout and student on train-doc keys, evaluate student on held-out source-free keys;
- report source-free NLL/rank/top1 improvement on high-error source-absent targets, true teacher vs false teacher, and whether copyable/random-group matched auxiliaries reproduce it.

Only if the true-source teacher uniquely improves source-free held-out prediction beyond false-source/control teachers would the evidence justify a real 4M-8M charged continuation. That continuation must be small, matched, and judged by both the held-out edit target and official-compatible family movement. A cheap7 gain alone is not enough after the private-scale and coherence-margin experience.

## Candidate real-training form if both zero-training tests pass

A first legal real-training screen should be a 4M-8M charged continuation from `chck_82M` or a matched early scale1.75 checkpoint, not a new 100M run. It should include:

- ordinary MLM replay control at matched exposure/update schedule;
- true edit-distillation arm;
- matched decoy-source distillation arm;
- copyable/random-target distillation arm if feasible;
- identical word-exposure accounting for all additional views;
- target-piece removal and leakage asserts;
- private/source-aware teacher path discarded or disabled for official inference;
- public source-free MLM/student path evaluated.

Continue only if the true arm uniquely improves the frozen/source-free edit target and preserves or improves BLiMP/Supplement/EWoK/Entity/Reading relative to controls. Stop if decoy matches true, source-free target does not improve, or gains resemble alpha endpoint redistribution.

## Secondary hypotheses to preserve, not execute first

Other proposed alternatives include bottlenecked edit-state latent modeling, edit-conditioned gradient agreement, and edit-equivariant representation operators. These are valuable if the zero-training current-anchor readout remains strong but teacher-student distillation looks leaky or non-transferable. They remain alternative constructions, not launched as parallel training variants before the discriminator above is read.
