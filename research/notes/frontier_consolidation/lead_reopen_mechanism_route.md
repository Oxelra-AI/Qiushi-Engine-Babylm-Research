# lead reopen mechanism route Lead synthesis: leaving likelihood/edit routes and rebuilding the next mechanism

## Current score-bearing state

The public protected endpoint is still scale1.75 `chck_82M`:

- local hardened Overall: `41.942481167385985`
- public displayed Overall: `41.94`; public row remains present in the live Strict-Small surface refreshed in context dependence ig probe result
- HF repo: `leslie721007/babylm-strict-small-scale1p75-chck82`, revision `f49775dc5eafbf3a14d6f2107ec5368188d2b1dd`
- trusted custom-code loading is required; native DeBERTa loading is the wrong function

The strongest local endpoint object remains coherent86 alpha0.75:

- cheap7: `44.18142857142857`
- SuperGLUE: `69.81922238969935`
- Overall(AoA0): `42.1210247099666`
- truthful carrier: `data/truthful_private_scale_carriers/coherent86_alpha0p75/all_full_preds_truthful_coherent86_alpha0p75_mlm.json`
- carrier SHA256: `40181994810e21bc823474a3e4ac84c8eb42213e03904d36a60a4698477d1994`
- endpoint interpretation: no-training amplitude-scaled coherent private residual, not a separately trained mechanism

## Closed Routes

edit state probe chck82 synthesis closed source/edit correspondence: at `chck_82M`, detached private readout still decodes true-vs-decoy source-absent edit information (`+0.7464` held-out), but raw true source remains harmful (`-0.1756`), the signal is content-word dominated, and static decoy contingency validation already showed mature source-free transfer fails. This is not a path to relation/state competence.

context dependence ig probe result and earlier analysis close the likelihood target-allocation line. Context dependence is real, but high-context-dependence tokens are already easy at `chck_82M`; the hard full-context residual is low-context-dependence. In matched clean80→reinvest80 probes, reinvestment strongly improves likelihood in the changed compact block, especially low-IG/high-error items, but this is a local compact-view likelihood signature. In direct official EWoK/Entity flip scoring, relation/state losses worsen from 82M to 100M, yet they are not the low-IG/high-error population. Therefore another low-IG mining or mask-reallocation trainer would not attack the current scientific obstruction.

The private-scale family remains endpoint engineering. It improves the scalar score but moves along one amplitude-controlled direction with mixed EWoK/Entity behavior and small GlobalPIQA example count.

## External source anchors read

The new source reading supports a stronger but more demanding route:

- Counterfactual data augmentation literature emphasizes that useful counterfactual/invariant data requires a known causal or structural object, not arbitrary swaps; failures come from semantic drift, incomplete context enumeration, and shortcut exploitation.
- `BabyLM's First Constructions` (`\cite{rozner2025babylma}`) shows BabyLM-scale models can learn constructional form-meaning constraints and that construction-sensitive probability patterns correlate with BabyLM benchmark scores. This supports probing construction/relational constraints directly rather than relying on global corpus NLL.
- ECONET (`\cite{han2021econet}`) demonstrates event/temporal continued pretraining by targeted event/temporal masks and contrastive signals, but it is close to risky targeted masking; broad target reallocation and contrastive/token objectives repeatedly damaged broad competence. We should inherit the focus on event-temporal information, not the same objective form.
- Entity tracking work (`\cite{kim2023entitya}` and `\cite{kim2024code}`) shows entity state tracking is learnable but natural text alone often does not make it surface; code/structured procedural data helps, and nontrivial state-tracking tests must avoid initial-state copying, empty-state shortcuts, lexical overlap, and slot memorization.
- Procedural-knowledge work (`\cite{ruis2024procedural}`) suggests reasoning may be driven by documents that demonstrate reusable procedures rather than answer retrieval; for BabyLM this points toward compact, repeated, varied structural experiences instead of more examples of the same surface form.
- Reasoning Core (`\cite{lacombe2026reasoning}`) supports the value of solver-verified, distributionally broad procedural data, but BabyLM Strict-Small legal rules mean any new text must count inside the 10M/10-epoch budget and cannot be an unaccounted external corpus.
- Synthetic-data survey (`\cite{long2024llms}`) reinforces that faithfulness and diversity are the central failure modes for generated training data.

## Source-only feasibility work done here

Two CPU-only legal-corpus scans were created:

1. `scripts/relation_marker_yield_audit.py` and `data/relation_marker_yield_audit/`. This broad scan over the exact legal compact-view reinvest pool (SHA `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`) found enormous relation-marker yield: 64,740 rows, 10M words, and broad markers in most rows. The result proves availability but is too permissive to select examples.
2. `scripts/structural_packet_candidate_selector.py` and `data/structural_packet_candidates/`. This excludes generated compact/rewrite rows and selects 400 source-only candidates balanced across five broad families: event-temporal-causal, entity-state-update, quantity-change-compare, social-belief-report, and polarity-contrast-event. Source mix is mostly Gutenberg/OpenSubtitles/SimpleWiki/BNC. Representative snippets show feasible material but also many risks: dialogue fragments, translation noise, obvious connective cues, quoted narrative fragments, and rows that would require careful proposition verification.

The scans support development of a small verified packet, not training. Raw marker counts are not evidence that relational competence will improve.

## Strongest next mechanism direction

The most promising route is not “more compact views” and not “mask more relation words.” It is:

**Graph-preserving relational experience:** convert a small number of legal source rows into compact, verified multi-view records that preserve an explicit relational/event object

`G = (entities, roles, initial state, action/event, resulting state, temporal/causal/quantity/social relation, polarity/modality/attribution)`.

Each record should have several views that express the same `G` with different wording, plus a separately coherent one-edge-changed version. The scientific hypothesis is that small models may need repeated, varied, compact experiences of the same latent event-state structure so that roles, entity bindings, temporal order, causal consequence, quantity change, and modality become reusable abstractions rather than isolated surface predictions.

This is distinct from closed routes only if the invariant object is verified. A naive paraphrase packet would just repeat the existing compact-view mechanism and could improve local NLL without moving relation/state competence.

## What must happen before any generation-scale or H100 training

The next work should build a tiny source-derived packet from the lead reopen mechanism route candidates, roughly 50-100 rows, with verified records and paired views. This is a proposed data-construction comparison, not a training result.

For each selected source row:

- extract a minimal `G` record in JSON: entities, roles, event/action, state before/after or premise/consequence, temporal/causal/quantity/social relation, polarity/modality/attribution, and source span;
- write 2 faithful compact views preserving `G`;
- write 1 entity-renamed view preserving roles and relation;
- write 1 one-edge-changed coherent counterpart that changes exactly one relation, role, order, polarity, quantity, or state while remaining internally coherent;
- write 2-3 cloze or pseudo-likelihood probes that require the invariant relation, not merely marker completion;
- include a same-source ordinary compact paraphrase baseline of similar length.

Before training, run a frozen `chck_82M` reader over the packet:

- correct preserved view versus one-edge-changed counterpart margin;
- cross-view transfer: does a compact view make the target in another view more predictable than an unrelated same-source view;
- entity-renaming robustness: margins should survive name changes;
- marker removal sensitivity: if removing obvious connectives destroys all signal, the packet is too superficial;
- compare with ordinary compact paraphrase and repetition-matched baselines.

This measurement should use no official items and no new pretraining. If the true graph-preserving views do not beat ordinary compaction and one-edge-changed controls on source-held-out records, stop the route before generation scale-up. If they do, a small 4M-8M matched screen can be designed later with identical source rows, identical word exposure, ordinary-MLM control, ordinary compact multi-view control, graph-preserving multi-view arm, and one-edge-changed coherent control.

## Alternative mechanism to keep alive: relational gradient subspaces

independent_review also proposed a different path: build source-only structural tasks and measure signed gradient geometry against the actual `82M -> 100M` displacement. This is attractive because it directly addresses late interference: ordinary likelihood improves while EWoK/Entity gold likelihood worsens. It differs from the already-null squared-gradient importance work by using signed task-gradient subspaces rather than per-parameter magnitude.

This comparison remains conditional on a verified packet. The packet can provide the structural tasks. The cheap measurement would compute small randomized gradient sketches at `chck_82M` for true structural views, one-edge-changed views, and matched ordinary MLM targets, then test whether projection onto `theta_100M - theta_82M` predicts held-out structural margin movement. If stable, it could motivate projected-update training rather than more data. But it should not be built before a verified source packet, because noisy marker-selected objectives would repeat previous failures.

## Next data-construction test

The proposed construction would build `data/graph_packet_v0/` from the lead reopen mechanism route candidate JSONL, starting with 40-60 high-quality Gutenberg/SimpleWiki rows rather than the noisiest OpenSubtitles/BNC rows. The output should be a JSONL packet with source spans, verified `G` records, faithful views, entity-renamed views, one-edge-changed coherent counterparts, and same-source ordinary compact baselines. Also write a short validation note naming rejected examples and why. No H100 training and no leaderboard action.

The next proposed measurement is a frozen `chck_82M` packet reader. Only if graph-preserving views show transfer beyond ordinary compaction and one-edge-changed controls should a small matched-exposure training screen be considered.
