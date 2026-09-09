# Candidate mechanisms after evidence synthesis

## Purpose

Choose the next mechanism-level route after the step100 204 systematic compression compression. The load-bearing unknown: induce reusable entity/relation/state variables from legal BabyLM experience so Entity/EWoK (and GlobalPIQA) improve together while preserving protected Supplement/Reading. Not another local tweak.

## Decisive evidence read this step

- Entity Tracking task (`kim2023entitya`): worlds of boxes/objects; operations move/remove/put; predict box contents after k operations. Task is *deliberately built* so that (a) probed states do not follow pretraining distributional patterns, (b) individual words/phrases cannot predict the state alone, (c) train/eval have low lexical overlap, (d) not solvable by slot-filling. Text-only pretraining does not make tracking surface, but finetuned T5 does nontrivial tracking and generalizes to more operations and new entities. This is an *ordered operation composition over bindings* task, not a world-knowledge lookup.
- `prakash2024fine` (metadata): entity tracking is implemented by a circuit that tracks the *position of the correct entity*; math/code finetuning enhances it primarily by improving positional-information handling. Mechanism = positional binding, and it is trainable/enhanceable.
- `chan2026pretraining` (metadata): pretraining data statistics causally drive whether the model learns a frequency heuristic, a position heuristic, or the true relational variable. Flattening a Zipfian entity distribution toward uniform removes the frequency phase and makes the task learn faster. Small synthetic experiments predicted training dynamics in real OLMo pretraining.
- `winston2026learning` SAMBAL (metadata): constrained relexicalization can preserve syntax while ablating semantic/world cues; content-neutral models generalize and adapt with minimal exposure. Relexicalization is a real lever on what variables the model must use.
- `chan/kim/prakash` together explain our recurring failure: plain WWM on official text lets the model win by frequency/lexical/position surface cues, so it never has to build the binding variable. Our data-density and simplification routes added propositions but not the *distributional pressure* that forces the relational variable.

## Candidate route R1 — Legal synthetic state-dynamics experience with relexicalization and flattened entity frequency (primary candidate)

Hypothesis (falsifiable): The Entity/EWoK deficit is not a capacity or tokenizer problem but a *training-experience distribution* problem. If a fraction of BabyLM-legal pretraining experience is replaced by procedurally generated, natural-language state-dynamics passages that (i) require composing ordered move/remove/put-style operations over entity–container/state bindings, (ii) use relexicalized, near-uniform entity names so frequency and lexical shortcuts are unavailable, and (iii) keep answers unpredictable from local surface, then a standard DeBERTa/WWM model will learn a reusable positional-binding variable that raises official Entity Tracking and transfers to EWoK/GlobalPIQA, while preserving grammar/Reading.

Why this differs from closed routes:
- Not WESS: no gold addresses at inference, no auxiliary slot module, no export gap — the mechanism must live in the ordinary backbone because the *data* forces it.
- Not structure-density selection: we do not select natural windows; we *construct* experience whose distribution removes the shortcut, per `chan2026pretraining`.
- Not simplification pairs: the target is ordered operation composition and binding, not lexical redundancy.
- Not contrastive templates: the signal is next-token/masked prediction on natural passages, judged by official columns, not a bespoke pairwise objective.

Legality and provenance (must verify before training):
- Generated text counts toward the 10M word budget and against exposure limits; must account exactly.
- Must not use or leak the official Entity Tracking / EWoK evaluation items, vocab, or templates. Entities and surface forms must be independently sourced and relexicalized; evaluation-set isolation must be audited.
- Must stay within official Strict-Small rules for self-constructed corpora.

Decisive short-budget experiment (before any 100M run):
- Arms at 10M exposure, protected DeBERTa-v2 8×480 WWM, baseline16k, per-1M checkpoints saved:
  1. official-only baseline (fresh, checkpoints saved);
  2. official + X% generated state-dynamics experience (relexicalized, flattened frequency);
  3. official + X% generated state-dynamics experience with entity names *frequency-Zipfian and fixed* (shortcut-permitting control);
  4. official + X% *shuffled-operation* generated experience (destroys ordered composition; content-matched control).
- Decisive comparison: arm 2 minus arm 3 and arm 2 minus arm 4 on Entity Tracking and EWoK, with grammar/Reading preserved. Arm 3/4 controls separate "the mechanism is ordered binding under flattened distribution" from "generated text just helps generally."
- Include a frozen-model binding probe on held-out relexicalized items to check whether the learned variable is position/binding rather than lexical.
- Scale to 100M only if arm 2 beats both controls on Entity/EWoK without spending Supplement/Reading.

Risk: generated experience may raise a proxy (e.g., GlobalPIQA) without Entity/EWoK; the arm-3/arm-4 controls and the official Entity column are the guard. Data legality/isolation is the main gating risk and must be audited first.

## Candidate route R2 — Architecture bias toward abstract variables (secondary)

Inherit the ELC-BERT/LTG-BERT finding that architectural choices shifted models toward linguistic (not surface) generalization (higher MSGS). Test a backbone-level inductive bias (e.g., layer weighting / initialization / attention structure) intended to prefer abstract features. Deprioritized because: (a) our DeBERTa is already a strong architecture, (b) architecture-only changes historically moved GlobalPIQA not Entity/EWoK here, (c) the primary papers point more strongly to a *data-distribution* cause of the binding failure.

## Candidate route R3 — Adaptive/hard MLM on unmastered tokens (tertiary)

`fraser2025mask` reports AMLM helps mainly (Super)GLUE, and sub-token embeddings help morphology, not EWoK. Keep as a possible complement to R1, not a standalone SOTA route.

## Recommendation

R1 is the strongest post-compression route: it directly attacks the documented cause of the Entity/EWoK failure (surface/frequency shortcuts under plain WWM) with a legal, exportable, backbone-native mechanism (constructed experience distribution), and it has a clean decisive experiment with shortcut-permitting and order-destroying controls. R2/R3 remain as secondary/complementary.

Before construction, an independent critique should test: (1) whether R1's mechanism claim is sound given Kim&Schuster's explicit anti-shortcut task design and Chan et al.'s distributional-cause result; (2) the strongest data-legality/eval-isolation risks; (3) whether the arm-3/arm-4 controls truly isolate ordered binding under flattened frequency; (4) whether R1 should be tested on the protected 8×480 or S1-shape backbone; (5) the right X% mixing fraction and how to keep grammar/Reading.
