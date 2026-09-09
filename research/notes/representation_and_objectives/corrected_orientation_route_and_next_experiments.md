# corrected orientation route and next experiments — Corrected orientation route and next experiments

## Why this note exists

complete orientation probe results produced a real behavioral phenomenon on the repaired equivariant symmetry repair and macro context equivariant substrate, but its initial interpretation was too strong. The central symmetry-identification prediction was **opposite aligned vs inverted orientations on unbridged mixed held-seen relations**. That did not occur:

- fine-tuned aligned mixed held-seen: **0.4922**;
- fine-tuned inverted mixed held-seen: **0.4992**;
- aligned-minus-inverted: **-0.0070**.

The corrected durable result is narrower: state-formatted supervision gives out-of-distribution state-query competence on new names/templates, while relation-coordinate orientation and held-held composition do not transfer under this protocol.

Canonical corrected readout: `research/documents/representation_and_objectives/data/orientation_evidence_readout/orientation_evidence_readout_summary.md`.

## What complete orientation probe results actually supports

From the corrected orientation route and next experiments readout of saved results only:

1. **Formal identifiability remains real in the slot-orbit solver.** The equivariant symmetry repair and macro context symbolic assignment set has the intended algebra: held-held-only and neutral preserve the two global role-swap assignments; aligned state evidence identifies the true assignment; inverted identifies the complement; mixed event evidence identifies true. This shows the text surface can represent the role-coordinate problem when a supplied slot parser is available.

2. **Neural DeBERTa does not propagate the identified coordinate.** Full fine-tuning fits training rows well, but all mixed held-seen readouts remain at chance. Even `mixed_event_bridge`, which directly trains relation-comparison rows, scores **0.4926** on held-out mixed relations despite train accuracy 1.0.

3. **State-query transfer is real but not cleanly polarity-controlled.** Fine-tuned state arms improve true-labeled changed-state readouts: aligned **0.7945**, inverted **0.7375**, neutral **0.5656**. The aligned-vs-inverted changed difference is only **+0.0570** and has unstable seed-paired values `[+0.188, +0.012, -0.086, +0.051, +0.121]`. Unchanged and joint readouts often favor inverted.

4. **Neutral is not a matched calibration reference yet.** Neutral has 704 supervised rows and 35,200 row-presentations at 50 epochs, compared with 320 rows and 16,000 presentations for aligned/inverted. It cannot be used to quantify the state-format contribution without a new row/update-matched run.

5. **Aggregate train accuracy does not prove local contradictory semantics.** In aligned/inverted arms the mean train accuracies are ~0.91 over 192 relation rows + 128 state rows. Even assuming perfect relation-row fit, aggregate accuracy only constrains state-query training fit to ranges: aligned `[0.7735, 1.0]`, inverted `[0.7860, 1.0]`. We have not yet measured whether inverted changed rows are locally coherent with preserved unchanged facts.

6. **Both-correct is not a protected state record.** State arms raise changed+unchanged both-correct above the no-state baseline (~0.25), but both-correct is close to the minimum overlap forced by marginal accuracies and below the independence reference. This is state-task transfer, not conserved event-state binding.

## Head-free cloze readout across pretraining checkpoints

I ran an inference-only DeBERTa MLM cloze owner preference on the equivariant symmetry repair and macro context state rows at four checkpoints. It scores prompts of the form `premise + After the event, [MASK] had the object.` by comparing candidate-name logits. This is not an NLI classifier; it is a direct owner-prior readout before supervised complete orientation probe results fine-tuning.

Files: `experiments/archive/representation_and_objectives/data/mlm_prior_state_readout`.

| checkpoint | overall groups | paired eval acc | cross-template eval acc | name-permutation eval acc | reading |
|---|---:|---:|---:|---:|---|
| chck_10M | 896 | 0.469 | 0.469 | 0.500 | chance / slight below |
| chck_40M | 896 | 0.469 | 0.484 | 0.469 | chance / below |
| chck_80M | 896 | 0.484 | 0.484 | 0.516 | chance |
| chck_100M | 896 | 0.492 | 0.477 | 0.555 | mostly chance, weak name-perm drift |

This weakens the idea that complete orientation probe results simply exposed a directly visible MLM owner prior. If a pretrained/default bias is involved, it is not available to this head-free cloze readout on the nonce held-state rows. The state gain may instead be created by state-format fine-tuning, surface grammar, optimization, a nonlinear/token-level latent feature, or a prior that is only expressed after supervised adaptation.

## Literature grounding added in corrected orientation route and next experiments

- `\cite{zhang2024unforgettable}` shows that contradictory or random-label fine-tuning can fit training examples while generalization outside the training set varies by task; retained behavior often reflects shallow behavioral change rather than erased capability. This is close to the local-override vs out-of-distribution-default question, but their tasks are broad multiple-choice benchmarks, not role-coordinate orientation.
- `\cite{geirhos2020shortcuta}` frames the correct caution: performance on a dataset should not be attributed to a high-level ability when shortcut, format, loss, optimization, or experience-induced biases can explain it. This directly supports keeping the complete orientation probe results state gain separate from role-coordinate induction.
- `\cite{mszros2024rule}` studies OOD rule extrapolation and emphasizes architecture- and rule-dependent simplicity priors. It offers a useful conceptual bridge: underspecified data select among simple hypotheses, but complete orientation probe results has not yet shown which hypothesis DeBERTa selected.

## Current scientific interpretation

The best current research object is not established selective frame activation. It is:

> Under limited, structured supervision, DeBERTa can acquire state-query competence that generalizes beyond training names and templates, but the same evidence does not induce a reusable absolute role coordinate or propagate bridge polarity to unbridged relation comparisons. The source of the state competence remains unresolved among state-format calibration, optimization/credit assignment, surface grammar, and pretraining-shaped defaults.

This still serves the BabyLM data-efficient learning goal because it identifies a precise failure mode of data-efficient generalization: sparse data can train the answer format and local first-order consequence without creating the more abstract, composable coordinate needed for transfer across relation components. But it is not yet a general principle until the unresolved alternatives are separated.

## Next experiments that are justified before any expression

Expensive work must decide between routes. The next work should use the repaired equivariant symmetry repair and macro context substrate and minimal reliable tests before any H100-scale BabyLM training.

### 1. Fine-grained complete orientation probe results training-set and support-distance audit (CPU from saved checkpoints if checkpoints exist; otherwise a cheap rerun that saves logits/checkpoints)

Purpose: test whether inverted evidence was actually learned locally as a coherent contradiction, and how far any contradiction generalizes.

Readouts:
- split training accuracy and margins by `task`, `query_kind`, `relation`, voice, candidate slot, and changed/static;
- inverted arm: exact bridge rows, same names/new objects/templates, same entity pair role-swapped, familiar entities in new pairings, one familiar/one new, two new;
- signed state orientation: true-owner margin minus inverted-owner margin;
- unchanged conservation and joint true/inverted changed+unchanged.

Route decision:
- if inverted rows were not locally coherent, complete orientation probe results is optimization/local-fit evidence only;
- if inverted is coherent only on exact rows, it is memorization;
- if inverted follows training entities but not new entities, it is entity-attached override;
- if inverted is coherent locally but true-direction or chance on new entities, then prior/default out-of-distribution behavior becomes a real target phenomenon.

### 2. Row/update-matched aligned, inverted, neutral, and random-label state-format forks

Purpose: distinguish bridge polarity from generic state-format calibration.

Use identical initial weights/head seeds, identical held-held rows, matched numbers of state rows, matched label/candidate/voice/static distributions, and matched update counts. Include:
- aligned held-state bridge;
- inverted held-state bridge;
- neutral seen-state rows with held distractors, downsampled/matched to 128 state rows;
- random-label state rows as a stress control if they fit.

Route decision:
- if neutral/random reproduce the same positive OOD state orientation, the complete orientation probe results phenomenon is generic state-format/default activation;
- if aligned and inverted create opposite signed state orientation at matched conservation, polarity matters;
- if only state format improves unchanged but not changed orientation, it is query/candidate calibration rather than event semantics.

### 3. Fully trainable random-init control matched on local competence

Purpose: distinguish pretraining from architecture/optimization. This is only interpretable if random-init reaches local state competence comparable to pretrained.

Use the same matched fork design, but from random DeBERTa init. If it cannot fit local bridge rows cheaply, use a smaller trainable transformer with the same text interface as a positive/negative computational control, not as direct BabyLM evidence.

Route decision:
- random learns global inverted orientation while pretrained does not: pretraining prior resists reversal;
- random and pretrained both show the same default: architecture/surface/optimization dominates;
- random never fits local state rows: it cannot decide prior vs no-prior, but shows pretraining supplies the usable state-task scaffold.

### 4. Contradictory-evidence dose and diversity sweep

Purpose: test whether enough contradictory evidence can globally reverse orientation or only produce local exceptions.

Vary inverted evidence by unique relation/name/object/template orbits (not just row repetitions), with total presentations held constant using neutral fillers. Save checkpoints and track signed orientation, conservation, and mixed relation orientation.

Route decision:
- repetition only deepens local fit while OOD stays default/chance: orbit diversity, not token count, governs reusable updating;
- broad diverse inverted evidence crosses signed orientation below zero while conserving unchanged facts: global reversal is possible;
- no crossing despite local coherent inversion: strong default/basin resistance in this coordinate;
- crossing only when conservation collapses: output remapping rather than semantic reversal.

### 5. register connection only through the same probe

shared factorization result synthesis prediction is a fixed-budget distribution-proximity/register-substitution prediction, not automatically this mechanism. Their committed future contrast is `childspeech_removed - adultprose_removed`, predicted positive by non-open-lexical profile distance (exEntity5 about +0.748 common10_80) after word-unigram JS failed as a negative control.

Mechanistic link to companion analysis should be narrow. If register checkpoints produce a real register divergence, test them with the same equivariant symmetry repair and macro context/280 orientation-and-conservation probe. Aggregate BabyLM or exEntity movement alone does not support the role-coordinate route.

## Proposed next experiments

The next proposed work is the matched-fork comparison and support-distance audit, beginning with the lowest-cost test:

1. modify or extend the complete orientation probe results fine-tuning script so it saves per-row train/eval logits and optionally final checkpoints for a small subset of arms;
2. run a short matched pretrained pilot for aligned/inverted/neutral with equal state-row counts and equal update counts on the available GPU if memory is available; otherwise run CPU-only audit of existing outputs and prepare the runnable script;
3. only after the matched pretrained pilot, launch trainable random-init or dose sweeps.

Do not write final synthesis around selective activation yet. The current contribution is a sharpened problem and a corrected set of discriminating experiments.
