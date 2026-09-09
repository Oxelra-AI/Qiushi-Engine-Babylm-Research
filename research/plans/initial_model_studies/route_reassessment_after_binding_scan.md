# earlier analysis — route reassessment after the binding-scan critique

## Correction to the binding feasibility and route implications interpretation

The binding feasibility and route implications scanner is a **lower-bound representation test**, not a closure of cross-entity credit assignment in the official corpus.

It only searched for a narrow construction:

- capitalized entities;
- short within-window patterns;
- rule predicates such as simple verb/preposition/object relations;
- downstream references in the same segmented window.

Interpretation boundary: sparsity under this construction does **not** prove that the official corpus lacks binding-dependent learning signal. It proves that the current representation is too narrow. It also does **not** justify attributing the public leader's Entity/EWoK advantage directly to gated FineWeb. That remains a hypothesis, not a proven cause.

What remains valid from structure density conclusion and objective pivot:

1. **Plain WWM on lexical/entity-state dense windows did not improve Entity or EWoK.** This is supported by the high_entity_state vs matched_low vs uniform training screen.
2. **The source-swap name probe in source swap antecedent probe was a copy-sensitivity probe, not a binding probe.** The target word's presence in context changed, so the positive result cannot localize entity-state credit assignment.
3. **The first binding scanner found little signal under one narrow representation.** It gives a lower bound and exposes representation weakness, not a final conclusion about official-corpus binding signal.

## Updated scientific question

The load-bearing question is now:

> Can we construct enough target-identical, bag-preserving, binding-dependent training signal from legal text — including cross-sentence context, pronouns, lowercase nominals, aliases, and event/state predicates — such that the model must predict downstream state/relation content from entity–property/location/action binding rather than from copying the answer token?

A valid binding example must preserve the same words/entities on both sides of the comparison and only swap the assignment:

- Correct: `Alice put the red cup in the box. Bob put the blue cup on the shelf. Later Alice picked up the red cup from [box].`
- Swapped control: same entities and state words, but red-cup→box / blue-cup→shelf binding is swapped before the downstream target.
- Target: a state/relation token such as `box`, not the repeated entity name.

## Route comparison

### Route A — Broader official-corpus binding substrate audit

Scientific role: determine whether the official corpus still contains enough usable binding-dependent signal after representation reconstruction.

Needed improvements over binding feasibility and route implications:

- cross-sentence windows over adjacent sentence pairs/triples, not only line/window-local patterns;
- pronoun/deictic chains (`he/she/it/they/this/that/the former/the latter`) and aliases, not only repeated capitalized words;
- lowercase entity-like nominals (`the boy`, `the girl`, `the red cup`, `the other one`) and object-property phrases;
- state predicates beyond simple verb-preposition-object patterns: possession, birth/death, membership, occupation, causality, temporal order, containment, motion, transfer, and attribute assignment;
- bag-preserving swap construction and target-identical likelihood probe, not merely counts;
- quality estimation by manual examples and probe deltas under the protected model.

Decision rule:

- If the reconstructed substrate produces a large number of high-quality bag-preserving binding probes and protected-model likelihood is binding-sensitive, build a target-identical training objective.
- If it remains sparse/noisy after this richer representation, do not keep repairing official-corpus binding heuristics; shift to legal binding-rich data and/or hybrid objective.

### Route B — Legal binding-rich data

Scientific role: provide the missing experience distribution if the official corpus is too weak under a proper representation.

This should not mean copying the gated leader or assuming FineWeb is the cause. It should mean legally sourcing or constructing text whose mechanism is explicit:

- accessible encyclopedic/factual text with entities, aliases, and attributes;
- procedural/event text with entities and state transitions;
- public synthetic-but-legal binding-rich microcorpora, only if the internal binding probe shows the model can learn the intended relation and transfer is tested against official evaluation;
- possible mixture with official data so grammar/Reading/Supplement advantages are preserved.

The strongest data route is not generic simplification; WikiAuto alignment failed. The data must carry binding, relation, and world-knowledge structure that official evaluation can use.

### Route C — GPT-BERT/MNTP or AntLM-style hybrid objective

Scientific role: an independent, literature-grounded route that attacks information flow and dense token supervision across the whole corpus, not only rare binding examples.

Evidence recovered this step:

- GPT-BERT unifies MLM and CLM through masked next-token prediction (MNTP): if token `k+1` is masked, output at `k` predicts it. It reports low-ratio causal/masked mixtures (final 1:15) and strong BabyLM 2024 Strict-Small scores.
- GPT-BERT shows that adding as little as 6.25% MNTP improved bidirectional performance in its setting.
- AntLM independently reports alternating CLM/MLM improvements, with LTG-BERT 10M BLiMP/EWoK gains in the 2024-style suite.
- Earlier analysis identified a careful hybrid branch as a live route, with the essential no-future-leak test and objective-identity test.

Risks:

- RecGPT shows causal/sequence modeling can boost BLiMP/COMPS/GlobalPIQA while leaving Entity weak.
- A hybrid route must not become only a non-Entity score chase. It must monitor Entity and EWoK early, alongside BLiMP/Supplement/Reading damage.
- Implementation must pass causal attention and MNTP identity tests before training.

## Updated route decision

Do **not** close official-corpus cross-entity credit assignment from binding feasibility and route implications. Instead, treat binding feasibility and route implications as evidence that the old uppercase short-window representation is too narrow.

The next best work is a two-asset execution sequence:

1. **Build a broader binding-substrate audit/probe** on the official corpus using cross-sentence, pronoun, lowercase-nominal, and bag-preserving binding-swap criteria. This directly answers the representation concern and prevents premature abandonment of official-corpus binding mechanisms.
2. **Prepare the GPT-BERT/MNTP hybrid trainer with no-leak tests** as the main independent SOTA route, because it is well supported by prior BabyLM literature and affects all tokens, not just rare binding examples.

If execution budget forces a single immediate task, start with (1) only if it can be completed quickly as a real probe; otherwise start the hybrid trainer's no-leak implementation because it is a stronger route toward the 1.27 Overall gap and has not yet been tested in these experiments.

## What not to do

- Do not train the old entity-name anchored objective.
- Do not train the current random `relation_broken_wwm` control.
- Do not assert the leader's advantage is due to gated FineWeb without source-level evidence.
- Do not chase only BLiMP/COMPS/GlobalPIQA while letting Entity/EWoK stagnate; the active goal is Overall SOTA through a real data-efficient learning principle.
