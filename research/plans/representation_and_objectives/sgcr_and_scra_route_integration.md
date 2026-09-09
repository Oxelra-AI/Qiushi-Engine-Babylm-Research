# source relation hard negative feasibility — SGCR boundary and SCRA route integration

This Explore step did not touch the managed SGCR jobs `s87_t29_tool1` and `s87_t37_tool1`, did not launch training, and did not run an official evaluation.  It integrated the live SGCR endpoint boundary with ewok contrast preservation anatomy's EWoK evidence and the new source relation hard negative feasibility source-relation hard-negative measurement.

## Current result state

No compliant SOTA endpoint is confirmed.  The known compliant endpoints remain below the visible 41.80 frontier: legal16k 8x480 at 40.704/41.024, legal40k 8x480 at 41.141/40.420, and legal40k 12x384 depth at 41.028.  The live differentiated endpoint is exact-prefix SGCR 12x384 K=50,d=64 seed43022, expected from the sgcr official span burden burden map to be most informative for COMPS and GlobalPIQA, not EWoK.

## What ewok contrast preservation anatomy added

ewok contrast preservation anatomy established that EWoK remains a large headroom column: leader EWoK 56.07 versus depth 50.547, a 5.52-point column gap.  Legal40 and depth are both wrong on 2,656/7,618 EWoK rows; all measured legal endpoints are wrong on 1,407 rows; many persistent errors use common context-difference words and do not concentrate low-support legal40 tokens.  The 132-row four-score probe sharpened this: depth persistent errors have mostly weak or negative context-target interaction (interaction-positive fraction 0.303, median -0.580), while all-legal-wrong rows are still worse (0.152 positive, median -0.674).  This supports a signed context-target interaction problem for depth, but does not by itself define a trainable mechanism.

## What source relation hard negative feasibility measured

The new CPU-only script `scripts/source_relation_hard_negative_feasibility.py` used the actual compact-view-reinvest 10M corpus and already-trained checkpoints.  It froze a small, general relation-alternative inventory before new evaluation reading, excluded very polysemous pairs such as `in/out`, `on/off`, `left/right`, `like/dislike`, `know/ignore`, and `can/cannot`, kept only sentences with exactly one safe relation term and no opposite term, and then scored the original relation term against its opposite under a masked source sentence.

Artifacts:

- JSON: `experiments/archive/representation_and_objectives/data/source_relation_hard_negative_feasibility/source_relation_hard_negative_feasibility.json`
- Candidates: `experiments/archive/representation_and_objectives/data/source_relation_hard_negative_feasibility/source_relation_hard_negative_candidates.csv`
- Margins: `experiments/archive/representation_and_objectives/data/source_relation_hard_negative_feasibility/source_relation_hard_negative_model_margins.csv`
- Note: `research/notes/representation_and_objectives/source_relation_hard_negative_feasibility.md`

Balanced capped run: 165 candidates (15 per group) after scanning 819 corpus rows / 114,478 words / 6,314 sentence fragments.  158/165 candidates were equal-token-span scoreable.  The existing models already prefer the source relation word in most cases:

| model | scoreable | positive margin | nonpositive margin | weak <= 1 nat | median margin | mean margin |
|---|---:|---:|---:|---:|---:|---:|
| legal40_depth_12x384_43022 | 158/165 | 0.778 | 0.222 | 0.342 | 2.195 | 2.159 |
| legal40_8x480_43022 | 158/165 | 0.810 | 0.190 | 0.373 | 1.884 | 2.018 |

Group details matter.  Affordance, containment, identity, possibility, and quantity are mostly already solved; spatial-containment, temporal-order, thermal, truth, and access contain more weak/negative margins but also visibly more noisy substitutions.  Sample inspection found grammatical and sense errors such as `allows -> prevents` producing `prevents us to confirm`, `different -> same` producing `same body compositions`, and `possible -> impossible` in idioms such as `as much as possible`.  Therefore broad single-slot hard-negative training would mostly reinforce lexical plausibility the model already has and would include noisy labels in the weakest groups.

## independent_review reading of this evidence

independent_review integration `data/external/independent_review01_verifier1_integration.md` converged on a strong judgment:

1. EWoK failure is not a rare-token-support phenomenon, so SGCR should not be expected to repair it by that mechanism.
2. The broad SCRA form, defined as preferring the observed relation word over an opposite word in the same source sentence, is not justified.  source relation hard negative feasibility measured exactly that single-slot game, and the existing checkpoints already win it at roughly 78–81% with around a 2-nat median margin.
3. The single-slot margin and the ewok contrast preservation anatomy four-cell interaction are different objects.  The model can know that a local source word is more plausible than its opposite while still failing to let the discriminating context steer target likelihoods in EWoK.  The broad objective would train the wrong object.
4. The surviving scientific target is the full interaction/specificity object: examples where the model locally knows the relation contrast, yet the context-target interaction is negative under official-like scoring.
5. A CPU-only full-EWoK measurement should precede any new training: four-cell interaction over all EWoK rows, a context-difference span margin, official correctness, summed and mean-normalized scores, context-difference deletion, and both legal40/depth checkpoints.

This moves SCRA from a broad candidate to a narrower unresolved mechanism: **interaction-specific relation anchoring**, not ordinary RTD, not single-slot hard negatives, and not relation-cue masking.

## Compliance reading

Official sources already in Knowledge support two load-bearing facts.  The BabyLM FAQ says any objective is allowed if data restrictions are followed, and that learned-on-language ancillary tools count toward the 100M word budget.  It also says synthetic data is allowed with accounting, and teacher models such as Qwen 3.5 up to 9B are approved for feedback/interactions.  The 2026 call states Strict-Small uses at most 10M corpus words and at most 100M words of counted exposure, normally 10 epochs, with required intermediate checkpoints.

Therefore:

- The lower-accounting single-input hard-negative form, which encodes the original corpus sentence once and compares the original span label against an alternative label, is probably the least ambiguous.  But source relation hard negative feasibility shows that broad single-input form is mostly redundant and targets the wrong object.
- The scientifically surviving full interaction-ranking form would encode a corrupted or paired sequence, so its exposure and generated-data accounting must be made explicit before training.  Conservative accounting should count every encoded corrupted sequence as exposure and should include any generated sequence in the dataset description.
- No SCRA-style training should launch until its exact exposure arithmetic and data provenance are written against the official wording.

## Route decision

Broad SCRA is closed as a next expensive route.  A narrower interaction-specific route remains plausible but unbuilt.  It must first demonstrate, using existing checkpoints, a sizable and stable population of EWoK items satisfying all of the following:

- official wrong under a legal endpoint,
- negative or weak four-cell interaction,
- positive local context-difference span margin, meaning the model locally recognizes the context word/phrase but fails to use it for target compatibility,
- robustness to summed versus mean-normalized target scoring,
- loss of the signal when the context-difference span is removed,
- similar population under both legal40 8x480 and 12x384 depth, or a clear explanation of why depth uniquely damages it.

If that population is small, unstable, or dominated by noisy lexical substitutions, the route should close.  If it is large and stable, the next proposed comparison requires an explicit interaction-ranking objective with matched exposure and controls before H100 training.

## Interaction with other live work

SGCR remains the live endpoint and takes precedence when delivered.  If SGCR clears or nearly reaches the 41.80 frontier, protect the endpoint and reproduce it before combining in any new mechanism.  If SGCR is sub-frontier but improves COMPS/GlobalPIQA, the first comparison is likely the mass-matched uniform residual control from sgcr official span burden, not SCRA.  If SGCR is flat/damaging and EWoK remains a major weak column, then the CPU full-EWoK interaction/specificity measurement is the next route-shaping work.

innovation-masking line remains distinct.  Its signal is source/rewrite novelty within compact pairs; the narrowed SCRA signal is relation context-target interaction.  Do not use line to choose relation lexicon, examples, or weights.  If both survive independently, a later factorial comparison would be needed, but neither should be combined before its own mechanism is real.

## Concrete next work

1. If the runtime delivers SGCR training/evaluation results, collect them and run `scripts/sgcr_endpoint_interpreter.py` before any SCRA work.
2. Otherwise, implement the full-EWoK interaction/specificity measurement over existing legal40 and depth checkpoints. It should reuse ewok contrast preservation anatomy scoring but add context-difference span extraction, local span margin, deleted-context target scores, summed/mean target interactions, and per-domain/per-context-type contingency tables.
3. If that measurement finds a real interaction-specific population, define the exact objective and accounting. If it does not, close SCRA and return to a different representation, objective, or data mechanism.
