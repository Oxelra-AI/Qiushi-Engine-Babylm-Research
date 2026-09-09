# route portfolio context alternative binding — route portfolio after curriculum and adapter readouts

## Active scientific target

At the time of this note, no compliant Strict-Small endpoint had exceeded the visible 41.80 frontier. The strongest completed compliant full endpoint remains the legal40k 8×480 fixed-256 AdamW compact-view run at Overall 41.1406, with cheap7 43.1079. The noncompliant inherited-tokenizer compact endpoint at 42.0331 remains useful mechanism evidence only.

The current hard object is context-conditioned alternative binding. The best compliant endpoint has GlobalPIQA_parallel 22.33% over 103 four-choice rows, with ranks 1/2/3/4 = 23/24/26/30 and a 52-row always-wrong core whose correct option is usually rank 3 or 4, mean top-minus-correct 1.953 nats. The same endpoint has 2,550 EWoK stable conditional-reversal failures; variable-swap contexts are especially severe: 1,030 stable failures out of 1,281 wrong rows. Spatial-relations and physical-interactions have high stable-failure fractions among wrong rows, 0.783 and 0.754.

Recent clean negative results matter for route choice:

- Corrected word-boundary AdamW sequence curriculum under the trusted legal40k coordinate scored cheap7 42.3529, below fixed-256 43.1079; GlobalPIQA_parallel stayed 18.45 and EWoK fell to 50.03.
- Ordinary joint residual adapters at 20M are trainable but redistribute competence. paired tail smoke and adapter ablation/140 inference readouts show GlobalPIQA damage is mostly direct residual output, while EWoK damage is mostly stock-trajectory movement. Post-hoc amplitude does not recover the aoa mincontext discrepancy audit key-column anchor.
- Corpus relation funnels repeatedly failed as trainable objects: dense PVDM hurt; sparse visible-target auxiliary was representation lookup; full-context pivot substitution and natural target ranking were saturated; broad two-context/two-target pools and slot/opposition pools measured attested fit; witness-grounded spatial exchange was too sparse and matched permutation-null behavior.
- Whole-batch causal substitution and same-corruption MNTP in COMPACT_EXPERIENCE moved some columns but damaged the aggregate. From-scratch RTD-GDES in improved its own readout but gave only small cheap7 movement with EWoK/GlobalPIQA/Reading damage; the delayed-tail RTD explanation lacked enough source-measured support.

The next route must not be another polished extraction funnel unless frozen checkpoints show unsaturated separation on the missing interaction and not only local fit.

## Mechanism class A — protected context-conditioned logit side path

Scientific idea: keep the compact-view base encoder and MLM head learning exactly as in the trusted route, but add a separate context-conditioned logit correction trained from detached base features. The side path is allowed to learn an alternative-sensitive distributional correction, while its loss does not backpropagate into the base encoder, base embeddings, relative-position channel, or base MLM head. This directly targets the adapter coupling problem: ordinary adapters can help BLiMP/Supplement/COMPS but alter the stock trajectory and directly damage GlobalPIQA; a protected side path tests whether additive context-specific capacity can improve final pseudo-likelihood without erasing the base path.

Proposed construction:

- Base forward produces hidden states and base MLM logits exactly as the trusted DeBERTa-v2 model.
- Base loss: ordinary WWM CE on `base_logits`, updating all baseline parameters exactly as before.
- Side branch input: `stopgrad(h_top)` or a small set of stopgrad layer summaries. Branch output: zero-initialized low-rank logit residual `r(h)` added only for the side loss and, at evaluation, to the exported logits.
- Side loss: CE on `base_logits.detach() + r(h.detach())`, updating only side branch parameters. This avoids changing the baseline gradient path while forcing the side branch to model residual context alternatives relative to the current base distribution.
- Optional training-only focus weights must be derived from train-time base uncertainty or target frequency, not official eval rows. Start without such weighting; add it only if unweighted side loss is too dominated by easy tokens.

Lowest-cost reliable work before H100:

1. Implement a CPU/GPU smoke with `side_enabled=false` reproducing baseline first loss, masks, and first AdamW update exactly; `side_enabled=true` must leave base-parameter gradients bitwise equal to baseline when the side loss uses fully detached base logits/features.
2. On frozen 20M/80M/100M checkpoints, fit only the side branch for a tiny number of batches and measure whether side residuals concentrate on high-entropy masked targets rather than simply copying frequent tokens. This is not score evidence, but it tells whether the branch can learn a nontrivial residual without changing base weights.
3. If mechanical identity and residual behavior hold, run a 20M matched-horizon screen from random initialization with live side path and disabled side control. Compare cheap7 and hard surfaces at chck_20M; continue only if broad columns are at least preserved and EWoK/GlobalPIQA do not show the ordinary-adapter damage signature.

Why it is genuinely different: it changes architecture and gradient coupling, not data extraction or optimizer trajectory. It can preserve compact-view broad learning by construction, while testing whether context-conditioned logit correction supplies the missing alternative competition.

Main risks: official likelihood is sensitive to raw logits; a side branch can still worsen GlobalPIQA by output scale. The first version should therefore include a predeclared small residual scale or zero-initialized scalar gate with recorded training values, not post-hoc score-picked scaling.

## Mechanism class B — balanced role-switch experience inside the primary MLM stream

Scientific idea: create a small, legal, evaluation-independent text experience where the same entities or alternatives repeatedly exchange roles under different contexts, so the only way to predict masked consequence words is to bind the current context. Unlike corpus-mined exchange, this does not depend on rare natural witness pairs; unlike perturbation detection, it is primary MLM text, not a binary auxiliary head.

A useful micro-corpus would use common words and many surface forms, e.g. spatial above/below, before/after temporal order, container/support, agent/patient transfer, comparison more/less, open/closed physical state. Each packet should contain both directions with balanced alternatives and paraphrases, such that target priors cancel within the packet: `A above B -> A high/B low`; `B above A -> B high/A low`; and analogous non-spatial families. It must avoid official eval wording and not import hidden teacher outputs. All generated words count inside a ≤10M pool; any endpoint must train from a legal pool that swaps out an equal number of existing words.

Lowest-cost reliable work before H100:

1. Build a 10k–50k-word candidate packet generator from a hand-written grammar and in-budget vocabulary lists sampled from the allowed corpus. Save texts, word counts, family labels, and template IDs.
2. Frozen-score current checkpoints on masked role targets with context-erasure, target-swap, and packet-shuffle comparisons. Accept this object for training only if current models fail the all-four role binding but target priors and local lexical cues are near-null, and if the row-block/compact/interleaved or Muon/AdamW differences align with the real EWoK/GlobalPIQA hard surfaces more than saturated local-fit probes did.
3. If the object passes the frozen test, create a small legal pool replacing equal word mass in low-value repeated/filler material, not appending. First training screen should be 20M matched-horizon or 80M-tail only if legality is preserved by a from-scratch pool design. Continue only if broad cheap columns are preserved while EWoK stable reversals and GlobalPIQA hard ranks move together.

Why it is genuinely different: it changes experience structure, not an extracted auxiliary loss. It deliberately creates high-density role reversals that the legal 10M substrate lacks.

Main risks: artificial packets can teach local template rules, harm natural reading, or overfit to pattern words. Frozen context-erasure and paraphrase-family splits are necessary before training. A small dose and replacement of weak material are safer than a large synthetic block.

## Mechanism class C — mature knowledge consolidation without new representation funnels

Scientific idea: the compact-view base may contain useful mature knowledge distributed across nearby checkpoints, while the endpoint overfits some likelihood surfaces. BabySteps used tail averaging; our prior simple soups were not enough, but a cheap existing-checkpoint consolidation can still be run without new pretraining if GPU lanes are idle. This route is not expected to solve the whole gap alone, but it can provide a stronger base for architecture or data routes.

Lowest-cost reliable work:

- Use existing legal40k fixed-256 checkpoints to build a few predeclared mature-window averages: 80–100 uniform, 90–100 uniform, 70–100 exponential, and perhaps last-k loss-agnostic average. Evaluate cheap7. Run SuperGLUE only if cheap7 improves by a meaningful amount without Supplement/Reading damage. Stop if movement is small or another broad-vs-relation tradeoff appears.

Why it is different: no new data, objective, architecture, or optimizer; it tests temporal consolidation of the existing legal trajectory.

Main risks: unlikely to close a ~0.7 cheap7 requirement alone; can blur calibrated MLM likelihoods.

## Mechanism class D — selective core-update geometry, held behind stronger route evidence

Continuous Muon showed real EWoK movement but broad tradeoff, and abrupt Muon→AdamW switches are closed. The proposed selective Muon/AdaMuon idea — applying orthogonalized updates to core attention/FFN matrices while keeping embeddings, relative positions, norms, and MLM head on AdamW — is mechanistically distinct from LAMB and from empty-moment switching. This route remained inactive pending a coherent gain from the paired tails or a non-training update-geometry probe showing that selective core updates improve relation surfaces without repeating the broad cost.

## Planned Comparisons

Mechanism class A and a lightweight version of class B were proposed as complementary alternatives: A protects the base path through architecture and gradient separation; B changes the legal experience distribution so alternatives exchange roles at density. Both require construction and frozen scoring before considering another 100M run.

Once the paired-tail results become available, the planned comparison is cheap7 against 43.1079. If neither improves the broad surface together with EWoK/GlobalPIQA hard readouts, fresh-moment reset and LR reheating should be closed without more optimizer variants.
