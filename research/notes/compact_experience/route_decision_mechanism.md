# route decision mechanism — Route decision mechanism note

Plan: `research/plans/compact_experience/frontier_route_decision.md`

The paired-continuation result motivates the following mechanism comparison.

## What paired continuation eval established

The paired continuation from the initial reference `chck_70M` gave the cleanest granularity test in these experiments:

- WWM continuation (restart control) reproduces the INITIAL_MODEL_STUDIES original `chck_100M` almost exactly.
- Token continuation (treatment), with the same weights and same post-70M data, is slightly worse overall:
  - equal-7 mean: −0.301
  - legacy weighted screen: −0.237
  - main losses: Supplement −1.20, COMPS −0.69, BLiMP −0.19, Entity −0.17, Reading −0.635
  - main gains: EWoK +0.73, GlobalPIQA_nonparallel +3.0

This closes unconditional late WWM→token as a main route on the inherited DeBERTa-v2 8×480 backbone.

## Why the next route is paired-rewrite data

Three backbone-internal routes have now failed to move the frontier:

1. residualized C/S selection;
2. same-content ordering;
3. unconditional late WWM→token.

They all redistribute supervision across columns instead of expanding the model's competence frontier. The only remaining candidate with external evidence of a true frontier shift is **aligned paired-rewrite / meaning-preserving multi-view data**.

## First experiment to build next

Use the fixed earlier analysis-family backbone and test paired-rewrite data with matched controls:

- **A_official**: official 10M corpus
- **B_pair**: ≤10M words of aligned original+rewrite pairs
- **C_simple_only**: the same simplified/rewrite side without originals

The key controls are:

- plain MLM only, no contrastive objective;
- ≤10M total words for the training corpus;
- full official-compatible nine-entry evaluation, including (Super)GLUE and AoA.

Candidate sources already probed:

- `hkust-nlp/SynCSE-partial-NLI` — MIT license, aligned sentence/paraphrase/hard-negative triples.
- `Nechba/wikilarge-text-simplification` — Apache-2.0, aligned Normal/Simple simplification pairs.

## Scientific target

The experiment should determine whether aligned multi-view data produces a real frontier shift on the inherited backbone, or whether the leader's advantage is coupled to 40k vocabulary / capacity / optimizer interactions. A null result with the simple-only control is also high-information because it localizes the leader edge to those couplings rather than to paired data itself.
