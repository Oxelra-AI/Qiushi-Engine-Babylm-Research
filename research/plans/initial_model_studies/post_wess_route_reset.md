# earlier analysis — Post-WESS route reset for BabyLM Strict-Small SOTA

## Current scientific state

The WESS mechanism has ended as a BabyLM route. It is a real mechanism when entity/state addresses are supplied, but it does not operate on unlabeled official inputs and did not transfer into the ordinary DeBERTa backbone. The decisive unlabeled-address experiment is `data/unlabeled_address_decisive.json`; the route note is `notes/wess_route_closed.md`.

The remaining score gap is concrete. Protected internal best:

- DeBERTa-v2 8×480, WWM, baseline16k, 100M exposure: Overall 40.5269.

Public leader:

- `go76dof/wwm_curriculum_simplification_40k`: Overall 41.8011.

Column gaps (protected minus leader):

| column | gap |
|---|---:|
| Entity Tracking | -5.83 |
| GlobalPIQA | -4.03 |
| EWoK | -3.88 |
| (Super)GLUE | -1.77 |
| COMPS | -1.38 |
| BLiMP | -0.44 |
| AoA | -0.17 |
| BLiMP Supplement | +3.84 |
| Reading | +2.19 |

The next route must improve Entity/EWoK/GlobalPIQA without giving back the protected Supplement/Reading advantages. A mechanism that only raises GlobalPIQA in the way no-address WESS did is not enough; Entity and EWoK must move.

## What the literature now adds

### 1. Morphology-aware representation is a live target, but not via the old 40k test

The morpheme-aware BabyLM paper reports large EWoK and Entity movement from Morfessor/rule-based morphology-aware tokenization under 10M/100M BabyLM settings: e.g. GPT-BERT Morfessor in their table has much higher EWoK/Entity than BPE, while GPT-BERT curriculum with Morfessor is higher still \cite{blc2025morphemea}. Their exact score scale and architecture/year differ from our 2026 setup, but the pattern aligns with the remaining gap: world knowledge and entity tracking may be representation-limited.

Local constraints:

- Our legal 40k ByteLevel/official-corpus tokenizer repeatedly damaged Reading and did not improve Entity/EWoK. That negative does **not** test morphology-aware segmentation; it tests larger frequency-based subword inventory.
- Early causal morph-side runs were not a clean DeBERTa MLM representation test and did not use the protected architecture.

Most relevant experimental variant:

- Keep baseline16k tokenizer for official compatibility and add a learned token-surface adapter to DeBERTa embeddings: char n-gram / suffix-prefix / simple-morpheme features projected into the token embedding with a learned gate.
- Include a parameter-matched token-ID adapter control to separate shared surface structure from extra capacity.
- Evaluate on protected-shape 10M first, then scale only if Entity/EWoK/COMPS move without BLiMP/Supplement/Reading loss.

This tests the morphology/representation hypothesis while avoiding the custom-tokenizer engineering and Reading collapse risk.

### 2. Model-centric curriculum is worth a new test, but source-block curriculum is not

The influence-driven curriculum paper reports large gains under BabyLM-limited training, especially for RoBERTa: up to about +9.36 pp macro on D2024 and +7.96 pp on a stratified dataset for influence-based orderings, with full per-epoch dataset coverage and local shuffling \cite{schoenegger2025influenceb}. The method differs from the source-block curriculum we already tested: it is model-centric and mostly preserves source distribution over time, rather than presenting hand-labeled source stages.

Local constraints:

- Our S2 leader-style word-clock curriculum hurt BLiMP/Entity/EWoK/COMPS while raising GlobalPIQA. This argues against hand-designed length/source curricula.
- Full gradient influence over all documents could be expensive, but a cheaper version can be tested first.

Most relevant experimental variant:

- Build an approximate influence/prototypicality score for official-corpus windows using an existing 10M or 100M DeBERTa checkpoint: per-window MLM loss, hidden-state similarity to corpus average, and optionally embedding-gradient cosine on a small representative sample.
- Make an epoch-wise schedule that covers the full 10M pool each epoch but orders windows by influence bands with within-band shuffling, avoiding source-block presentation.
- Compare against existing random-order 10M protected/S1 coordinates. If only GlobalPIQA moves, do not scale; if Entity/EWoK move, run the same schedule at 100M.

### 3. Sequence length/context remains under-tested on the protected backbone

The sequence-length paper reports task-specific length effects: longer contexts are associated with Entity Tracking and Reading gains in transformer-family models, while shorter contexts favor BLiMP/Supplement; EWoK is relatively stable but can peak at longer lengths in their OPT table \cite{salhan2025whata}. Our own earlier note observed that the 256-token protected training truncates a substantial fraction of examples. We have not run a clean 8×480 DeBERTa WWM 512-token or 1024-token comparison.

Local constraints:

- A longer sequence gives fewer parameter updates for the same word exposure if batch geometry changes; this can hurt local grammar.
- Reading is already stronger than the leader, so the purpose is Entity/GlobalPIQA/EWoK, not Reading alone.

Most relevant experimental variant:

- Run protected 8×480 DeBERTa WWM at 10M with seq512 and exact word accounting, matched to the existing protected 8x480 10m available coordinate protected 10M coordinate as much as possible.
- If seq512 raises Entity/GlobalPIQA/EWoK without major Supplement/Reading loss, run 100M. If it only raises Reading or harms grammar, stop.

This is likely the fastest non-WESS screen because it reuses current trainer and evaluator.

### 4. Adaptive masking is implementable and may affect Entity/GLUE, but its target is not EWoK

AMLM hard/decay increases several 2025 BabyLM metrics and reports Entity movement under a DeBERTa-v2 MLM setup with 40k BPE, custom data, and decaying mask ratio \cite{edman2025maskc}. The most relevant mechanism is not the final 40k/tokenizer bundle, but the dynamic allocation of masked-token prediction pressure toward tokens the model has not mastered.

Local constraints:

- We have not tested AMLM locally. We tested MNTP and annotated WESS variants, not adaptive WWM.
- The paper reports EWoK only near 50–51, so AMLM is unlikely to solve EWoK alone.
- N-hot/character information in the AMLM paper helps morphology-heavy tasks but can hurt BLiMP; a morphology adapter should be tested separately from AMLM.

Most relevant experimental variant:

- Implement hard adaptive WWM on baseline16k DeBERTa: maintain per-token (or per-whole-word head-token) smoothed prediction accuracy, update mask weights every fixed number of batches, and optionally decay global mask ratio 0.40→0.15.
- First compare at 10M against protected 8×480 and S1 10M baselines. Watch Entity, GLUE proxy when feasible, COMPS, and BLiMP/Supplement damage.

## Priority order for execution

### Route A — Long-context protected DeBERTa WWM

Rationale: fastest to test, directly targets context-dependent Entity/GlobalPIQA, and was never cleanly run on the protected backbone.

Experiment:

- DeBERTa-v2 8×480, baseline16k, WWM, official corpus only.
- seq_length/max_seq_length 512; max_position_embeddings at least 1024.
- 10M exposure first; match seeds/RNG as closely as current trainer allows.
- Evaluate BLiMP, Supplement, Entity, COMPS, GlobalPIQA, Reading, full EWoK if promising.

Expected useful outcomes:

- If Entity/EWoK/GlobalPIQA rise with acceptable tradeoff: scale to 100M and add 9-column evaluation.
- If only Reading rises or grammar drops sharply: do not scale.

### Route B — Morphology/surface adapter on protected DeBERTa

Rationale: strongest literature signal specifically for Entity/EWoK; avoids repeating the failed frequency-based 40k tokenizer test.

Experiment:

- Add a token-surface adapter to the input embedding: char n-gram/suffix-prefix/simple-morpheme feature vector or small char-CNN over tokenizer strings, projected to hidden and gated into token embeddings.
- Matched control: same parameter count as a token-ID adapter with no surface sharing.
- 10M exposure, official corpus only, baseline16k tokenizer.
- Evaluate Entity/EWoK/COMPS and BLiMP/Supplement/Reading.

Expected useful outcomes:

- Surface adapter > token-ID adapter on Entity/EWoK/COMPS with small grammar cost: scale and combine with Route A if compatible.
- Both adapters similar: improvement is capacity, not representation.
- Surface adapter harms BLiMP/Supplement like n-hot variants: redesign or stop.

### Route C — Approximate influence curriculum with full pool coverage

Rationale: literature suggests model-centric order can matter; local hand curricula failed but did not test influence ordering.

Experiment:

- Use an existing checkpoint to score official-corpus windows by approximate influence/prototypicality.
- Train with full-coverage epoch-wise schedule: ordered influence bands, within-band shuffle; no source-block curriculum.
- 10M screen first against random-order baseline.

Expected useful outcomes:

- Entity/EWoK and GlobalPIQA improve without grammar collapse: scale.
- Only GlobalPIQA moves, matching earlier S2/no-address pattern: deprioritize.

### Route D — Adaptive WWM hard/decay

Rationale: objective-side pressure toward unmastered tokens may improve Entity/GLUE/COMPS and is mechanically distinct from WESS.

Experiment:

- Maintain per-token masked prediction accuracy and adjust WWM selection probabilities while preserving exact word exposure.
- Compare constant 15% WWM, hard adaptive at 15%, and hard adaptive with 40%→15% mask decay at 10M.

Expected useful outcomes:

- Entity/COMPS/GlobalPIQA improve without EWoK/BLiMP loss: consider full run.
- It behaves like paper AMLM with EWoK flat and grammar tradeoff: combine only if another route supplies EWoK/Entity.

## What not to do next

- Do not keep repairing WESS or a synthetic gold-address proxy.
- Do not scale annotated WESS auxiliary training: exported base threeway official interpretation showed no Entity/EWoK transfer and no address-specific official gains.
- Do not infer SOTA progress from GlobalPIQA alone; no-address reproduced that movement.
- Do not repeat frequency-based 40k tokenizer by itself; our S3/official40k results already showed Reading collapse and no target-cluster gain.
- Do not re-run source-block curriculum as a standalone route; S2 and the influence paper both point away from it.

## Next experiment

Before Execute, a critical research-reader should challenge this reset once: whether Route A/B/C/D truly address the 11.47 summed-point gap, whether any hidden assumption from the WESS period remains, and whether the first executed screen should be long-context, surface adapter, approximate influence curriculum, or adaptive WWM.

If proceeding directly to construction, the fastest high-value screen is Route A (seq512 protected DeBERTa 10M), while Route B requires a new implementation and has the strongest Entity/EWoK literature support.
