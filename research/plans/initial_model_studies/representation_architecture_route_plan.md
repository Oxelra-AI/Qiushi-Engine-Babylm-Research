# earlier analysis — Representation/architecture route after cadence closure

## Starting scientific state

Cadence is no longer the active training route. The matched all-arms 1M experiment showed that length scheduling, mask decay, and their combination all reduce EWoK while only trading small gains into Supplement or Entity. Broad prefix objectives are also not active: ordered-prefix dependence exists, but the clean local-neighbor held-out surface is too sparse and mostly lexical/topic/discourse continuation rather than reusable entity-state binding.

The experimental constraint is: do not fall back to ordinary surface morphology adapters or vocabulary-size variants. Those have already been tested or are too close to tested negative routes. The next route must compare genuinely different representation or architecture mechanisms that can let low-frequency entity-property bindings form across context and directly influence model output, while separating mechanism effects from added capacity.

One measurement issue remains visible: the 2025 `Mask and You Shall Receive` table reports Entity Tracking around 42-44 for regular/hard MLM, while our exact local MLM fast Entity numbers are around 17-21. The official scorer exact-matches the selected candidate completion to `options[0]`, so this is probably not free-generation failure. It may reflect task version, full-vs-fast split, backend/package version, or leaderboard packaging. Do not let it paralyze mechanism work, but resolve it before making endpoint-level Entity conclusions.

## Route comparison

### Route A — Contextual lexical binding head (CLBH)

**Mechanism.** Add an output-side context pointer/retrieval component to the DeBERTa MLM head. For every masked position, the normal vocabulary logits are mixed with a context-copy distribution over visible word-start/content tokens in the same sequence. The copy distribution is computed from the masked-position hidden state attending to earlier and later visible token states, with optional rarity weighting and word-start grouping. If the target token/string appears in context, its probability can be raised through the pointer path; if not, the normal MLM head dominates. This creates a direct path from cross-context entity/property mentions to output probabilities without gold entity roles.

**Why it is different from failed routes.** It does not add a broad auxiliary objective, does not require a query-entity router, and does not merely add surface features to input embeddings. It changes the output computation so contextual evidence can directly alter candidate scores. It also has a natural control: the same parameters with shuffled pointer token identities or random context keys.

**Main risk.** It may mostly copy repeated lexical items and help examples where the answer appears verbatim, without improving abstract property binding or EWoK. Therefore the experiment must report target-in-context and no-target-in-context subsets for Entity/EWoK probes, and include a shuffled-identity pointer control.

**Short experiment.** Two seeds at 1M on official data:

1. matched WWM reference using the same trainer/config;
2. CLBH pointer model;
3. shuffled-identity CLBH control with same parameters and attention computation but token-to-output accumulation randomly permuted within each batch or sequence.

Useful signal: CLBH beats both reference and shuffled control on Entity and does not reduce EWoK/Supplement/COMPS/Reading. If it helps only target-in-context rows and the shuffled control matches it, the route is not the missing mechanism.

### Route B — Soft latent entity binding slots (LEBS)

**Mechanism.** Replace WESS gold/predicted hard slots with soft, differentiable latent slots inside the encoder. A small set of learned slot queries attends to word-start rare/content token states, updates slot states through gated recurrence, and exposes the slots back to token states or the MLM head through cross-attention. No token-role classifier, no gold entity spans, no hard query parsing. The model can learn reusable binding variables as latent attractors, while every read remains differentiable.

**Why it is different from WESS.** WESS succeeded with gold addresses but failed when a discrete router could not find query entities. LEBS removes the brittle parse stage: every masked position softly reads all slots, and slots are trained only through MLM. It is architecture-level memory, not annotated auxiliary supervision.

**Main risk.** Soft slots may collapse into generic topic vectors or act as extra capacity. Controls must include param-matched no-slot adapters, random slot reads, and slot-read ablation at evaluation.

**Short experiment.** Two seeds at 1M:

1. matched WWM reference;
2. LEBS with 8 or 16 slots, small bottleneck, gated residual read into the MLM head;
3. parameter-matched bottleneck adapter with no sequence memory.

Useful signal: LEBS specifically improves Entity/EWoK over the param-matched adapter, and slot-read ablation removes the gain. If LEBS and adapter move together, the effect is capacity/optimization rather than binding.

### Route C — Entity-conditioned output factorization (ECOF)

**Mechanism.** Factor MLM logits into the standard token logit plus a low-rank context-conditioned bias whose basis is built from visible rare/content token hidden states grouped by token identity. The bias is not a char/morph feature; it is a dynamic sequence-conditioned output factor. The same rare token can write a context vector that biases semantically related property/continuation tokens through learned low-rank maps.

**Why it is different from surface adapters.** It does not change tokenization, vocabulary size, or input morphology. It changes how context-derived entity evidence shifts the output distribution, allowing low-frequency identity/property evidence to affect candidate scoring.

**Main risk.** It may become a general topic model. Controls should include a topic-only pooled-context bias with the same rank and a token-identity-shuffled version.

**Short experiment.** Two seeds at 1M, compared against fixed WWM and topic-only low-rank bias. Useful signal is Entity/EWoK movement beyond the topic-only bias without Supplement/Reading loss.

## Route choice for next construction

Start with Route A, CLBH. It is the most direct test of the current bottleneck because it makes contextual entity/property evidence affect output scores immediately and provides clean mechanism controls. It is also simpler than latent slots and can be implemented as an MLM-head extension without changing the whole encoder.

Route B is the second choice if CLBH only copies lexical repeats or fails to move Entity/EWoK. Route C is a refinement of Route A if the pointer path is too literal but the output-side context-bias idea shows promise.

## First construction task

The CLBH trainer requires an explicit design before training:

- reuse protected DeBERTa-v2 WWM data selection, tokenizer, word accounting, and evaluation paths;
- add a pointer/retrieval MLM head that mixes standard vocab logits with context-copy logits over visible token identities;
- ensure the model can be saved/loaded by the local evaluator or provide a compatible wrapper;
- implement two controls: shuffled-token pointer and parameter-matched no-pointer adapter;
- record pointer-gate magnitude, fraction of MLM targets whose gold token appears in visible context, and performance by target frequency/context-repeat band;
- run a smoke and then two-seed 1M comparison only after the head is verified to affect output probabilities and the controls are matched.

The next Execute result should not be a large training run. It should be a precise CLBH implementation plus a 20k smoke showing that the pointer path changes MLM logits in the intended direction and that shuffled/no-pointer controls are available.
