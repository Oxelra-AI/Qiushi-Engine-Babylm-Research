# route reconstruction — Route reconstruction after both natural endpoints closed

## Where the research stands (evidence, not assumption)

Best complete **legal** endpoint: scale1.75 adapter128 100M, Overall **41.5707**, still **-0.2293** below 41.8.
U256 faithful row-tail visibility 100M: Overall **41.3292**, only +0.0715 over spatial repair route status.
spatial repair route status legal compact-view reinvest baseline: Overall **41.2578**.
The old inherited-tokenizer 42.0331 endpoint is non-submittable (tokenizer-fitting leakage) and is evidence only.

### Shared Limitation Across Comparisons

Both closed endpoints redirect the shared backbone through a **large multidirectional rotation**, and the displacement is **structurally concentrated the same way**:

| Endpoint | overall stock cosine vs spatial repair route status | embeddings diff2 share | mid/upper attn cosine | MLM head cosine |
|---|---:|---:|---:|---:|
| scale1.75 100M | 0.7353 | 0.168 (largest single group) | 0.60–0.67 | ~0.98 |
| U256 100M | 0.6453 | 0.224 (largest single group) | 0.54–0.58 | ~0.98 |

Early gains (scale1.75 +0.86 cheap7 @80M; U256 +0.92 @20M) do **not** survive as broad endpoint gains. They are absorbed as boundary rotation that repairs some families while overwriting others (EWoK material/spatial/quantitative, Supplement subject-aux inversion, Reading, SuperGLUE). Endpoint hard-majority recovers only +0.0258 despite +16.8 oracle headroom → not an ensemble problem.

The route conclusions agree: private scale endpoint vs mechanism synthesis "scale1.75 valuable broad-score substrate, not a SOTA endpoint"; lead route synthesis after cohmargin closed protected late-path / frozen-readout / local-logit branches and concluded **"relation cost is trajectory/representation; next route should FORM context-sensitive representations, not replay a protected readout."**

### earlier analysis probe result (constrains the paired-view route)

Frozen spatial repair route status **already** resolves true legal compact pairs almost perfectly even with partner attention removed (same-row top1 1.000, margin 0.62). A detached private channel adds only +0.017 / +0.011 margin. **Paired-view retrieval is saturated in the frozen model** → aligning paraphrase views is not an unsaturated target. The limitation is that pairs validate *pathway separation* but not *acquisition*.

## Route judgment

The next signal must be an **incremental quantity that is NOT already resolved by frozen spatial repair route status and IS connected to next-token / masked-token prediction**, and it must enter the model in a way that does not simply add another shared-backbone rotation that overwrites protected families.

Ruled OUT as next main route (evidence, not taste):
- another residual-scale amplitude, U256 chunking variant, tokenizer-size sweep, RTD/Muon/LAMB reprise, naive endpoint ensemble (all closed);
- paired-view alignment as retrieval (saturated in frozen spatial repair route status);
- frozen protected readout / late-path / local-logit protection (lead route synthesis after cohmargin closed);
- global word-mean / innovation / pivot masking targets (closed, all redistribute).

### Verified Legal Discourse-State Substrate

The legal 10M pool sources: CHILDES 2.57M words (16,093 rows), Gutenberg 1.86M, OpenSubtitles 1.83M, SimpleWiki 1.06M, BNC Spoken 0.58M, Switchboard 21k — **plus** qwen_pair_packed 1.66M and FineWeb compact-reinvest 0.42M.
- CHILDES/Switchboard/BNC/OpenSubtitles are conversational; rows are multi-utterance (CHILDES ~160 words/row).
- Pool rows are **shuffled** (example_ids non-monotone within source), so cross-row ordering is destroyed at pool level, BUT **within-row utterance/sentence adjacency is intact and legal** (each packed row is consecutive same-document text). This is the most robust discourse substrate and needs no cross-row reconstruction.

## Candidate next route (leading, to be pressure-tested)

**Context-sensitive discourse-state acquisition with train-time protection of established competence.**

Mechanism (unsaturated, prediction-connected): predict a stop-gradient latent of a held-out sentence/utterance from the surrounding intra-row context (a discourse-state target), *jointly* with ordinary MLM, so the model must form a persistent cross-sentence representation that ordinary MLM does not require. This is a different learning unit (predict missing event given episode) rather than a chunk-length change.

Protection (addresses the shared obstruction): during acquisition, damp updates on the parameter directions the endpoints most damaged (embeddings + mid/upper attention) using spatial repair route status corpus-derived importance, so the new signal is acquired without the observed EWoK/Supplement/Reading/SuperGLUE overwrite. Not a frozen readout (closed that) — it is train-time selective plasticity during representation formation.

## Cheap discriminating tests (no endpoint-scale training)

1. **Saved-artifact importance-vs-displacement (zero training):** compute spatial repair route status corpus MLM squared-gradient importance per parameter; correlate with the scale1.75/U256 endpoint displacement per group/decile. Question: were the *damaged* families' directions the *high-importance* ones? If yes, selective protection is well-motivated; if displacement is importance-agnostic, protection reduces to a step-size change → stop.
2. **Frozen discourse-state recoverability (zero training):** with frozen spatial repair route status, test whether a held-out intra-row sentence latent is predictable from context better than from shuffled-context controls, by source. If true-context >> shuffled, an unsaturated discourse signal exists; if equal, stop.
3. Only if 1 and 2 are positive: matched 10–20M continuations {MLM replay; discourse-state aux true; discourse-state aux shuffled; +/- importance protection}, with mandatory sentinels: Supplement subject-aux inversion, EWoK material/spatial/quantitative, Reading, small SuperGLUE panel, cheap7.

## Complementary Mechanisms
Context-sensitive representation formation is also the subject of lead route synthesis after cohmargin. The complementary scientific constructions are **discourse-state + selective-plasticity** and the importance/displacement test using saved artifacts.
