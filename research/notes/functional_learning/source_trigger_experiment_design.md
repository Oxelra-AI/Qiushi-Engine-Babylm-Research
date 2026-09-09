# source trigger experiment design: Matched source-trigger experiment design

## Scientific question

When does identity practice create a **source-triggered competing prediction**
(positive D_m, as in BabyLM) versus **broad target-prior harm** (D_m ≈ 0, as in Step8b)?

The earlier matched-support experiment already had same-window identity practice with full attention access but
found near-zero source-specific excess (T+1.827, U+1.881, excess -0.054±0.045).
The BabyLM relation-learning integration shows BabyLM REPEAT has a strong source-triggered effect with 11-14×
specificity ratio (T vs U). What condition distinguishes these outcomes?

## Key design difference from Step8b

Step8b compared REPEAT (identity) against SUPPORT (no identity, no SRC-family targets).
The arms differed in both **identity pairing** and **SRC-family target exposure**.
Broad harm could come from either factor.

source trigger experiment design compares IDENT_FULL against UNPAIRED_SRC: same contexts, same SRC-family
target tokens, same cue, same entity/position distribution. The ONLY difference
is the source-target pairing:
- IDENT_FULL: SRC(a) → SRC(a) (identity)
- UNPAIRED_SRC: SRC(a) → SRC(b), b≠a (deranged)

Both practice SRC-family output alongside the shared correspondence backbone.
If identity *pairing* (not merely SRC-family exposure) creates a source-triggered
competitor, D_m should be positive.

## Design details

### Training
- 100 shared backbone correspondence seqs/epoch (REWRITE_CUE + RWT targets)
- 400 extra seqs/epoch differ by arm:
  - IDENT_FULL: SRC(a)→SRC(a), full causal attention
  - IDENT_BLOCKED: same tokens, query→source attention blocked (positions 14-16 blocked from qi's context event)
  - UNPAIRED_SRC: same contexts, SRC(a)→SRC(b) deranged
- IDENT_FULL and IDENT_BLOCKED share identical token sequences
- IDENT and UNPAIRED share identical contexts (differ only in target token)

### Cue regimes
- TYPED: COPY_CUE for SRC-family rows, REWRITE_CUE for backbone, task-distinguishable
- UNINFORMATIVE: CONST_CUE for all rows, no task signal

### Corrected T/U probes
For held-out entity h with attribute a:
- **T**: context contains ENT(h) HAS SRC(a) at source slot. Query: ENT(h).
- **U**: same entities, same token multiset, but SRC(a) is under another entity
  and ENT(h) has SRC(b). Entity-attribute swap preserves exact token multiset.
- **NS**: ENT(h) absent from context entirely. Reference only.

Target: RWT(a) for rewrite probe. Also measure p(SRC(a)) from same forward pass.

### Source-specific interaction
D_m(L,Q) = (m_IDENT,T - m_UNPAIRED,T) - (m_IDENT,U - m_UNPAIRED,U)

For NLL: positive D = identity pairing causes extra true-source damage
For p(SRC(a)): positive D = identity pairing causes T-specific source-token elevation

## Predictions and interpretations

### Outcome A: Step8b-like (D_m ≈ 0)
T and U move together for all arms. Identity pairing still insufficient in
synthetic causal-LM. Missing condition may be:
- Natural-language token overlap (function words, frequency structure)
- MLM objective (consistent masking vs mixed copy/rewrite output)
- Scale or training phase effects
- Multi-token source spans vs single-token

### Outcome B: BabyLM-like (D_m > 0 for IDENT_FULL)
Only IDENT_FULL raises p(SRC(a)) and damages RWT(a) specifically under T.
IDENT_BLOCKED and UNPAIRED remove it. This would establish:
- Identity PAIRING (not SRC exposure) creates the competitor
- In-context attention access is required (BLOCKED vs FULL)
- The synthetic substrate CAN reproduce source-triggered competition when
  the paired comparator controls for SRC-family exposure

### Outcome C: Coexistence (within-RWT improves, family/SRC competes)
Identity FULL improves within-RWT ranking while increasing source-token mass
specifically under T. This would support the corrected Step10b finding:
facilitation and competition can coexist in the same model.

### Outcome D: Shared-path (blocking removes both effects)
IDENT_BLOCKED shows neither within-RWT improvement nor source competition
relative to UNPAIRED. This argues against separability: the same contextual
retrieval path supports both identification and competition.

### Outcome E: Typed rescues competition (D_m smaller under TYPED cue)
Typed cue reduces D_m relative to uninformative. The source-triggered competitor
is redirectable by task specification. Cue-swap test at final epoch tests this
within a single trained model.

## Relation to BabyLM evidence

If source trigger experiment design produces Outcome B, the synthetic mechanism matches the BabyLM result:
- Identity pairing → source-triggered competitor (the BabyLM 11-14× specificity)
- Blocked/split → eliminates it (the BabyLM REPEAT_SPLIT arm)
- SOURCE specificity → fires on recognized true source (both studies)

The additional insight from source trigger experiment design would be that the PAIRED identity relation
(not just SRC exposure or recurrence) is the causal factor, because UNPAIRED_SRC
has the same tokens but no identity pairing.

## Metrics
- rwt_nll: total NLL for RWT(a)
- rwt_fnll: RWT family NLL = -log p(RWT)
- rwt_wnll: within-RWT NLL = -log p(RWT(a)|RWT)
- rwt_top1, rwt_mrr: within-RWT ranking
- p_src_a: probability of exact source token SRC(a)
- src_fam: SRC family mass
- copy_nll: -log p(SRC(a))

Verify: rwt_nll = rwt_fnll + rwt_wnll (to numerical tolerance)
