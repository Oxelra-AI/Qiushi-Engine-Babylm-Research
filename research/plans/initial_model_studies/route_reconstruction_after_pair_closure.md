# earlier analysis — Route reconstruction after clean paired-restatement result

## What changed

The token-aware equal-row pair discriminator tokenaware 1m profile recheck closes the current FineWeb paired-restatement route as a SOTA path. With exact 1M words, 5,988 rows per arm, 47 b128 updates, and zero truncation, `true_pair_adjacent` does not improve the target columns:

- true − hard-negative: Entity −0.28, EWoK −3.55
- true − orig-only: Entity −1.69, EWoK −1.46
- true − shuffled: Entity −1.79, EWoK −2.55

The pair discriminator 1m profile apparent true-vs-hard movement was not a transferable same-binding restatement effect. It came from a weak hard arm plus truncation mismatch. The current implementation of adjacent deterministic simplification does not teach the protected DeBERTa WWM model the missing Entity/EWoK competence.

## Surviving positive data signal

fineweb relation vs random 1m profile remains the best verified data lever:

- FineWeb relation-explicit − random-quality at 1M: EWoK +3.27, BLiMP +0.71, Supplement 0.00, Entity −0.12, COMPS −0.29, Reading −0.41.
- Training losses and tokenization summaries were close across arms, so the EWoK movement is not just easier MLM optimization.
- fineweb experience characterization shows this filter mostly increases static relation and entity-rich educational text, not dynamic state-change threads: transition threads rise only about +0.019 per example and has-transition only about +1.4 percentage points.

Scientific reading: relation-explicit selection supplies real relation/world-knowledge experience, but it does not supply enough cross-sentence entity update pressure. The next route must combine the EWoK relation lever with a mechanism that forces state or binding-relevant tokens to affect MLM logits.

## Route to test next

### A. Scale relation-explicit FineWeb to a moderate budget

Purpose: learn whether the fineweb relation vs random 1m profile EWoK signal persists beyond the 1M short run and whether it decays, saturates, or damages other columns.

Already launched in earlier analysis:

- Output directory: `data/fineweb_relation_matched_3M/`
- Command: `fineweb_relation_matched_materialize.py --target_words 3000000 --max_stream_docs 120000`

After collection, train the two arms with the same protected short-run configuration as fineweb relation vs random 1m profile unless a controlled design justifies b256: DeBERTa-v2 8×480, baseline16k, WWM p=0.15, seed 42/456/789, exact 3M exposure, matched checkpoints, fast profile.

Interpretation:

- If relation-explicit preserves EWoK gain at 3M without new losses, it becomes a real data lane to combine with an Entity mechanism.
- If it decays like AMLM, relation-explicit is only an early-exposure effect and should not dominate compute.
- If it raises EWoK while Entity remains flat, the split is confirmed and the Entity mechanism must be added rather than expecting scale alone to solve Entity.

### B. Construct a binding-sensitive delayed-state masking mechanism

Keep the main DeBERTa WWM objective and model fixed. Allocate a controlled fraction of the mask budget to tokens whose prediction should depend on earlier context:

- repeated entity later mentions;
- locations, possessors, roles, properties, final states following verbs such as moved, gave, kept, became, joined, located, carried, returned;
- relation objects in sentences where the local sentence is ambiguous but previous context identifies the correct entity or value;
- cross-sentence entity chains in relation-explicit FineWeb, not adjacent simplification pairs.

Important: total masked tokens per word and update count must remain matched. This should use ordinary MLM vocabulary logits, not an auxiliary classifier that the main head can ignore.

Small decisive design:

| data | standard WWM | delayed-state/binding mask |
|---|---:|---:|
| FineWeb random-quality | A | B |
| FineWeb relation-explicit | C | D |

Use shared initialization per seed. Start at 1M or 3M depending on construction complexity. The contrast separates the relation data effect from the masking mechanism effect and their combination. It also shows whether the mechanism works only on relation-rich text or generally.

Scale only if D improves both Entity and EWoK relative to C and A without a broad BLiMP/Supplement/Reading drop, and if B/D also improve a frozen held-out entity-chain probe built from non-evaluation text. If the masking mechanism only lowers training loss or only improves easy local tokens, it is not enough.

### C. Add counterfactual consequence training only after B is implemented cleanly

If delayed-state masking yields a real Entity movement, extend it to related/unrelated history edits:

- relevant edit: same sentence locally, earlier event changes final location/possessor/role/property;
- unrelated edit: changes a different entity and should not change the target token distribution;
- loss is still on downstream consequence tokens through the MLM head.

This directly attacks the observed bottleneck from earlier probes: history information can be present but not converted into logits.

## Work not to repeat

Do not scale the earlier analysis–296 paired-restatement corpora. Do not reopen intra-simplification entity swaps, same-entity derangement variants, or token-aware pair packing unless a genuinely new mechanism changes the learning signal rather than the text arrangement. The clean pair discriminator tokenaware 1m profile result is sufficient to stop that path.

Do not treat full EWoK as completed. The research report shows full EWoK remains gated; fast EWoK is usable for internal screening but not a final official score.

## Immediate next action

The proposed comparison requires the 3M materialization metadata to establish exact random/relation arms before matched 3M training/evaluation. In parallel or immediately after, the proposed diagnostic is the delayed-state/binding mask extractor on the same random/relation JSONLs so the next experiment becomes a data × masking mechanism comparison rather than another one-dimensional corpus filter.
