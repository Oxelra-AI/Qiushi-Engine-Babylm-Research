# Route after the 50/50 mixture result

## What the mixture adjacent vs shuffled 1m profile result changes

The 50/50 official+rewrite mixture was a fair test of whether the pair vs shuffle 1m profile/59 rewrite-adjacency signal transfers into an official-corpus distribution. It did not transfer cleanly.

Mixture-adjacent minus mixture-shuffled:

| column | seed42 | seed43 | mean |
|---|---:|---:|---:|
| BLiMP | -0.12 | -0.69 | -0.405 |
| Supplement | -3.20 | 0.00 | -1.60 |
| EWoK | -1.82 | -1.46 | -1.64 |
| Entity | +0.23 | +0.50 | +0.365 |
| COMPS | +0.56 | -0.22 | +0.17 |
| Reading eye | +0.06 | -0.15 | -0.045 |
| Reading self-paced | -0.11 | -0.05 | -0.08 |

Absolute mixture-adjacent mean vs official WWM mean:

| column | official WWM mean | mixture-adjacent mean | mixture-adjacent minus official |
|---|---:|---:|---:|
| BLiMP | 56.40 | 54.18 | -2.23 |
| Supplement | 50.80 | 46.00 | -4.80 |
| EWoK | 50.18 | 49.14 | -1.04 |
| Entity | 17.85 | 18.02 | +0.17 |
| COMPS | 50.32 | 50.26 | -0.06 |
| Reading eye | 9.82 | 7.30 | -2.52 |
| Reading self-paced | 3.74 | 2.54 | -1.20 |

Thus the current evidence says:

- Pure pair adjacency contained a real seed-consistent signal for EWoK and Entity.
- Random 50/50 mixture preserves only a small Entity signal and reverses EWoK.
- The mixed data distribution still damages Supplement and Reading relative to official WWM.
- Scaling pure pair data or the random 50/50 mixture would amplify known weaknesses rather than move toward SOTA.

The top-system card remains important but not directly attributable: it uses FineWeb simplification pairs, 40k SentencePiece, 34.7M DeBERTa-v2, LAMB, 10M words, 10 epochs, length curriculum, and WWM7→token3 masking. Our isolated tests show that single mechanical pieces did not reproduce its advantage. Its strongest clue is still paired meaning-preserving data, but the current WikiAuto random-mixture implementation is not enough.

## Mechanistic interpretation

Two constraints shape the proposed follow-up.

First, ordinary WWM may let the model ignore the paired sentence. In a pair example, most masked words can be predicted from their local sentence context. If so, simple adjacency adds little and can even interfere with EWoK when mixed with official data.

Second, the current adjacent-vs-shuffled tests do not isolate only one mechanism. They also change local coherence, topic continuity, sentence-pair length geometry, and per-update padding/mask distribution. The controls are strong at the level of retained content and aggregate token opportunity, but not perfect at the level of what the model is forced to use.

These points argue against another plain fraction sweep as the next move. A 25/75 mixture may reduce distribution damage, but it does not explain why EWoK reversed at 50/50 and is likely to shrink the already small Entity signal. It can become useful later only if a stronger pair-use mechanism emerges.

## Primary next mechanism: cross-view selective masking

The next experiment should turn rewrite pairs from passive redundancy into an active prediction problem. For high-confidence shared anchors between source and target — named entities, numbers, content nouns, predicates, and other nontrivial shared words — mask the anchor on one side while leaving the aligned expression on the other side visible.

Informally:

- Standard WWM asks: predict a word from its sentence context.
- Cross-view masking asks: predict a content word/entity in one view while the other meaning-equivalent view is available.

This directly tests whether the missing factor is active use of the paired expression. It is a learning-mechanism change, not another data mixture.

### Minimal decisive comparison at 1M

Use the same validated 50/50 mixture content as mixture adjacent vs shuffled 1m profile when possible:

1. `mixture_adjacent_standard_wwm`: mixture adjacent vs shuffled 1m profile adjacent result can serve as the standard WWM reference for the same content and seeds.
2. `mixture_adjacent_crossview_mask`: official examples use standard WWM; rewrite-pair examples use cross-view selective masking on aligned content/entity anchors.
3. `mixture_shuffled_crossview_matched`: same official examples, same source/target sentence multisets, same anchor-selection rule and mask budget, but targets are deranged as in mixture adjacent vs shuffled 1m profile. This tests whether the visible other side is actually the paired view rather than merely a distributional co-occurrence.

Keep fixed:

- BERT 8×256, baseline 16k tokenizer, fixed 256 length, `max_position_embeddings=512`;
- exact 1M words, same seed/init/RNG triples 42/456/789 and 43/457/790;
- same 98 updates, same batch size schedule as mixture adjacent vs shuffled 1m profile when using the same materialized examples;
- official examples unchanged and still standard WWM;
- no DeBERTa, no 40k tokenizer, no length schedule, no LAMB, no 10M scaling.

The first implementation should be a separate trainer script cloned from the repaired masked trainer, not a fragile patch to the current trusted trainer.

### Implementation shape

Create a new isolated materializer/trainer pair or a trainer clone:

- Do not edit the stable `babylm_masked_train.py` unless a tiny safe helper is needed; preferably create `babylm_crossview_mask_train.py`.
- Materialized JSONL records for pair examples should explicitly contain:
  - `source_text`
  - `target_text`
  - `mode` (`adjacent` or `shuffled`)
  - `source_pair_id`
  - `target_pair_id`
  - optional `anchor_words` / `anchor_side` metadata.
- The trainer should reconstruct token word groups and side spans, then for pair examples choose high-confidence anchor groups on one side while leaving the other side unmasked.
- Anchor extraction should start conservative: lowercase exact word overlap after stopword removal, numbers, capitalized entities/proper-name tokens, and high-frequency content words only if present in both views. Do not use a noisy external aligner first.
- Match total selected word groups and predicted tokens to the standard WWM budget as closely as possible, and report the actual budget per source.
- For deranged cross-view control, use the same source-side anchor selection distribution; the other view is present but not its true counterpart.
- Save per-run counts of pair examples with anchors, selected anchor groups, masked tokens per word, source vs target masking balance, official vs pair contribution, and token/truncation summaries.

### What would change judgment

The route strengthens if cross-view masking improves Entity and EWoK over both standard adjacent WWM and deranged cross-view control with stable signs, while not increasing the Supplement/Reading damage beyond the mixture adjacent vs shuffled 1m profile adjacent arm.

The route weakens if cross-view masking improves adjacent and deranged equally, helps only Entity by a very small amount, or further damages EWoK/Supplement/Reading. In that case, the current WikiAuto pairs are probably the wrong substrate, or the route needs pair quality/source changes rather than pair-use objectives.

## Secondary routes if cross-view masking fails or is too noisy

1. **Pair quality filtering.** Build high-retention subsets by named-entity overlap, number overlap, content-word overlap, source-target length ratio, and low deletion of relation-bearing words. Compare unfiltered mixture against high-retention mixture with matched length/entity density and a shuffled control. This tests whether EWoK reversal is caused by WikiAuto simplifications deleting facts and relations.

2. **Local paired views vs dispersed paired views.** Materialize source and target as separate examples rather than concatenating source with unrelated target. Compare local placement of true pair partners against dispersed placement with exactly the same individual sentence examples. This removes the artificial incoherent sequence from the shuffled condition and tests whether time-local semantic redundancy matters. This is scientifically clean but may be less directly performance-improving because the transformer does not condition across examples.

3. **Source-presentation dynamics.** With the same 50/50 multiset, compare uniform random mixing against source-homogeneous alternating batches and official-first-then-pair blocks, saving intermediate profiles. This tests whether official and rewrite gradients interfere. It should follow, not precede, a stronger reason to keep pair data.

4. **Official-only semantic WWM.** If external rewrite data keeps damaging broad columns, reallocate the WWM prediction budget inside official data toward entities, numbers, predicates, and pronoun/antecedent contexts, with a matched random span control. This may attack Entity/EWoK without distribution shift.

## Next construction task

The proposed implementation is a minimal, reliable cross-view masking path:

1. Produce a small 10k pair+official JSONL with explicit `source_text`/`target_text` fields and anchor metadata from existing mixture materialization and runner materialization.
2. Create a separate `babylm_crossview_mask_train.py` based on the repaired trainer, preserving saving/loading/evaluation compatibility.
3. Run a tiny smoke that verifies:
   - official examples still use WWM;
   - pair examples have cross-view anchor masks;
   - root and `chck_1M` load with `AutoModelForMaskedLM`;
   - selected group/token budgets are recorded;
   - official fast BLiMP+Entity can run.
4. Only after the smoke passes, run the 1M two-seed comparison: adjacent cross-view vs shuffled cross-view, using mixture adjacent vs shuffled 1m profile standard adjacent as the reference.

Do not run 10M/100M, full nine-column evaluation, submission packaging, or final writing from the current evidence. The current mixture is a useful negative constraint, not a candidate model.
