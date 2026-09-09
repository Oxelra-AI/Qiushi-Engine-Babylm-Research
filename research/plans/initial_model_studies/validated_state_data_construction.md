# earlier analysis — Validated-State Training Data: Construction Specification

## Experimental requirement

Before the 1M matched training, every candidate passage in the "narrative-state" arm must pass the relation-specificity test from corrected history content probe/275: same local continuation, state-swap history versus unrelated-swap history, with **positive R_extra** confirming that the earlier state genuinely changes the later masked prediction. Without this gate, the filter degenerates into density selection and will reproduce the fineweb relation vs random 1m profile pattern (EWoK up, Entity flat).

## Pipeline overview

```
FineWeb-Edu stream
    ↓
EXTRACT: repeated-entity threads with state-changing events
    ↓
CONSTRUCT: counterfactual conditions (state-cf + unrelated-cf)
    ↓
VALIDATE: frozen wwm43_80M scores R_extra for each thread
    ↓
SELECT: R_extra > 0 → validated passages  |  R_extra ≤ 0 → failed passages
    ↓
PACK: exact 1M words, token-aware (<220 tokens/row)
```

## Three matched arms for training

| Arm | Content | What it tests |
|---|---|---|
| **validated_state** | Passages containing threads where R_extra > 0 | Does genuine state-dependent prediction signal improve Entity? |
| **failed_state** | Passages with same surface structure (repeated entity + cue + filler appearing twice) but R_extra ≤ 0 | Controls for density/structure without validated state signal |
| **random_quality** | Basic high-quality FineWeb without entity structure requirement | Pure source-distribution baseline |

If validated_state improves Entity over failed_state, the improvement is from the **validated state signal**, not entity/relation density. This is the decisive comparison required to isolate the mechanism.

## Phase 1: thread extraction from FineWeb-Edu

Reuse the state counterfactual ranking smoke results v2 `find_relations` pattern on FineWeb-Edu text:

1. Stream `HuggingFaceFW/fineweb-edu`, `sample-10BT`, using local HF cache.
2. For each document, find word matches and identify capitalized entities + content words.
3. Run `find_relations(ms, tok)`: for each capitalized entity within 8 words before a relation cue, look for a single-token content filler within 5 words after the cue. Record the `RelOcc`.
4. Group by (family, entity, filler). Require the same (entity, filler) to appear in **at least two** relation occurrences within the document (analogous to state counterfactual ranking smoke results v2's repeated-relation requirement).
5. For each pair `(r1, r2)` with gap 10–160 word positions: this is one **thread candidate**.

Relation families (same as state counterfactual ranking smoke results v2):
- location: in, at, on, inside, outside, near, from, to, into, onto, around, within
- possession: has, had, held, carried, took, brought, found, gave, kept, bought, put, left, lost, received
- role_attribute: is, was, became, called, named, known, made, elected, appointed, worked, lived
- container: contained, contains, filled, opened, closed, locked, covered, stored, kept
- action_consequence: broke, repaired, opened, closed, moved, changed, turned, killed, saved, built, destroyed, sent

## Phase 2: counterfactual construction

For each thread candidate `(r1, r2)`:

1. **y1** = the repeated single-token filler (appears at both r1.fill and r2.fill positions).
2. **y2** = a frequency-matched, same-tokenizer-length replacement from the corpus frequency bands (not in entity/cue/y1 set, not a relation cue itself).
3. **State-cf**: replace y1 at r1's filler position with y2. Shift all later positions by `len(y2_surface) - len(y1_surface)`.
4. **Unrelated-cf**: find a single-token content word before the later target at similar gap distance, not entity/cue/y1/y2. Replace it with a frequency-matched alternative. Shift later positions.
5. **Make conditions**: extract a context window centered on the later filler position (left 650 chars, right 220 chars). Record separate `(target_start, target_end)` for each condition.
6. **Verify**: the target substring and local ±80 chars around it must be identical across original, state-cf, and unrelated-cf. If not, skip.

This is the corrected construction from state counterfactual ranking smoke results v2, applied to FineWeb documents.

## Phase 3: model validation

Load frozen `wwm43_80M` once. For each validated thread:

```python
score_orig = log_p(y1 | orig_masked) - log_p(y2 | orig_masked)
score_state_cf = log_p(y1 | state_cf_masked) - log_p(y2 | state_cf_masked)
score_unrel_cf = log_p(y1 | unrel_cf_masked) - log_p(y2 | unrel_cf_masked)

R_state = score_orig - score_state_cf
R_unrelated = score_orig - score_unrel_cf
R_extra = R_state - R_unrelated
```

**Selection gate**: keep thread if `R_extra > 0`.

A passage enters the **validated_state** arm if it contains at least one thread with R_extra > 0. A passage enters the **failed_state** arm if all its threads have R_extra ≤ 0 but it does have the same surface structure (repeated entity + cue + single-token filler appearing twice).

## Phase 4: token-aware packing

For each arm:
- Pack validated/failed/random passages into training rows.
- Each row: up to **220 baseline16k tokens** (leaving room for special tokens within seq_length 256).
- Variable whitespace words per row (no fixed 160-word packing).
- Total: exactly **1,000,000 whitespace words** per arm.
- If a passage exceeds 220 tokens, split at a sentence boundary within the passage.
- Record: total rows, tokens per row distribution, words per row distribution, truncation stats.
- Target: <5% rows exceeding 220 tokens (compared with fineweb relation vs random 1m profile's 24%).

## Yield estimation

From state counterfactual ranking smoke results v2 on official corpus: 11 cases from 1,500 docs (0.7% yield per doc).
From fineweb relation vs random 1m profile data: FineWeb-Edu has ~54% of docs passing the relation filter.
On FineWeb-Edu, relation threads should be much denser. Estimated:
- ~40–60% of relation-filtered FineWeb docs will have repeated (entity, filler) pairs.
- ~30–50% of those will have R_extra > 0 (based on state counterfactual ranking smoke results v2's 9/11 = 82% positive rate on the strict inventory, though FineWeb text is noisier than the highly curated 11).
- Conservative estimate: ~15–25% of streamed FineWeb docs will produce validated threads.
- For 1M words at ~350 words/doc, need ~2,857 validated docs.
- At 20% validation rate from 5,000 qualifying docs, need to stream ~25,000 docs.
- FineWeb-Edu sample-10BT has millions of docs; yield is not a constraint.

## Output files

```
data/validated_state_1M/
  validated_state_1000000w.jsonl
  failed_state_1000000w.jsonl
  random_quality_1000000w.jsonl
  materialization_meta.json
  validation_stats.json     # R_extra distribution, pass/fail rates, family breakdown
  samples_validated.json    # 30 examples with thread annotations and R_extra values
  samples_failed.json       # 30 examples showing what failed validation
```

## What this proves before training

Before any GPU training hours:
- The validation_stats.json shows how many FineWeb passages contain genuine state-dependent prediction signal.
- If R_extra pass rate is very low (<5%), the route has no real Entity-relevant signal from data selection alone, and training should not proceed without an objective change.
- If R_extra pass rate is reasonable (>15%) and the validated passages have clearly different content from failed passages, the three-arm training comparison is scientifically meaningful.

## After materialization: short training

Only if validation_stats confirms adequate yield and clear validated/failed separation:

- Train all three arms: DeBERTa-v2 8×480, baseline16k, WWM p=0.15, seed 42, extra_init_seed 456, train_rng_seed 789, batch 128, 1M word exposure each.
- Evaluate at chck_1M: BLiMP fast, Supplement fast, EWoK fast, Entity Tracking fast, COMPS, Reading.
- **Primary comparison**: validated_state minus failed_state on Entity Tracking fast.
- **Secondary**: validated_state minus random_quality on EWoK fast and Entity fast.
- If validated_state > failed_state on Entity: the mechanism works and scaling is justified.
- If validated_state ≈ failed_state on Entity: model-validated selection alone doesn't create the training signal; need objective/architecture change alongside data.

## Implementation notes

- The model validation is the most expensive phase (~0.05s per thread on H100). Budget ~30–50 min for 30,000–60,000 thread candidates.
- Load the model once, process threads in batches.
- Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for memory safety.
- The tokenizer is the same baseline16k used for training, so token-length estimates are exact.
- All text is from publicly available FineWeb-Edu; no generated text, no evaluation contamination.
- Word counting uses whitespace split, matching the BabyLM standard.

## What should NOT happen

- Do not train before validation_stats confirms the R_extra gate works on FineWeb.
- Do not scale to 10M/100M until the 1M three-arm comparison shows Entity gain from validated_state over failed_state.
- Do not add simplification pairs, AMLM, RMEC, or objective changes until the data-only validated-state hypothesis is tested.
- Do not call this "narrative state transition filtering" unless the actual test shows state-specific prediction change, not just entity/event-word density.
