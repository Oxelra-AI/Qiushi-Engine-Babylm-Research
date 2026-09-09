# cue audit answer slot rows: Trusted T/U/N Base Characterization and Entity-Binding Construction

## Trusted Base State-Margin Characterization (base43022, chck_100M)

Under trusted adapter-aware loading (AdapterDebertaV2ForMaskedLM, 35,463,008 params,
995,584 adapter params, scale 1.75), the 400 balanced held-out packets reveal a 
definitive entity-binding failure.

### Gold accuracy: does the model predict the correct current state?

| Packet Type | Condition | Gold Accuracy | Mean Gold Margin (nats) | SE |
|---|---|---|---|---|
| **UPDATED_USE** | T (true update) | **60.0%** | +0.91 | 0.22 |
| UPDATED_USE | U (foreign update) | 10.0% | -4.75 | 0.25 |
| UPDATED_USE | N (no update) | 8.5% | -4.88 | 0.24 |
| **DISTRACTOR** | T (true update) | **39.0%** | **-0.59** | 0.16 |
| DISTRACTOR | U (foreign update) | 90.5% | +3.88 | 0.22 |
| DISTRACTOR | N (no update) | 91.5% | +3.99 | 0.22 |

### Interpretation

1. **Entity-binding failure at DISTRACTOR-T (39%)**: When the model sees an update about 
   entity B but is asked about entity A (whose state is unchanged), it preferentially 
   predicts entity B's new state (61% of the time). The true update about B *pulls* 
   prediction away from A's correct source_state. Mean margin -0.59 nats.

2. **Weak update incorporation at UPDATED-T (60%)**: The model partially uses the true 
   update when asked about the same entity, but 40% still prefer source_state. Modest 
   margin +0.91 nats.

3. **Recency dominance without entity gating**: T-N shift is symmetric:
   - UPDATED: 8.5% → 60% (+51.5 pp) — update shifts toward new_state ✓
   - DISTRACTOR: 91.5% → 39% (−52.5 pp) — update shifts toward new_state ✗
   
   Both shifts are ~52 pp, indicating the model applies the update to ANY entity 
   equally, not just the entity it concerns. Entity identity is not gated.

4. **Foreign update discrimination works**: UPDATED-U at 10% shows the model correctly
   resists a random foreign update. The failure is specific to the true update appearing
   in the context but being about a different entity.

5. **No update = strong source preference**: Both U and N conditions show ~90% source_state
   preference, with large positive margins (+3.9 to +5.1 nats). The model's default is to
   predict source_state; the true update overwhelms this default without checking entity.

### Significance for the joint principle

This is precisely the deficit the joint principle (uniquely sufficient variable + 
concentrated credit) should address. The recombination minimal pairs make entity identity 
the only distinguishing cue, and answer-only training concentrates credit on the state 
phrase where entity-conditioned gating must operate.

## Answer-Slot Cue Audit (earlier analysis rows)

The earlier analysis answer-slot construction was found to have systematic local-copy cues:
- 79.6% of answers are single tokens; 87.2% of foils are longer
- 94.2% of DISTRACTOR answers appear verbatim in source sentence
- 100% of DISTRACTOR rows have only source-state hits in the use sentence
- Base achieved 81% accuracy through local copy, not entity binding

This disqualifies the earlier analysis rows for binding training. The recombination rows fix this.

## Recombination Construction

From each DISTRACTOR packet, two training rows:
- Row A: query target_entity → source_state (entity unchanged)  
- Row B: query updated_entity → new_state (entity changed)
- Same source + update context; only entity name differs

From each UPDATED packet, one training row:
- Row: query entity → new_state

Dataset sizes:
- Train: 4,998 rows (3,332 DISTRACTOR pairs + 1,666 UPDATED singles)
- Heldout: 600 rows (400 DISTRACTOR pairs + 200 UPDATED singles)
- Binding pairs: 200 for joint correctness evaluation

Answer phrase lengths: mostly 2-6 words (mode at 4), not single tokens.

## Training Harness

`answer_only_binding_train.py` smoke-tested on CPU with 8 rows / 2 epochs:
- Trusted loading verified (35,463,008 params, 995,584 adapter trainable, 170 frozen)
- Answer-only loss: mask answer tokens, loss only at those positions
- Training loss decreased: 7.66 → 7.25
- Held-out NLL decreased: 7.59 → 7.14
- Joint binding starts at 0; one B-half correct after 2 epochs

Ready for GPU execution when dose training completes.

## Files

- Base T/U/N: `data/trusted_base_TUN_rescore/`
- Recombination rows: `data/recombination_rows/`
- Training harness: `scripts/answer_only_binding_train.py`
- Cue audit: `notes/cue_audit_answer_slot_rows.md`
