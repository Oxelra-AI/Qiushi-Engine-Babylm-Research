# earlier analysis mirrored four-row relation-fact binding family

From assignment-reversal maps: values present as literal strings in context.
Valid maps: 116/120
Relations: {'death_place': 104, 'birthplace': 11, 'founded_year': 1}

## Design
C_A = source + update_a + strangers; C_B = source + update_b + strangers
1. (C_A, ask A) -> nv [updated]   2. (C_A, ask B) -> v_B [retained]
3. (C_B, ask A) -> v_A [retained]  4. (C_B, ask B) -> nv [updated]

Structural balance: same context in both answer roles; each entity in both roles;
operation count identical within quad; stranger updates identical for both contexts.
Only sufficient predictor: identity-conditioned state assignment.

## train_frame_seen: 8600 rows, 644285 words
P(updated) = 0.500
Shared-context word mismatches: 0
C_A vs C_B word delta: mean=-0.02, range=[-2, 2]
Answer words: updated=1.00, retained=1.38
  k=0: n=1720, P(updated)=0.500
  k=1: n=1720, P(updated)=0.500
  k=2: n=1720, P(updated)=0.500
  k=3: n=1720, P(updated)=0.500
  k=4: n=1720, P(updated)=0.500

## heldout_frame_all: 4800 rows, 358896 words
P(updated) = 0.500
Shared-context word mismatches: 0
C_A vs C_B word delta: mean=-0.13, range=[-1, 1]
Answer words: updated=1.00, retained=1.32
  k=0: n=960, P(updated)=0.500
  k=1: n=960, P(updated)=0.500
  k=2: n=960, P(updated)=0.500
  k=3: n=960, P(updated)=0.500
  k=4: n=960, P(updated)=0.500

## Skipped maps
  rf_death_place_10314: entity_b 'Pone' not whole-word in source
  rf_death_place_4727: entity_a 'Cormack' not whole-word in source
  rf_death_place_4636: entity_a 'Cormack' not whole-word in source
  rf_death_place_4681: entity_a 'Cormack' not whole-word in source
