# earlier analysis mirrored four-row family with assignment-position variation

Valid maps: 116/120
Relations: {'death_place': 104, 'birthplace': 11, 'founded_year': 1}

The real assignment update is inserted at a balanced position among the k+1 operation sentences, with the same position for C_A and C_B inside each quad.
Each context still appears once with an updated answer and once with a retained answer; each query entity appears in both roles; k and assignment position have P(updated)=0.5.

## train_frame_seen: 8600 rows, 644105 words
P(updated) = 0.500
Shared-context word mismatches: 0
By k:
  k=0: n=1720, P(updated)=0.500
  k=1: n=1720, P(updated)=0.500
  k=2: n=1720, P(updated)=0.500
  k=3: n=1720, P(updated)=0.500
  k=4: n=1720, P(updated)=0.500
Assignment-position cells:
  k0_pos0: n=1720, P(updated)=0.500
  k1_pos0: n=860, P(updated)=0.500
  k1_pos1: n=860, P(updated)=0.500
  k2_pos0: n=580, P(updated)=0.500
  k2_pos1: n=580, P(updated)=0.500
  k2_pos2: n=560, P(updated)=0.500
  k3_pos0: n=440, P(updated)=0.500
  k3_pos1: n=440, P(updated)=0.500
  k3_pos2: n=420, P(updated)=0.500
  k3_pos3: n=420, P(updated)=0.500
  k4_pos0: n=360, P(updated)=0.500
  k4_pos1: n=340, P(updated)=0.500
  k4_pos2: n=340, P(updated)=0.500
  k4_pos3: n=340, P(updated)=0.500
  k4_pos4: n=340, P(updated)=0.500
C_A-C_B context word delta mean=-0.023, range=[-2, 2]
Answer words updated=1.000, retained=1.378

## heldout_frame_all: 4800 rows, 358608 words
P(updated) = 0.500
Shared-context word mismatches: 0
By k:
  k=0: n=960, P(updated)=0.500
  k=1: n=960, P(updated)=0.500
  k=2: n=960, P(updated)=0.500
  k=3: n=960, P(updated)=0.500
  k=4: n=960, P(updated)=0.500
Assignment-position cells:
  k0_pos0: n=960, P(updated)=0.500
  k1_pos0: n=480, P(updated)=0.500
  k1_pos1: n=480, P(updated)=0.500
  k2_pos0: n=320, P(updated)=0.500
  k2_pos1: n=320, P(updated)=0.500
  k2_pos2: n=320, P(updated)=0.500
  k3_pos0: n=256, P(updated)=0.500
  k3_pos1: n=256, P(updated)=0.500
  k3_pos2: n=224, P(updated)=0.500
  k3_pos3: n=224, P(updated)=0.500
  k4_pos0: n=192, P(updated)=0.500
  k4_pos1: n=192, P(updated)=0.500
  k4_pos2: n=192, P(updated)=0.500
  k4_pos3: n=192, P(updated)=0.500
  k4_pos4: n=192, P(updated)=0.500
C_A-C_B context word delta mean=-0.133, range=[-1, 1]
Answer words updated=1.000, retained=1.317

## Skipped maps
  rf_death_place_10314: entity_b 'Pone' not whole-word in source
  rf_death_place_4727: entity_a 'Cormack' not whole-word in source
  rf_death_place_4636: entity_a 'Cormack' not whole-word in source
  rf_death_place_4681: entity_a 'Cormack' not whole-word in source
