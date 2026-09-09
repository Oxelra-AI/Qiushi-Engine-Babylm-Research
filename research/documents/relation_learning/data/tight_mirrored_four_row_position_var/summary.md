# earlier analysis mirrored four-row family with assignment-position variation

Valid maps: 480/480
Relations: {'birth_year': 150, 'birthplace': 130, 'death_place': 190, 'founded_year': 10}

The real assignment update is inserted at a balanced position among the k+1 operation sentences, with the same position for C_A and C_B inside each quad.
Each context still appears once with an updated answer and once with a retained answer; each query entity appears in both roles; k and assignment position have P(updated)=0.5.

## train_frame_seen: 39700 rows, 3291820 words
P(updated) = 0.500
Shared-context word mismatches: 0
By k:
  k=0: n=7940, P(updated)=0.500
  k=1: n=7940, P(updated)=0.500
  k=2: n=7940, P(updated)=0.500
  k=3: n=7940, P(updated)=0.500
  k=4: n=7940, P(updated)=0.500
Assignment-position cells:
  k0_pos0: n=7940, P(updated)=0.500
  k1_pos0: n=3980, P(updated)=0.500
  k1_pos1: n=3960, P(updated)=0.500
  k2_pos0: n=2660, P(updated)=0.500
  k2_pos1: n=2640, P(updated)=0.500
  k2_pos2: n=2640, P(updated)=0.500
  k3_pos0: n=2000, P(updated)=0.500
  k3_pos1: n=1980, P(updated)=0.500
  k3_pos2: n=1980, P(updated)=0.500
  k3_pos3: n=1980, P(updated)=0.500
  k4_pos0: n=1600, P(updated)=0.500
  k4_pos1: n=1600, P(updated)=0.500
  k4_pos2: n=1580, P(updated)=0.500
  k4_pos3: n=1580, P(updated)=0.500
  k4_pos4: n=1580, P(updated)=0.500
C_A-C_B context word delta mean=-0.033, range=[-2, 2]
Answer words updated=1.000, retained=1.146

## heldout_frame_all: 13280 rows, 1076904 words
P(updated) = 0.500
Shared-context word mismatches: 0
By k:
  k=0: n=2656, P(updated)=0.500
  k=1: n=2656, P(updated)=0.500
  k=2: n=2656, P(updated)=0.500
  k=3: n=2656, P(updated)=0.500
  k=4: n=2656, P(updated)=0.500
Assignment-position cells:
  k0_pos0: n=2656, P(updated)=0.500
  k1_pos0: n=1344, P(updated)=0.500
  k1_pos1: n=1312, P(updated)=0.500
  k2_pos0: n=896, P(updated)=0.500
  k2_pos1: n=896, P(updated)=0.500
  k2_pos2: n=864, P(updated)=0.500
  k3_pos0: n=672, P(updated)=0.500
  k3_pos1: n=672, P(updated)=0.500
  k3_pos2: n=672, P(updated)=0.500
  k3_pos3: n=640, P(updated)=0.500
  k4_pos0: n=544, P(updated)=0.500
  k4_pos1: n=544, P(updated)=0.500
  k4_pos2: n=544, P(updated)=0.500
  k4_pos3: n=512, P(updated)=0.500
  k4_pos4: n=512, P(updated)=0.500
C_A-C_B context word delta mean=0.133, range=[-2, 2]
Answer words updated=1.000, retained=1.301

