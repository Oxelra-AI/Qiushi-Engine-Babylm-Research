# earlier analysis mirrored four-row family with assignment-position variation

Valid maps: 556/556
Relations: {'birth_year': 150, 'birthplace': 154, 'death_place': 190, 'founded_year': 10, 'located_in': 20, 'nationality': 16, 'occupation': 16}

The real assignment update is inserted at a balanced position among the k+1 operation sentences, with the same position for C_A and C_B inside each quad.
Each context still appears once with an updated answer and once with a retained answer; each query entity appears in both roles; k and assignment position have P(updated)=0.5.

## train_frame_seen: 46000 rows, 3860405 words
P(updated) = 0.500
Shared-context word mismatches: 0
By k:
  k=0: n=9200, P(updated)=0.500
  k=1: n=9200, P(updated)=0.500
  k=2: n=9200, P(updated)=0.500
  k=3: n=9200, P(updated)=0.500
  k=4: n=9200, P(updated)=0.500
Assignment-position cells:
  k0_pos0: n=9200, P(updated)=0.500
  k1_pos0: n=4600, P(updated)=0.500
  k1_pos1: n=4600, P(updated)=0.500
  k2_pos0: n=3080, P(updated)=0.500
  k2_pos1: n=3060, P(updated)=0.500
  k2_pos2: n=3060, P(updated)=0.500
  k3_pos0: n=2300, P(updated)=0.500
  k3_pos1: n=2300, P(updated)=0.500
  k3_pos2: n=2300, P(updated)=0.500
  k3_pos3: n=2300, P(updated)=0.500
  k4_pos0: n=1840, P(updated)=0.500
  k4_pos1: n=1840, P(updated)=0.500
  k4_pos2: n=1840, P(updated)=0.500
  k4_pos3: n=1840, P(updated)=0.500
  k4_pos4: n=1840, P(updated)=0.500
C_A-C_B context word delta mean=-0.026, range=[-2, 2]
Answer words updated=1.000, retained=1.232

## heldout_frame_all: 15360 rows, 1231144 words
P(updated) = 0.500
Shared-context word mismatches: 0
By k:
  k=0: n=3072, P(updated)=0.500
  k=1: n=3072, P(updated)=0.500
  k=2: n=3072, P(updated)=0.500
  k=3: n=3072, P(updated)=0.500
  k=4: n=3072, P(updated)=0.500
Assignment-position cells:
  k0_pos0: n=3072, P(updated)=0.500
  k1_pos0: n=1536, P(updated)=0.500
  k1_pos1: n=1536, P(updated)=0.500
  k2_pos0: n=1024, P(updated)=0.500
  k2_pos1: n=1024, P(updated)=0.500
  k2_pos2: n=1024, P(updated)=0.500
  k3_pos0: n=768, P(updated)=0.500
  k3_pos1: n=768, P(updated)=0.500
  k3_pos2: n=768, P(updated)=0.500
  k3_pos3: n=768, P(updated)=0.500
  k4_pos0: n=640, P(updated)=0.500
  k4_pos1: n=608, P(updated)=0.500
  k4_pos2: n=608, P(updated)=0.500
  k4_pos3: n=608, P(updated)=0.500
  k4_pos4: n=608, P(updated)=0.500
C_A-C_B context word delta mean=-0.031, range=[-2, 2]
Answer words updated=1.000, retained=1.203

