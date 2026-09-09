# semantic repair frontier and refill plan — semantic-force repair frontier

Selected compact-reinvest block: 12155 rows / 423511 pair words. Unused accepted compact pool: 6527 rows / 221722 pair words.

## Rule frontier
- force_only: removes 113387 pair words (26.77%); unused passing 156174 words; available:needed=1.38; refill=True; domain shortfalls=2
- force_plus_pronoun: removes 124339 pair words (29.36%); unused passing 148447 words; available:needed=1.19; refill=True; domain shortfalls=3
- force_content_ge_0.45: removes 113387 pair words (26.77%); unused passing 106242 words; available:needed=0.94; refill=False; domain shortfalls=5
- force_content_ge_0.50: removes 128019 pair words (30.23%); unused passing 102361 words; available:needed=0.80; refill=False; domain shortfalls=8
- force_content_ge_0.55: removes 173452 pair words (40.96%); unused passing 89865 words; available:needed=0.52; refill=False; domain shortfalls=8
- force_content_ge_0.60: removes 209463 pair words (49.46%); unused passing 78950 words; available:needed=0.38; refill=False; domain shortfalls=8
- force_content_ge_0.65: removes 263879 pair words (62.31%); unused passing 62850 words; available:needed=0.24; refill=False; domain shortfalls=8
- force_content_ge_0.70: removes 301649 pair words (71.23%); unused passing 51915 words; available:needed=0.17; refill=False; domain shortfalls=8
- force_pronoun_content_ge_0.45: removes 124339 pair words (29.36%); unused passing 102029 words; available:needed=0.82; refill=False; domain shortfalls=8
- force_pronoun_content_ge_0.50: removes 138151 pair words (32.62%); unused passing 98289 words; available:needed=0.71; refill=False; domain shortfalls=8
- force_pronoun_content_ge_0.55: removes 181896 pair words (42.95%); unused passing 86366 words; available:needed=0.47; refill=False; domain shortfalls=8
- force_pronoun_content_ge_0.60: removes 216287 pair words (51.07%); unused passing 76060 words; available:needed=0.35; refill=False; domain shortfalls=8
- force_pronoun_content_ge_0.65: removes 268734 pair words (63.45%); unused passing 60910 words; available:needed=0.23; refill=False; domain shortfalls=8
- force_surface_content_ge_none: removes 121567 pair words (28.70%); unused passing 152042 words; available:needed=1.25; refill=True; domain shortfalls=3
- force_surface_content_ge_0.55: removes 179802 pair words (42.46%); unused passing 87085 words; available:needed=0.48; refill=False; domain shortfalls=8
- force_surface_content_ge_0.60: removes 214459 pair words (50.64%); unused passing 76367 words; available:needed=0.36; refill=False; domain shortfalls=8
- force_surface_content_ge_0.65: removes 267357 pair words (63.13%); unused passing 60999 words; available:needed=0.23; refill=False; domain shortfalls=8

## Read for future repair
- The strongest refillable repairs are force_only and force_plus_pronoun; they keep the density intervention mechanically feasible without new teacher generation.
- Content-recall floors at 0.60 or above are not refillable from the unused accepted pool and would also erase much of the compressed-view block.
- If a later exact temporal comparison selects semantic repair, the next construction should use a domain-aware replacement list rather than simple greedy refill, because preserving physical/causal exposure matters for EWoK.

Machine-readable output: `experiments/archive/frontier_consolidation/data/semantic_repair_frontier/semantic_repair_frontier.json`; table: `experiments/archive/frontier_consolidation/data/semantic_repair_frontier/semantic_repair_frontier.csv`
