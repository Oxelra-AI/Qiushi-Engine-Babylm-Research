# Strict Innovation Targets and Isolated Supervision Reallocation

Status: loose construction superseded; corrected masking checks and 1M execution check completed; mature downstream effect pending.

The loose map included duplicate and function/relation-like source-absent groups, set copyable groups to zero probability and raised source/filler probability to about 0.19. It therefore introduced more than the intended innovation-target intervention and was not used for the proposed 80M test.

The strict target had to be a rewrite word group in a both-visible pair, absent in normalized form from its source and every other row group, and content-like: not a stopword or relation/function cue, at least three characters and containing a letter.

Across 3,005 changed rows, strict innovations numbered 17,968 groups and 28,007 potential BPE labels per pass; copyable rewrites numbered 131,675 groups and 205,941 labels. Ordinary groups included 3,035 duplicate source-absent, 5,432 unique function-like source-absent, 3,314 short/empty source-absent and 261,644 source-only groups. Of 3,005 rows, 2,999 had a strict innovation.

Strict targets received probability 0.50; copied targets received 0.10240165872749962, computed from token-label mass; all other groups stayed at 0.15. The CPU check measured strict/copy/source/other/ordinary rates 0.4985/0.1002/0.1484/0.1456/0.1483 and selected-token mass ratio 1.000166 against standard WWM.

The 1,000,078-word check completed 26 updates with 34,467,424 parameters and final loss 6.850553. Selected/available counts were strict 1002/2000, copyable 1478/14323, source 4325/28561 and other rewrite 195/1280. These validate the intended reallocation, not downstream competence. Continuation required broad 70M/80M gains, particularly BLiMP/Supplement/EWoK, rather than another GlobalPIQA-nonparallel trade-off.
