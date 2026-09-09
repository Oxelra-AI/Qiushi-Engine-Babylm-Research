# Pivot Visibility: Matching and Continuation Exposure

Status: repaired batch invariants and short checks completed; 70M-to-80M treatment/control outcomes were pending.

Pivot-visible dependent masking (PVDM) tested whether keeping a relation pivot visible while predicting its dependent consequence improves context-conditioned relation learning. The control retained identical dependent targets while swapping the visible anchor. The desired signature was EWoK four-cell interaction and GlobalPIQA hard-row rank improvement without Entity/Supplement damage.

File-level target identity and per-event anchor-length matching were insufficient: overlapping event roles produced masked-token mass 35 versus 36 in one row. Requiring disjoint pivot/target/control roles repaired this. Over 8,192 real rows, 6,283 were selected (0.76697), with 11,450 dependent target groups (1.3977 per row). Both arms had effective mask rate 0.1490442801; target-label, target-replacement, original-ID, background and anchor-action mismatches were zero. Per-row mass absolute-sum/maximum differences were 0/0. Both arms had mask/random/keep counts 219,923/27,279/27,792, but true-pivot visible tokens were 11,911 versus 0.

Selected category counts were physical change 2,922, causal connector 2,485, temporal 2,363, spatial 2,132, negation 1,360 and comparative 188.

The nominal 70M checkpoint had already seen 70,040,037 words. Its preceding update ended at 69,999,840. Continuation therefore skipped 40,037 already-seen tail words, began at tail row 255 and trained 9,971,289 additional words through row 64,510, reaching the archived 80M boundary at 80,011,326 words. This prevented double counting.

The design used effective batch 256, microbatch 8, masked-token-weighted accumulation, symmetric AdamW reset, learning rate 0.001, warmup 0.06 and a 752-update possible continuation schedule, physically stopping at 80M. One-batch losses were 2.56399 for legacy WWM, 2.61174 for treatment and 2.69620 for control; these were execution checks, not competence scores. Extension to 100M required the predeclared relational signature and preservation criteria, not lower loss alone.
