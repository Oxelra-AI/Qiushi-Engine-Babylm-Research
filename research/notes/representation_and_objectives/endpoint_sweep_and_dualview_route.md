# endpoint sweep and dualview route — bounded scale1.75 peak establishment and dual-view route context

## Why the checkpoint sweep is scientifically justified

The scale1.75 80M endpoint is the strongest completed compliant score-bearing object in this comparison: Overall 41.77163494053074, only 0.028365 below the visible 41.80 frontier. Its 100M endpoint regresses to 41.5699554881152. A complete 1M-spaced checkpoint ladder was saved, but the actual peak of this already-trained trajectory near 80M has not been established.

The endpoint sweep and dualview route sweep therefore evaluates only the official-compatible zero-shot/Reading cheap7 surface for chck_77M, chck_78M, chck_79M, chck_81M, chck_82M, and chck_83M. chck_80M is not rerun because scale1p75 bottleneck and next route already completed its full score. The cheap7 threshold that would justify complete SuperGLUE/AoA verification under the observed 80M SuperGLUE and AoA=0 is 43.848612219317616. This is a bounded endpoint-establishment experiment: no new training, no new corpus, and no scan outside this small neighborhood unless a checkpoint exceeds the threshold.

Launched evaluation command:

`python3 -B experiments/archive/representation_and_objectives/scripts/scale1p75_checkpoint_sweep.py`

Plan file: `experiments/archive/representation_and_objectives/data/scale1p75_checkpoint_sweep/summary/sweep_plan.json`.

## Why this does not replace the learning-principle route

A neighboring checkpoint above the score frontier would be important to preserve and verify, but exposure selection alone would not satisfy the requested scientific advance. Scale1.75 remains a broad-capability substrate with matched EWoK and GlobalPIQA hard-surface costs. earlier analysis-160 evidence shows those costs are embedded in the trained trajectory/representation, not an inference-time adapter amplitude or protected late-readout defect.

The representation-forming route must still be tested on natural transfer. sparse relation aux calibration dual-view evidence is the strongest current route because it directly addresses the transfer problem left by edit-state and targeted-masking failures:

- sparse relation aux preflight source-token-absent edit-state: frozen true source is worse than decoy in raw NLL (-0.312), but a detached private readout still gets +0.836 held-out advantage and +1.240 on high-error items. This shows non-copy source→compact transformation information exists.
- sparse relation aux calibration all-edit transfer: a shared readout trained jointly on source-conditioned and source-free states with true alignment reaches NLL 7.3997 on source-free held-out, beating shared shuffled by 1.007 and source-free-only by 1.132.
- sparse relation aux calibration source-absent transfer: shared true reaches NLL 5.5977, beating shared shuffled by 0.323 and source-free-only by 0.248, while single-condition transfer does not separate true from shuffled.

This matters because older targeted source-absent innovation masking damaged cheap7 by about -1.053 at 80M despite local target motivation. The dual-view pathway is not merely reallocating WWM target mass; it tries to make source-conditioned structure train features that remain useful when the source view is absent.

## Current coordination

companion analysis owns only the bounded scale1.75 sweep. companion analysis owns the dual-view shared private pathway construction. companion analysis should not duplicate that implementation. If the sweep does not produce a checkpoint whose cheap7 exceeds 43.848612, scale1.75 trajectory scanning should stop and the next high-value work is to read dual-view train-time screen when it appears, then compare its natural EWoK/GlobalPIQA hard-surface behavior against the existing ACS, Route B, scale1.75, and targeted-masking boundaries.
