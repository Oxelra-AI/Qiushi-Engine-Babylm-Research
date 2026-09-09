# Parent-82M Reproduction and Prediction Completeness

Status: completed checkpoint reproduction and score verification; fast-result packaging was incomplete in this record.

The reproduced scale-1.75 82M endpoint scored Overall 41.942481167385985. The contemporaneous displayed reference was rounded to 41.80, making the reported difference +0.142481. This historical comparison is not a current-ranking assertion.

From-corpus reproduction matched checkpoints at 20M, 50M, 80M, 82M and 100M bit-for-bit. The 82M model SHA-256 was `93ceb76adf5a33d349f1de33e988e6ed0c2b2a547dbd92cf83cc952f8e2591b3`. Hardened and collated scores agreed on all nine columns, and prediction row counts included the full 19-by-8005 AoA ladder.

The remaining defect was a missing top-level `fast_eval_results` block. A prediction file accepted by a format checker was not necessarily complete under the recorded evaluation contract: missing fast results could be scored as zero. This distinguished a prediction-carrier defect from a new model-score, training or data-legality issue. Completing the existing measured prediction material did not require retraining and could not justify changing the scientific endpoint.
