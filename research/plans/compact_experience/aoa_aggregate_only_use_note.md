# all mask endpoint interpretation AoA aggregate-only use note

The mask endpoint full eval summary/048 full-eval payloads preserve detailed AoA runner internals so the local evaluator can be reproduced and checked. Some of those JSON fields contain official AoA/CDI word lists and curve-fitting internals. They are not research material for training, schedule design, corpus selection, or model-selection beyond the completed official-style aggregate score.

Evaluation-use constraint:

- permitted: final aggregate `aoa_raw_correlation`, `aoa_leaderboard_score`, `aoa_status`, row counts, finite-score status, and complete Overall arithmetic as completed measurement of a frozen model;
- not permitted: using official AoA/CDI word identities, child curves, per-word model AoA values, or checkpoint AoA curves to choose text, choose masks, choose schedules, tune checkpoints, or construct new objectives.

The inverse-priority result should therefore be interpreted only at aggregate-score level: the true 100M endpoint has AoA raw `-0.18344781331440352`, leaderboard AoA `-18.344781331440352`, and Overall `39.714154598526164`; the endpoint-frozen 95M measurement has AoA raw `-0.1719406337063024`, leaderboard AoA `-17.19406337063024`, and Overall `39.961218980736284`. This is enough to close inverse-priority as an immediate SOTA route without using official AoA item content.
