# official geometry noaoa interpretation official row-geometry interpretation model

This note was written while the official-geometry no-AoA rerun was active. It uses only already inspected training/audit facts and literature/source notes, not SuperGLUE, AoA, child curves, AoA predictions, or new trajectory outcomes.

## Active scientific question

The current experiment compares two official-only corpora under the same model family, tokenizer, WWM recipe, seed/init convention, word budget, and batch256 launcher:

- **G0 official160**: original 160-word row geometry. official geometry exposure and steps audit: 62,500 rows per 10M pool, 14,896,688 untruncated tokenizer tokens, 14,444,429 trainer-visible tokens under seq256, 18,253 rows over seq256 (29.2048%), 2,442 training steps for 100M words.
- **G3 cap120 official geometry**: official-only lengthmatched rows from the contextual cap120 control. official geometry exposure and steps audit: 71,898 rows per 10M pool, mean 139.0859 words, the same 14,896,688 untruncated tokenizer tokens, 14,637,292 trainer-visible tokens, 10,320 rows over seq256 (14.3537%), 2,809 training steps for 100M words.

Thus G3 is not a pure cap-length experiment. It changes row boundary frequency, truncation loss, visible-token exposure, number of optimizer updates, batch-word distribution, and learning-rate time. A raw G3--G0 gain can be scientifically useful, but only as a whole-recipe signal until update/batch sensitivity is tested.

## Relation to prior BabyLM evidence

The local evidence aligns with prior BabyLM reports that data format and sequence length can be load-bearing under small data:

- Salhan et al., *What is the Best Sequence Length for BabyLM?*, argue that sequence length effects are task- and architecture-dependent. Their abstract and results state that shorter sequences are sufficient or better for grammatical generalization / BLiMP-style tasks, while longer contexts can benefit Entity Tracking, Wug/morphological analogy, and Reading. They explicitly note that shorter sequences also mean more updates under fixed data, so length is confounded with update count unless batch/learning-rate time is controlled.
- The 2023 BabyLM findings summarize two relevant submissions: Edman and Bylinina found reducing context length to 32 produced significant consistent improvements in their setting, and the McGill submission found data-format choices such as not using sequence packing, using sentence examples, avoiding truncation, and reducing maximum sequence length were highly effective.
- The 2025 findings summary emphasizes that training cadence can rival architecture: smaller effective batches and optimizer-state resets changed generalization; this matters because G3 has 2,809 updates at batch256 whereas G0 has 2,442.

These sources make the G0/G3 experiment a plausible high-value probe, not a local artifact: it tests whether better example geometry and lower truncation can move the BabyLM Strict-Small frontier. But they also warn against reading BLiMP gains alone as a general principle.

## Outcome interpretation before seeing the rerun

Use only BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, and Reading to compare trajectories. Do not use SuperGLUE or AoA until targets are frozen.

1. **Broad transferable G3 signal**: If G3's best no-AoA equal7 is above both G0 and the clean-Qwen reference, and the gain is not concentrated only in BLiMP/Supplement, run exactly one minimal update-sensitivity wave before full eval. A useful sensitivity pair is G3 near G0 update count (e.g. batch295/lr2442) and/or G0 near G3 update count (batch223/lr2809). The purpose is to test whether the apparent geometry signal survives when update/LR-time is changed.

2. **Syntax-only G3 exchange**: If G3 improves BLiMP/Supplement but remains below clean-Qwen on equal7, or damages Entity/GlobalPIQA/Reading/COMPS enough that the aggregate is not competitive, stop the official-geometry branch as a frontier route. The evidence can still explain why cap120 control looked strong, but it should not become cap-length optimization.

3. **No G3 advantage over G0**: If G3 does not clearly beat G0 on no-AoA equal7, the shorter-row/topology route is closed at batch256. The next useful work should not be matched-update G3; it should return to a stronger load-bearing layer such as acquisition curriculum, tokenizer/architecture, or a lower-dose second-view experiment only if tied to a better substrate.

4. **G0 itself unusually strong**: If official160 G0 batch256 reaches or exceeds the clean-Qwen no-AoA profile, interpret it as a random-initialization/launcher/row-order reproduction clue rather than a new mechanism. It may justify a corrected full nine-column evaluation of the frozen G0 checkpoint, but not a claim of innovation unless replicated or mechanism-linked.

## Decision discipline

- Never combine scores from different models/checkpoints into a synthetic overall.
- The clean complete reference remains `qwen_clean_aligned` seed43022 chck_100M Overall 41.34429066479573 until a new corrected nine-column full eval exceeds it.
- The visible target remains the 41.8 leaderboard leader. A no-AoA equal7 gain is only a reason to run final measurement, not a result.
- If the official geometry signal is not broad enough, keep the prepared A0--A3 acquisition curriculum and low-dose second-view designs as candidates, not as an automatic GPU queue.
