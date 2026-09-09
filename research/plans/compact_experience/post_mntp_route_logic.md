# Post-MNTP decision rules for isolated mechanism tests

## Why this note exists

A matched 8×480 clean-Qwen MLM-primary same-corruption MNTP auxiliary model was produced. Complete official-style nine-column evaluation of its frozen `chck_100M` was pending in this record. The scientific decision after that measurement must come from the observed column structure.

A negative result must **not** automatically promote the old Phase2 12×384/40k/LAMB package. That old package changed too many load-bearing factors together:

- depth/width: 8×480 → 12×384;
- vocabulary/tokenizer: 16k baseline tokenizer → 40k SentencePiece;
- optimizer: AdamW-like inherited recipe → LAMB with LR 0.007;
- sequence schedule: fixed seq256 → 64→128→256 curriculum;
- masking schedule: fixed WWM versus WWM→token switch in some variants;
- data state: earlier official-only or contaminated mix25, not the final clean-Qwen corpus.

Therefore it does not isolate a clean-Qwen × architecture interaction. It is useful as a warning and as a source of candidate factors, not as a ready fallback route.

## Current trusted scientific facts

1. The trusted admissible best remains clean-Qwen 8×480 seed43022 `chck_100M`, Overall `41.34429066479573`, AoA leaderboard `0.0`, below the visible 41.8 leader.
2. Same-window clean-Qwen second-view data is a real mechanism: aligned same-window beats exact selected-original duplication and separated-pair coexistence.
3. Whole-batch causal substitution is negative: causal15 Overall `41.2091`, causal50 `40.0999`; causal15 preserved AoA but traded Supplement/EWoK for GlobalPIQA/Reading.
4. The MLM-primary MNTP gradient probe showed same-corruption MNTP is not globally gradient-conflicting with MLM; the matched training comparison tests whether this compatible gradient is useful in transfer.
5. Tail-restart/mask allocation, row geometry, cluster packing, agreement objective, SWA, simple ordering, and contaminated mix25 are not current SOTA routes.

## How to interpret the MNTP columns

Use `scripts/analyze_mntp_full_eval.py` after the full evaluation payload exists. It compares the MNTP model to clean-Qwen, causal15, and the visible leader.

### If Overall crosses 41.8

Treat it as a real local SOTA candidate but not a final claim. The next work is strict integrity and replication:

- inspect payload completeness and endpoint identity;
- confirm `chck_100M` exposure and 100-checkpoint AoA ladder;
- run a fixed second seed replication;
- prepare official-server compatible submission only after replication or strong stability evidence.

### If Overall improves over clean-Qwen but remains below 41.8

Do not discard the signal. Read which columns moved.

- If EWoK/Entity rise without Supplement/Reading collapse, the auxiliary may better exploit same-window correspondence. Test one isolated strengthening, not a broad sweep.
- If only GlobalPIQA/Reading rise while Supplement/EWoK fall, it repeats the causal15 tradeoff and does not solve the main gap. Prefer representation separation or gradient shaping rather than more full 100M objective-ratio scans.
- If AoA becomes negative, do not mine endpoints or use AoA contents. Treat it as aggregate-balance damage and avoid continuation tricks that already failed.

### If Overall is flat or worse than clean-Qwen

Close this exact same-stack MNTP auxiliary construction as a SOTA route. The result still matters mechanistically: it tells whether gradient compatibility alone translated into official transfer.

The next run should isolate one factor. Candidate follow-ups must be compact and matched enough to change scientific understanding:

1. **Tokenizer isolation on current recipe:** clean-Qwen data with a clean 40k tokenizer while preserving the current 8×480 architecture, optimizer, WWM policy, fixed seq256, exposure, and seed. This separates vocabulary/segmentation from depth/optimizer/schedule.
2. **Minimal representation intervention on current tokenizer/recipe:** keep clean-Qwen data and 16k tokenizer, change one internal representation mechanism such as GEGLU FFN, an attention-output residual gate, or layer weighting. The purpose is to improve compositional transfer while preserving the known same-window data mechanism.
3. **Compact data-by-recipe interaction screen:** if a recipe factor really requires 12×384/40k/LAMB context, first run a small matched factorial or no-AoA screen rather than a full package: e.g., official vs clean-Qwen under exactly the same modified recipe, with the same tokenizer source and same exposure checkpoint. This isolates whether clean-Qwen pairs interact with the recipe.

## Preferred immediate next scientific action after a negative MNTP result

The most mechanism-faithful next action is not a leader-inspired bundle but an isolation experiment chosen from the observed failure mode:

- If the MNTP result shows objective tradeoff, the proposed follow-up is a representation intervention that preserves bidirectional reconstruction while adding directional/discourse capacity only through architecture or feature routing.
- If the main remaining gap is EWoK/Entity/GlobalPIQA and MNTP does not move them, the proposed follow-up is a compact tokenizer-isolation or recipe-factor screen around clean-Qwen, not a full unisolated 12×384/40k/LAMB run.
- If the score unexpectedly improves in the leader-gap columns, continue with a narrowly matched replication/strengthening of the same mechanism before changing data or architecture.

This note supersedes the overconfident language in `plans/fallback_phase2_architecture.md` that suggested clean-Qwen on the entire 12×384/40k/LAMB bundle as the direct fallback.
