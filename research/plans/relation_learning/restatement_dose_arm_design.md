# calibrated route options: Restatement dose arm — experiment design

Created UTC: 2026-09-07

## Scientific question

Does the strongest established relation effect (aligned restatement, ALN) continue to pay
when its dose is increased from 16.6% to ~25% of the word budget by replacing filler?
Or does the return saturate or cross over? Either answer is a paper-level result.

## Context from existing data

| Measurement | Value | Source |
|---|---|---|
| ALN pair words | 1,656,800 per 10M (16.568%) | COMPACT_EXPERIENCE earlier analysis |
| ALN Overall effect (vs OFF) | +0.35 at 1 seed (AoA clean) | COMPACT_EXPERIENCE |
| ALN ordinary-fit movement on later 6,992 corpus-derived rows | −0.0125, −0.0128 at 2 seeds, but dose training and leakage repair shows these rows were re-admitted as trained text in OFF/ALN and the movement concentrates on inherited pair-text-hit rows; use as trained-text fit/reallocation evidence, not clean broad-fit calibration | dose training and leakage repair |
| ALN column vector | BLiMP +2.1, Supplement +1.6, EWoK +1.8, Entity +1.4, GlobalPIQA −4.5 | COMPACT_EXPERIENCE |
| Compact-view-reinvest (additional ~4.2%) | ~+0.07 Overall at 1 seed | frontier_consolidation |
| COMPACT_EXPERIENCE pair word cap | 25% (= 2,500,000 words per 10M) | COMPACT_EXPERIENCE earlier analysis |
| Available new words before cap | ~843,200 words (~19,163 pairs) | calculation |

## Stream topology

The SOTA stream (compact-view-reinvest) has per 10M pass:
- `qwen_pair_packed`: 12,236 rows, 1,656,800 pair words (37,594 pairs)
- Compact-view-reinvest block: 2,647 rows, 423,520 words
- Common filler: 49,857 rows, ~7,919,680 words (after compact block deduction)
- Total: 64,740 rows, 10,000,000 words
- 10 passes → 647,400 rows, 100,000,000 words

The dose arm replaces FILLER rows with new pair-packed rows. ALN and compact blocks
are preserved unchanged.

## Source material for new pairs

COMPACT_EXPERIENCE earlier analysis generated 110,608 rewrite outputs from the official pool:
- 37,704 clean pairs (37,594 selected for training)
- 72,904 rejected for: copy_overlap (44,347), entity_recall (20,379),
  source_incomplete_or_fragment (18,218), rewrite_incomplete_or_fragment (4,051),
  low_content_overlap (3,026), number_mismatch (1,771), repetition (1,526),
  len_ratio (312), meta_prefix (258), bad_terminal_fragment (118), etc.

**Salvageable originals**: Exclude `source_incomplete_or_fragment` (18,218) and
`too_short` (69) since the SOURCE itself is unsuitable. Remaining: ~54,617
originals whose rewrites were rejected but whose source text is adequate.
These can be re-prompted with Qwen3.5-9B and an improved prompt.

Expected acceptance rate with improved model + prompt: 25-40% (COMPACT_EXPERIENCE had 34.1%
with first-pass GPT-4/Qwen-2). At 33%: ~18,024 new pairs → ~794,257 words at
44 words/pair.

Target: ~19,000 new pairs to reach the 25% cap.

## Generation design

**Model**: Qwen3.5-9B (same as state-update generation, validated quality)

**Prompt template** (improved from COMPACT_EXPERIENCE earlier analysis extra prompt):
```
Rewrite the following sentence to mean exactly the same thing, using
different words and sentence structure.

Rules:
- Keep every named entity, proper name, speaker label, and number unchanged.
- Do not add, remove, or change any facts.
- Use different vocabulary and phrasing — do not copy more than 3 consecutive
  words from the original.
- Output one complete rewritten sentence only. No explanation.

Original: {source_sentence}

Rewritten:
```

Key improvements over COMPACT_EXPERIENCE:
1. Explicit 3-consecutive-word copy limit (addresses the 60.8% copy_overlap rejection)
2. Same entity preservation requirement
3. Qwen3.5-9B vs GPT-4/Qwen-2 (better instruction following at this scale)

**Validation filters** (same as COMPACT_EXPERIENCE earlier analysis + improvements):
- Content overlap: [0.14, 0.94]
- Entity recall: ≥0.75 for long entities, 1.0 for short
- Length ratio (rewrite/source): [0.45, 2.1]
- Min rewrite words: 6, max: 70
- No source_incomplete_or_fragment
- No meta prefix artifacts
- No exact substring > 6 tokens (from state-update validator)
- No temporal/persistence cue injection (from state-update validator)

**Packing**: Same three-pair packer as COMPACT_EXPERIENCE earlier analysis. New pairs are packed into
rows targeting ~160 words. Source register is preserved within rows where possible.

## Materialization

A filler-row-aware materializer (adapted from qwen_row_materializer.py):
1. Identify available filler rows (not `qwen_pair_packed`, not compact-view-reinvest)
2. Replace selected filler rows with new pair-packed rows
3. Match word budgets by adjusting how many filler rows are replaced
4. Handle token overflow by reverting pairs (same as state update intervention prestate)
5. Record exact accounting: pairs replaced, overflow count, word delta

## Dose levels

**Primary scientific curve**: materialize both intermediate and cap-near points.

- ~21% total Qwen pair fraction: add about 443,200 pair words per 10M pass (roughly 4.4% of the pool).
- ~25% total Qwen pair fraction: add about 843,200 pair words per 10M pass (roughly 8.4% of the pool).

The 21% arm is not optional: the fixed-budget result is the shape of the curve through OFF, inherited ALN (~16.6%), ~21%, and ~25%. A single 25% endpoint cannot distinguish a continuing approximately linear return from saturation or crossover.

## Readout priority (repaired before dose scores land)

dose training and leakage repair changed the role of ordinary-fit measurements. The old ALN→OFF movement on the later 6,992-row set is not a clean calibration for up-dose continuation because both old streams re-admitted those rows and the ALN movement is larger on rows whose text appears in inherited pair sources/rewrites. The dose readout therefore treats ordinary MLM loss as a **cost ordinate** of fixed-budget substitution, not as the main proof that the relation was learned.

1. **Clean ordinary-fit cost ordinates**, reported as paired within-seed values at 100M for base0, dose21, and dose25. Current required axes are the seed43222 entity clean integration 2,647-row set and the 1,743-row subset with no inherited ALN pair-text hits; if a same-source raw BabyLM development split is confirmed and screened, it becomes the main clean broad-fit axis. These numbers describe how much ordinary prediction fit was paid for the extra restatement relation. A flat or slightly worse clean-axis loss at dose25 is compatible with the expected substitution law if practiced-class readouts move: VIEW was worse than CLEAN on ordinary fit while winning relation-sensitive deployment. A reproduced increase larger than about +0.015 nats on both clean axes is a real cost signal; a reproduced increase around +0.025 nats or larger, especially if the future dev axis agrees, means the dose route should not spend official/coherent evaluation unless the task vector gives an unusually strong compensating gain. Differences inside roughly ±0.005 nats are treated as ordinary-fit neutral at this measurement scale.

2. **Practiced-class source-conditioned faces**, reported by dose and seed before expensive leaderboard-style evaluation. The expected signal is dose-ordered improvement with diminishing return on source-recurring changed-form use: compact overlap/nonoverlap T/U/N or G, Wikipedia source-recurring target terms, and the repaired state-margin T-U uptake face. The companion source-absent/nonoverlap terms should remain approximately flat; growth of source-absent damage would reveal a restatement-side liability rather than useful reusable knowledge. Natural-copy gain is read separately because exact recurrence and restatement can trade off there.

3. **Cheap7 task-family vector at 100M for both seeds**, read column by column against the compact-view-reinvest base with any missing base cells scored alongside. The continuation signal for spending replay and official hours is not an absolute ordinary-fit win; it is whether dose25 improves or preserves the columns previously moved by aligned restatement and compact-view practice—especially BLiMP, Supplement, EWoK, and Entity—without a reproduced GlobalPIQA cost large enough to erase the vector. Reading and COMPS remain part of the vector but have lower route-setting power unless their movement is large and replicated. Aggregates are secondary summaries, not route-setting evidence.

4. **Coherent replay and full official evaluation** are reserved for a 100M dose25 candidate whose two-seed cheap7 vector is promising relative to base and whose clean ordinary-fit cost is not broadly damaging. Dose21 remains a curve-shape point: it can reveal saturation, diminishing return, or a crossover even when dose25 is the practical candidate. A single biased fit-axis improvement must not by itself trigger replay, and a small clean-axis loss must not by itself discard a candidate whose practiced-class and cheap7 vectors move in the intended direction.

## Timeline estimate

| Phase | Time | GPU |
|---|---|---|
| Extract rejected originals, build prompts | 1h | CPU |
| Generate rewrites (Qwen3.5-9B, ~55k prompts) | 3-5h | 1 GPU |
| Validate, pack, materialize | 1h | CPU |
| Two-seed training (parallel) | 2.5h | 2 GPUs |
| Cheap7 evaluation (2 seeds × 2 checkpoints) | 1h | 1 GPU |
| Total wall time (with parallelism) | ~9h | — |

## Pre-registered prediction after leakage repair

The cleaned nested dose arm primarily tests how much relation-typed source-conditioned readout additional aligned restatement buys under fixed budget, and what ordinary prediction cost accompanies it.

Expected pattern:
1. Dose21 and dose25 should improve practiced source-recurring changed-frame readouts relative to base0, with dose25 ≥ dose21 and diminishing return possible.
2. Source-absent substitution and compact nonoverlap damage should remain near base if the asymmetry holds; a monotonic worsening there would be evidence that restatement has its own changed-form liability at high dose.
3. Clean ordinary-fit losses may be flat or slightly worse because filler is displaced. They are interpreted as cost, not as the main success signal. A small loss increase within the ranges above is compatible with useful relation learning.
4. The cheap7 vector should be read by columns. The most relevant positive columns are BLiMP, Supplement, EWoK, and Entity; GlobalPIQA is the watched cost column. Overall movement is only meaningful after the vector and seed agreement are known.

If dose25 improves practiced-class readouts but not clean ordinary fit, the scientific result is not saturation of aligned restatement in general; it is that extra restatement bought relation-conditioned uptake rather than broad ordinary MLM fit. If dose21 improves but dose25 does not, the result supports a dose crossover or saturation of this relation under the compact-view substrate. If neither dose improves practiced-class readouts, the up-dose construction failed to add the intended reusable relation despite legal accounting and should not proceed to replay.

## Files to produce

- `extract_rejected_originals.py` → rejected originals JSONL
- `build_dose_arm_prompts.py` → generation prompts JSONL
- `generate_dose_arm_rewrites.py` → generation launcher
- `validate_dose_arm_outputs.py` → validation with improved filters
- `pack_and_materialize_dose_arm.py` → packing and stream creation
- `launch_dose_arm_training.py` → two-seed training launcher
