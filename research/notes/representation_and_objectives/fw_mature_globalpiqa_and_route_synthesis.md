# execution synthesis — FW mature GlobalPIQA readout and route synthesis

## What changed

The two completed FW endpoints (`chck_100M`) were read directly with the all-option GlobalPIQA scorer:

- `fw_compact_fullbatch_seed43022`: compact same-proposition recurrence anchor, shared legal 16k tokenizer, full-batch trainer.
- `fw_breadth_rowblock_fullbatch_seed43022`: row-block whole-sentence independent FineWeb breadth, same tokenizer/trainer/seed coordinate.

The readout is CPU-only and uses existing checkpoints. It does not change training or tune a submission scorer.

## Mature 100M GlobalPIQA split

| arm | GlobalPIQA_parallel | GlobalPIQA_nonparallel | aggregate | fw globalpiqa relevant substrate hard52 acc | hard52 rank1 | hard52 mean top-minus-correct |
|---|---:|---:|---:|---:|---:|---:|
| compact | 24.27 | 53.00 | 38.64 | 3.85 | 2 | 1.717 |
| row-block breadth | 29.13 | 45.00 | 37.06 | 5.77 | 3 | 1.433 |

Row-block breadth minus compact at 100M:

- parallel: +4.854 points
- nonparallel: -8.000 points
- aggregate GlobalPIQA: -1.573 points
- hard52 mean top-minus-correct: -0.284 nats
- hard52 rank1: +1 row
- hard52 small-wrong-margin rows within 0.50 nats: +4 rows

## Interpretation

This is not a simple compact victory. Row-block breadth creates the first meaningful movement in the deep GlobalPIQA hard-row preferences: it reduces hard52 wrong-option dominance and reduces rank4 failures. That matters because previous compliant, depth, SGCR, and inherited-tokenizer endpoints all had stable deep-rank failure on these rows.

However, the movement is not yet a useful endpoint mechanism. It is paid for by a larger loss on nonparallel GlobalPIQA and by the 70M cheap vector's broad losses, especially Entity. At 100M, compact has higher aggregate GlobalPIQA despite weaker hard-parallel ranks because it preserves nonparallel accuracy. The desired combination from the strategist note — compact-level broad capability together with improved hard-row relational ranks — does not appear in the row-block breadth arm.

The 100M compact aggregate GlobalPIQA (38.64) is essentially unchanged from the 70M compact cheap value (38.605), so the completed 70M cheap vector remains a credible arithmetic anchor for SOTA plausibility. With compact 70M cheap7 = 42.657, reaching Overall 41.80 would require SuperGLUE+AoA = 77.60. The known local range is far lower: compliant complete endpoints observed here are around 67.08–68.23, while the non-submission inherited-tokenizer endpoint reached 71.04. Row-block breadth is farther away, requiring SuperGLUE+AoA = 81.98 at its 70M cheap7.

Thus compact and row-block breadth should not receive full official endpoint evaluation solely from the current score arithmetic. Their value is mechanism evidence: compact preserves broad GlobalPIQA but fails the deep parallel relation surface; row-block breadth moves the deep parallel surface but damages broader capability.

## Pending evidence that can still matter

1. `s106_t10_tool1` is still the companion analysis interleaved whole-sentence breadth training task. It is the layout-matched test of whether row-block breadth's tradeoff is caused by row-block layout or intrinsic to replacing same-proposition compact recurrence with independent FineWeb coverage. Do not duplicate or restart it.
2. `s112_t16_tool1` is the CPU EWoK four-cell interaction reader for the two completed 100M endpoints. Its result will show whether row-block breadth's EWoK movement also improves the stable conditional-reversal failure mode or only moves domain-weighted surface means.

## Next execution logic

After interleaved-arm training completes:

- First verify training completion and stream/tokenizer/trainer identity against the compact anchor match preflight.
- Run the minimum cheap readout and the GlobalPIQA margin wrapper. Use direct checkpoint directories for nonfinal checkpoints; for `chck_100M`, parent and checkpoint were verified equal for companion analysis but direct paths remain safer.
- Compare interleaved against both arms on three quantities together: cheap7/broad columns, GlobalPIQA hard52 rank/margin, and EWoK stable conditional-reversal movement.

Research interpretation:

- If interleaved preserves compact-like broad capability while keeping row-block's hard-row rank/margin improvements, layout/source alternation becomes an active mechanism worth developing.
- If interleaved resembles compact, row-block's relational movement likely came from its row-block independent-coverage arrangement and is not enough for SOTA.
- If interleaved resembles row-block breadth, independent coverage carries an intrinsic broad-capability cost under this 813k-word FW allocation.
- If no arm shows the desired combination, stop extending FW allocation variants. Use the anchor matched controls ultra-clean 30k transition/control assets only as a small probe to understand a stronger mechanism, not as authorization for a noisy full 100M relation-corpus route.

## Evidence files

- 100M GlobalPIQA mature synthesis: `research/notes/representation_and_objectives/fw_100m_globalpiqa_mature_synthesis.md`
- 100M GlobalPIQA JSON: `experiments/archive/representation_and_objectives/data/fw_100m_globalpiqa_mature_synthesis/fw_100m_globalpiqa_mature_synthesis.json`
- Source margin output: `experiments/archive/representation_and_objectives/data/fw_globalpiqa_margin_reader/globalpiqa_margin_reader_results.json`
- 70M GlobalPIQA margin synthesis: `research/notes/representation_and_objectives/fw_70m_globalpiqa_margin_synthesis.md`
- 70M cheap vector and SOTA arithmetic: `research/notes/representation_and_objectives/fw_sota_plausibility_calculator.md`
