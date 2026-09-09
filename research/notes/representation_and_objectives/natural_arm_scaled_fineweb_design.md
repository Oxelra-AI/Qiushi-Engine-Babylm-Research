# live fineweb core fact filter natural-arm FineWeb scale design

## Design Correction

A generated view can beat artificial same-source repetition and still fail to beat the trusted natural allocation. That happened in SimpleWiki semantic-view endpoint, and the same risk is visible in near-view FineWeb result. Therefore the next scaled FineWeb study must include a strong natural lengthmatched arm in the same materialization family.

## near-view result read against clean-Qwen

`near_view - near_repeat` gives Entity_full +3.57 and equal7_full_entity +0.5543. But this is a repeat-control contrast.

Against the COMPACT_EXPERIENCE clean-Qwen natural reference, `near_view` has full-Entity equal7 delta -0.233571; component deltas: BLiMP -0.02, Supplement 0.36, EWoK -1.01, Entity 1.37, COMPS -0.61, GlobalPIQA -2.485, Reading 0.76.

`near_repeat` against the same reference has full-Entity equal7 delta -0.787857; this shows that the repeat arm is itself a weak baseline for SOTA-facing judgment.

## What pending seqsafe96 run already tests

The pending source-breadth contrast has a natural lengthmatched control: qwen pairs 1,656,800 words, official lengthmatched block 1,753,280 words, identical official tail 6,589,920 words. Its FineWeb source-only arm replaces the lengthmatched block with 1,753,280 cached FineWeb words. It does not contain a source-plus-view arm.

## Next experiment shape

After the source-breadth comparison completes, the proposed next H100 study is one matched family:

- A_natural_lengthmatched_cleanqwen_slice: inherited clean-Qwen qwen_pair_packed rows preserved; a coherent official/non-Qwen slice of the same word budget is repacked to the changed-block row lengths; no FineWeb, no source repetition.
- B_fineweb_source_only: same common filler and same changed-block word budget; selected FineWeb source rows only, arranged with the same row-length sequence as A.
- C_fineweb_source_plus_faithful_view: same common filler, same changed-block word budget, and preferably the same FineWeb source set as B; one faithful near or compact view is adjacent to its source.

This separates source breadth (A→B), faithful-view utility (B→C), and net movement over the strong coordinate (A→C). A rewrite-versus-repeat-only study cannot answer the SOTA question.

If compact views are chosen and the saved word budget is reinvested into additional sources, include the source-only reinvest counterpart as a fourth arm; otherwise source diversity and generated-view effects are mixed together.

## Scale and source-resource calculation

Current accepted near resource: 446,188 source words + 415,097 rewrite words = 861,285 pair words. Current accepted compact resource: 400,665 source + 244,568 rewrite = 645,233 pair words. These are useful but smaller than a 2.5M–3.5M changed-block study.

live relation-dense extraction found 80,459 dense words from 708,520 scanned doc words (yield 11.36%). Samples still show some web-fragment noise, so stricter source filtering must precede any large generation.

| changed-block words | near source needed | compact source needed | dense doc words for near source | dense doc words for compact source | common filler if qwen pairs preserved |
|---:|---:|---:|---:|---:|---:|
| 1,750,000 | 906,586 | 1,086,683 | 7,983,375 | 9,569,305 | 6,593,200 |
| 2,500,000 | 1,295,123 | 1,552,404 | 11,404,822 | 13,670,432 | 5,843,200 |
| 3,500,000 | 1,813,172 | 2,173,366 | 15,966,749 | 19,138,609 | 4,843,200 |
| 5,000,000 | 2,590,246 | 3,104,809 | 22,809,644 | 27,340,873 | 3,343,200 |

## How to use the pending source-breadth result

- If the 17.5% source-only arm gives a substantial EWoK+Entity+COMPS+GlobalPIQA gain without erasing BLiMP/Supplement/Reading, build a larger natural/source/source+view family, likely 3.5M changed-block words.
- If it is small positive, do not call source breadth dead; first improve live FineWeb source quality and test a 2.5M natural/source/source+view family.
- If it is flat or damaging, do not extend cached source-only repetition. A moderate natural/source/source+view study is only worth training after source quality or view mechanism has changed.

JSON: `experiments/archive/representation_and_objectives/data/natural_arm_design_analysis/natural_arm_scaled_fineweb_design_analysis.json`
