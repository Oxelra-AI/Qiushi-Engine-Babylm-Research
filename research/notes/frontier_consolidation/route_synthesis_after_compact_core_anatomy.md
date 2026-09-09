# compact core full loss anatomy route synthesis after compact-core full-surface anatomy

AoA localization, full compact-reinvest evaluation and the independent-seed comparison remain unresolved. No new training, generation or GPU evaluation was launched for this density-route analysis.

## What changed

The compact-view-core complete result should be read as a structured task-profile change, not as a single AoA story and not as proof that semantic compression is bad.

From existing full official-compatible outputs, compact_view_core relative to the inherited clean-Qwen model:

- gains BLiMP +0.31, EWoK +1.07, Entity +2.09, COMPS +0.40, Reading +0.49;
- loses Supplement -0.99, GlobalPIQA -1.485, SuperGLUE -1.407;
- receives counted AoA -12.687 because a small negative raw AoA correlation crosses the official p<=0.1 scoring boundary.

The compact core full loss anatomy subtask anatomy localizes the non-AoA losses:

- Supplement: the loss is concentrated in `qa_congruence_easy` (-7.81) and `qa_congruence_tricky` (-3.64), while `turn_taking`, `subject_aux_inversion`, and `hypernym` improve. The weakest QA slices are answer-type/action-role contrasts such as animate vs. inanimate, object answer vs. person answer, locative answer vs. NP answer.
- GlobalPIQA: compact_view_core is only 1 net parallel example and 2 net nonparallel examples below clean-Qwen. Both endpoints are weak relative to the target surface, so small practical swaps alone cannot carry SOTA unless other columns remain intact.
- SuperGLUE: the mean loss is mainly RTE (-5.755) and MRPC (-4.412), with MNLI/QQP/MultiRC flat-to-positive and WSC unchanged. This points to a sentence-pair entailment/paraphrase calibration issue rather than broad finetuning damage.

Evidence files:

- `research/notes/frontier_consolidation/compact_core_full_loss_anatomy.md`
- `experiments/archive/frontier_consolidation/data/compact_core_full_loss_anatomy/compact_core_full_loss_anatomy.json`

## Better current hypothesis

The failed FineWeb compact_core overlay simultaneously changed two things:

1. It inserted compact same-source FineWeb view packets.
2. It removed 423,520 words of the inherited clean-Qwen official distribution: CHILDES 137,760; Gutenberg 99,520; OpenSubtitles 97,760; SimpleWiki 56,640; BNC Spoken 30,720; Switchboard 1,120.

Given the loss pattern, especially QA answer-type/action-role and RTE/MRPC changes, a plausible explanation is that source-distribution displacement is entangled with the compact-view mechanism. The compact-view gains on EWoK/Entity/COMPS/Reading remain real, but the current FineWeb overlay may have bought those gains by removing official dialogue/event/paraphrase evidence needed for other columns.

## Conditional next route: Qwen-internal density

If the pending matched evidence continues to support density while implicating source displacement, the next experiment should not be another same-substrate FineWeb overlay. It should move the information-density operation inside the already-beneficial inherited Qwen-pair block.

Core idea:

- keep the official clean-Qwen filler distribution protected;
- keep original sides of the inherited Qwen pairs protected;
- replace only selected near-length generated sides with shorter faithful second views or mixed-resolution views;
- spend the released words on additional official/practical/event-rich packets or restored official rows;
- preserve row geometry and same-window pair visibility.

Why this is scientifically cleaner:

- It tests whether second-view redundancy can be reduced without paying the cost of removing a broad official slice.
- It directly attacks the original inefficiency: 807,787 generated words in the inherited Qwen-pair block, weighted rewrite/source ratio 0.951, many near-length/high-overlap pairs.
- It preserves the source mixture that produced the strongest local complete result.

CPU budget arithmetic from compact core full loss anatomy:

- all inherited Qwen rewrites compacted to target rewrite/source 0.60 would save 283,208 words, enough for about 12,540 additional mean-length originals;
- target 0.65 would save 239,212 words;
- compacting only the high near-length subset (`len_ratio>=0.90`) to 0.60 would save 233,903 words;
- compacting only high-overlap plus near-length pairs (`content_overlap>=0.80 && len_ratio>=0.90`) to 0.60 would save 96,190 words.

Evidence files:

- `research/notes/frontier_consolidation/qwen_pair_density_redesign_budget.md`
- `experiments/archive/frontier_consolidation/data/qwen_pair_density_redesign_budget/qwen_pair_density_redesign_budget.json`

## How to use pending results

The next scientific decision should come from existing managed work, not from speculation:

- AoA localization over near_repeat, near_view, compact_repeat_core, and compact_view_reinvest ladders.
- compact_view_reinvest full evaluation.
- compact_view_reinvest seed43122 fast result.

If reinvest preserves full Supplement/SuperGLUE and AoA while keeping fast EWoK/Entity gains, then the current density family may still be alive, and a narrow next experiment can focus on GlobalPIQA/practical coverage.

If reinvest also loses the same complete-surface columns or AoA localization suggests compression itself drives unstable developmental trajectories across controls, then same-substrate compact FineWeb work should stop.

If compact effects remain useful but the losses track official-source displacement, the next low-cost route is to construct a small Qwen-internal density pilot: select high-redundancy inherited Qwen pairs, generate faithful shorter second views with strict entity/number/modality/role preservation, materialize a 10M pool that protects the official filler distribution, and use a minimal no-AoA screen before any 100M run.

## Work that should not happen yet

- Do not launch another 100M FineWeb compact variant before the pending complete results arrive.
- Do not optimize pretraining toward official AoA words or age labels.
- Do not treat a practical-packet swap as sufficient unless Supplement, SuperGLUE, and AoA survival are measured.
- The reinvest full evaluation is already running; its results are pending.
