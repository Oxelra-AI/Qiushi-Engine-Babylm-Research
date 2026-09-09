# compact_experience relation design prestate pre-state: COMPACT_EXPERIENCE qwen relation-typed design

This note is written before scoring the inherited five-arm Qwen mechanism readouts. It fixes the interpretation frame so the new numbers are read as scientific evidence rather than as a post-hoc extension of leaderboard scores.

## Why this design matters

The inherited COMPACT_EXPERIENCE/REPRESENTATION_FRONTIER_STUDIES `qwen_pair_packed` block is not merely accidental high-overlap adjacency inside CLEAN. The COMPACT_EXPERIENCE materialization contains 37,594 filtered original--Qwen-rewrite pairs, 1,656,800 words per 10M pool, mean content overlap about 0.63, preserved pair boundaries, and about 10,822 selected pairs from `simple_wiki`. clean natural nearduplicate adjacency's refined audit found about 22,117 local high-overlap hits at tau=0.50 per 10M in this block. Therefore the REPRESENTATION_FRONTIER_STUDIES qwen-clean baseline already practices an in-window restatement relation at substantial dose.

COMPACT_EXPERIENCE trained a zero-new-training relation design at seed43022:

- `official_lengthmatched`: no selected qwen-pair practice; qwen block replaced by official filler.
- `selected_original_dup_all`: selected official originals duplicated locally, no Qwen words. This is the exact-recurrence analogue after verifying the materializer appends each selected original twice in a packed row (`cur_segments.extend([p.original, p.original])`) and metadata shows pair boundaries preserved, no truncation, and 12,550 duplicate-pair rows.
- `qwen_clean_aligned`: selected original paired locally with its own Qwen rewrite.
- `qwen_shuffled_control`: selected original paired locally with a wrong Qwen rewrite; original and rewrite multisets are preserved but correspondence is broken in 37,594/37,594 pairs.
- `qwen_separated_pair`: selected originals and their Qwen rewrites both appear, but originals and rewrites are placed in separate rows/windows; it is a valuable locality-reduced point, although its row-count/row-length sequence differs from the aligned arm.

This design can test, on the SOTA ingredient itself, whether correspondence inside the window matters beyond (i) exact recurrence of selected source text, (ii) wrong-rewrite register adjacency, (iii) split coexistence of the same originals and rewrites, and (iv) no qwen-pair practice.

## Readouts and branches

The source-conditioning readouts will be used rather than official-style Overall/Human-like aggregates, because AoA accounting differs across these arms and those aggregates are not mechanism measurements.

Readouts:

1. Compact FineWeb-register T/U/N on the existing held-out compact rewrite probe: `A_T=N-T`, `A_U=N-U`, and `G=U-T`, with positive `A_T` meaning more true-source benefit.
2. Wikipedia/Simple-English T/U/N by target class: source-recurring (`overlap`) versus source-absent (`nonoverlap`) targets. This probe is near-register for the qwen substrate because a large slice of qwen pairs is from `simple_wiki`.
3. Held-out natural-copy gain on natural source-repeat rows.
4. Entity official predictions summarized by relevant queried-state updates, using existing per-target prediction files only as item-level predictions, not as aggregate score evidence.
5. N anchor and ordinary held-out loss should be added later if the separated arm lands in an ambiguous middle, because its row-count mismatch can otherwise mimic or mask a locality effect.

Pre-stated scientific branches:

- If `qwen_clean_aligned` exceeds `qwen_shuffled_control` on compact or Wikipedia true-source benefit while `qwen_shuffled_control` sits near `qwen_separated_pair` or `official_lengthmatched`, the SOTA ingredient supports a stronger claim: source--rewrite correspondence, not mere same-window rewrite-register adjacency, installs the useful routine.
- If `qwen_shuffled_control` also exceeds `official_lengthmatched` substantially, local rewrite-register adjacency has an independent effect, but correspondence is still measured by aligned-minus-shuffled.
- If `qwen_shuffled_control` falls below `official_lengthmatched` on compact or Wikipedia `A_T=N-T` while `A_U=N-U` is flat or higher, wrong local restatement is not inert adjacency: it has practiced non-correspondence/discounting between the current source and adjacent restatement-register span. This would be the restatement-side analogue of exact recurrence's active changed-form cost, separated from general fit damage by the neutral `N` anchor and later ordinary held-out loss. It would upgrade the design from a two-relation contrast (exact recurrence versus own restatement) to a three-relation principle including practiced mismatch.
- If `qwen_separated_pair` is near `qwen_clean_aligned`, the SOTA ingredient's effect may be exposure/coexistence rather than same-window locality; because that arm is not row-length-sequence matched, N and ordinary held-out checks would be needed before training an exact QWEN_SPLIT.
- If `selected_original_dup_all` improves natural-copy gain but harms changed-form T/U/N relative to `official_lengthmatched`, the designed REPEAT/VIEW dissociation reappears on the SOTA ingredient axis. If it does not, exact recurrence behavior is bounded to the compact/FineWeb intervention dose or to its exact construction.
- A large Wikipedia effect with a small compact effect would mean installed restatement use is register-shaped and transfers best near SimpleWiki/Qwen-style restatement; this sharpens the principle rather than refuting relation locality.
- A null across all readouts at the qwen dose would bound the designed compact result to its own register/dose and would argue against using qwen-clean as evidence for the same mechanism.

Output target for the scoring run: `experiments/archive/relation_learning/data/paired_context_relation_design_probe`.
