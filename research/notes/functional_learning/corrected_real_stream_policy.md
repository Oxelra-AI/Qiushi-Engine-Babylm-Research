# corrected real stream policy corrected real-stream policy

## Why wwm translation result decisions were repaired

The wwm translation result `risk0_admission_decisions.jsonl` file encoded `semantic_label=faithful_shortening` whenever the automatic structural screen returned risk=0/no tags. That is not supported. The risk screen detects a limited set of surface and structural warning signs; absence of those warning signs is not equivalent to semantic preservation. The previous calibration and the corrected real stream policy full-text packet both show counterexamples.

Concrete examples:

- `rw2s0_030370`: source/current contain a belief/inference frame (`I think` / `I believe`); generated compact says `something else happening`, converting a subjective inference into a more direct assertion. This row is excluded.
- `rw2s0_001402`: generated compact returns exactly to the original source sentence. That may repair the inherited rewrite because `being a nurse like she was` preserves the speaker's shared-profession relation, but it is not a genuine shorter second expression. Exact source returns are separated from compaction and excluded from the default policy.
- earlier analysis known altered IDs `rw2s0_029078` and `rw_035551` are also excluded.

`scripts/structural_policy_materializer.py` therefore rebuilds provenance from the full selective-generation output and semantic screen. It creates a policy file in which `admit_compact=True` means only:

> structurally filtered, non-exact generated shortening candidate from selective generation; no independent semantic certification.

No admitted row is labeled `faithful_shortening`.

## Policy counts

Output summary: `data/structural_policy_materialization/structural_policy_summary.json`.

- Total selective-generation rows: 26,567.
- Not structural candidates: 15,013.
- KEEP_CURRENT rows: 1,957.
- Exact source returns / repair-or-recurrence candidates: 1,313, saving 2,963 unique-pair words if used, but excluded from the default compaction policy.
- Demonstrated exclusions: 3.
- Admitted non-exact structural shortening candidates: 8,281.
- Unique-pair saved words for admitted non-exact candidates: 33,398, mean 4.03 words.
- Admitted sources: simple_wiki 3,431; gutenberg 2,332; open_subtitles 1,631; bnc_spoken 507; childes 376; switchboard 4.

## Full-tail materialization

Built tail: `data/structural_policy_materialization/compact_structural_reinvest/compact_structural_reinvest_reference_tail_wordpaced_segments.jsonl`.

Static check: `data/structural_policy_materialization/static_tail_check.json`.

The built tail:

- exactly matches all 17,107 `qwen_pair_packed` reference-tail rows to packed-pair metadata;
- compacts 8,941 qwen rows and 11,641 pair occurrences from 8,281 unique admitted pairs;
- saves 46,914 tail words, then reinvests 46,880 words into 293 reviewed/top-up rows, leaving 34 words unspent;
- remains legal relative to the coherent86 parent: 99,999,966 total words if continued from 86,005,295 consumed words;
- carries `qwen_pair_segments` for every qwen row so a trainer can identify source and second-view spans;
- passes static provenance checks: no exact-source return admitted, no demonstrated failure admitted, no fabricated faithful label, and all qwen segment spans align with row text.

The initial 80-update prefix contains 3,162,798 words, 20,545 rows, 3,848 qwen rows, 11,827 qwen pair segments, 2,057 compact-modified rows, and 2,681 compact-modified pair occurrences.

## independent_review sample reading

independent_review reviewed the 90-row full-text packet at . The memo says the packet is semantically heterogeneous and not a random prevalence estimate. It identifies clear faithful shortenings, exact source returns/repairs, information-losing partial views, and altered meanings within the risk=0/no-tag region. Important policy implication: any learner result from the 8,281-row policy is the effect of a mixed structural text intervention, not clean evidence for meaning-preserving compaction.

Therefore a positive model result would justify stronger evaluation and subsequent separation of policies; it would not by itself establish the mechanism. A neutral or negative result would reject this tested mixture/schedule/objective, not compaction in general.

## Real-stream learner comparison now running

Script: `scripts/real_stream_train.py`.

The corrected real-stream training comparison was in progress at the time of this note.

It compares two objectives on the same packed 80-update prefix:

1. `inherited_wwm`: standard row-keyed 15% whole-word masking on all rows, including compacted qwen rows.
2. `correspondence_focus`: ordinary WWM on non-qwen rows, while qwen_pair_packed rows use source-visible, view-focused targets on content words in second-view segments. This retains ordinary learning elsewhere while transporting the successful allocation balanced interpretation learning condition into realistic stream order and competing ordinary experience.

Dry-run target preview for the first macro-update:

- inherited WWM: 252 rows, 39,581 words, 33 qwen rows, 8,711 ordinary target tokens.
- correspondence-focus: same 252 rows and words; 219 non-qwen WWM rows plus 33 qwen focus rows; 7,817 ordinary target tokens and 284 focus target tokens. The focus targets include non-admitted, admitted, exact-source-return, keep-current, and not-in-policy segments because every qwen source/view pair has a second view; the compact substitution itself only applies to admitted non-exact rows.

The planned evaluation applies the earlier common-target probe and fast Cheap7 to the 80-update checkpoints. This is a prefix test for mechanism and broad preservation before any full-tail investment; the evaluation was incomplete at the time of this note.

## BabyLM rule check

The BabyLM FAQ in `data/external/FAQs.md` says: "Any training objective/regime is permitted as long as the data restrictions are followed" and external learned linguistic data counts toward the 100M word budget. Therefore standard WWM is inherited practice, not a rule-imposed objective. A source-visible, view-focused objective remains a candidate training principle if word accounting and scoring-function requirements are satisfied.
