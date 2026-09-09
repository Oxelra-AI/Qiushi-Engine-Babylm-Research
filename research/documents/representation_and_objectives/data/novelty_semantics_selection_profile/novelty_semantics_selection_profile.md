# target channel semantic structure novelty × semantic target-set profile

Matched selections are written as JSONL target/matched files under this directory. They are not training results.

## Group counts

- copied|capitalized_or_number: groups=19422, pieces=45026, types=8367
- copied|generic_entity_nonrel: groups=2233, pieces=2923, types=38
- copied|ordinary_nonrel_nonentity: groups=69431, pieces=118130, types=15577
- copied|rel_event: groups=5647, pieces=6481, types=156
- source_absent|capitalized_or_number: groups=1365, pieces=3398, types=950
- source_absent|generic_entity_nonrel: groups=234, pieces=289, types=34
- source_absent|ordinary_nonrel_nonentity: groups=16112, pieces=25183, types=5659
- source_absent|rel_event: groups=2886, pieces=3088, types=140

## Pairwise match summaries

- novel_rel_vs_copied_rel: matched 2886/2886 target groups; pieces 3088 vs 3088; support_log diff mean 0.011632; position diff mean 0.017155
- novel_ord_vs_copied_ord: matched 16112/16112 target groups; pieces 25183 vs 25183; support_log diff mean 0.015937; position diff mean 0.003365
- novel_capnum_vs_copied_capnum: matched 1365/1365 target groups; pieces 3398 vs 3398; support_log diff mean 0.087519; position diff mean 0.003193
- novel_rel_vs_novel_ord: matched 2886/2886 target groups; pieces 3088 vs 3088; support_log diff mean 0.022816; position diff mean 0.01432
- copied_rel_vs_copied_ord: matched 5647/5647 target groups; pieces 6481 vs 6481; support_log diff mean 0.001744; position diff mean 0.009098

## Reading

These selections define the next possible separation: source novelty at fixed relational/event semantics and relational/event semantics at fixed source novelty. Use them only after the pending 100M endpoint and companion analysis transfer result show the local compact channel reaches task competence.
