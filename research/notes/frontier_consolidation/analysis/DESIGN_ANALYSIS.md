# Design analysis and stress-test interpretation

## Construction selected

Use broad source-absent rewrite innovations as the treatment pool and keep
relation cues as a logged slice.  Start from the exact baseline 15% Bernoulli
WWM selection.  Each changed row proposes at most one hash-rotated innovation
group.  A baseline collision causes no change.  Otherwise, swap the proposed
group with a selected non-pair group of identical token length elsewhere in the
batch, preferring an ordinary row.  This exactly preserves selected group and
token mass per batch while leaving source and copyable pair masks unchanged.

The first row-local version was rejected after stress testing: 17,519 of 18,112
donors were copyable rewrite groups.  Although mass matched exactly, that would
confound innovation upweighting with copyable suppression.  The final batch-wide
policy protects every source/rewrite group and produced zero copyable or source
selection delta.

## Metadata and edge cases

The builder joined all 3,005 route portfolio and intervention assets ids to the frozen stream and verified ten
identical occurrences per id.  All route portfolio and intervention assets alignments were exact.  It found
12,145 both-visible pairs, seven source-only pairs, and three invisible pairs;
only both-visible pairs are eligible.  All 3,005 rows contain multiple pairs and
52 are truncated at 256 tokens.  There are 1,827 discontiguous source-pair spans
and 1,817 discontiguous rewrite-pair spans among eligible pairs.

Groups are rebuilt exactly as in the baseline trainer.  A target is eligible
only if the complete tokenizer WWM group is contained in the union of one
rewrite's ranges.  This excluded 2,595 rewrite groups touching a boundary or
gap and 2,596 analogous source groups.  The strict result is 26,315 innovation
groups / 36,899 tokens per 10M pool, versus the earlier permissive probe's
26,435 / 37,453.  The 120-group / 554-token reduction is the price of eliminating
partial-piece targets.

The full audit JSONL retains decoded norms for human review.  A separate lean
`innovation_metadata_train.jsonl` removes all normalized strings and carries
only the ids/positions/lengths and protection sets required by the selector.

Classification is pair-local exact normalized membership.  A repeated source
word still makes a rewrite group copyable; 18,361 copyable groups had more than
one matching source occurrence.  A word in another same-row pair is irrelevant,
and a same-row decoy is never used.  Source/rewrite overlap is a hard error;
the complete build found none.  No rewrite WWM group belonged to multiple pair
records.  Copyables, innovations, and relation-cue innovations are stored as
separate audit categories; relation-cue status does not influence selection.

## Corrected broad smoke

The final CPU smoke used all 3,005 unique changed rows, 3,005 reservoir-sampled
ordinary rows, four independent replicates, balanced batch size 64, seed 43022,
and 15% WWM.  This is 24,040 row exposures / 3,559,420 charged whitespace-word
exposures.  It selected 527,935 groups / 764,428 tokens in both baseline and
treatment, with zero aggregate or per-batch group/token delta.

There were 12,016 eligible proposals, 1,822 baseline collisions (15.163%),
10,183 successful swaps, and 11 no-equal-length-donor skips.  Candidate-level
baseline collision was 14.856%.  Innovation selections rose from 15,637 to
25,820; changed-row coverage by at least one innovation rose from 72.57% to
99.94%.  Across four rotations, 8,584 of 26,315 unique candidates were forced at
least once.  Forced relation cues were 1,318 (12.94%), close to their natural
candidate share; forced copyables were zero.

All 10,183 donors came from ordinary rows.  Copyable selections remained 78,046
in both arms; source selections remained 155,690; forced copyables, copyable
donors, protected-source donors, protected-pair donors, and rows with changed
source selection were all zero.  Every forced target retained at least one
visible paired-source group.  Only 4.03% retained an entirely unmasked source,
which is expected under independent baseline WWM across the many source groups;
the intervention does not alter that source mask.

The frozen batch-256 order is easier than the balanced smoke: all 2,529 batches
contain 2–24 changed rows (mean 11.882), so every real batch has at least 232
ordinary rows.  No batch is all-changed.  Thus the stress no-donor rate is a
conservative bound.

## Whole-word leakage interpretation

Every forced group had all pieces selected and labeled.  Of 10,183 forced
groups, 2,809 were multi-piece.  The unchanged per-token 80/10/10 corruption
made 74.68% of forced groups entirely mask-branch; 12.98% had at least one
intentional gold keep-branch piece; 603 multi-piece groups mixed a visible gold
piece with other corruption branches.  This is not an accidental range leak,
but it dilutes the conditional signal.  Retain it in the first screen to avoid a
second corruption-policy variable and log it.  Group-coherent corruption should
only be a later separately controlled variant.

## Approximate full-100M treatment size

There are 30,040 eligible changed-row exposures (3,004 rows with innovation,
repeated ten times).  At a 15% collision probability, one proposal per row has
about 25,534 non-colliding opportunities.  Applying the conservative smoke
no-donor rate projects roughly 25,506 added innovation selections.  Baseline
expected innovation selection is 263,150 × 0.15 = 39,472.5, so expected
innovation exposure rises to about 64,979, a 64.6% group-specific increase.

pairaware sequence accounting measured 9,802,194 trainer-visible fixed-256 groups per 10M epoch, or
98,021,940 over 100M.  Baseline expected selected-group mass is therefore about
14.70M, making the projected reallocation only about 0.174% of global selected
groups.  This projection is for scale; the actual run must log realized counts.
