# natural cluster preflight review next mechanism candidates after branch closures

## What this step has removed from expensive training

Completed nine-column and no-AoA evidence now removes three tempting but weak continuations from the next 100M wave:

1. Simple first-pass developmental order from corpus source/statistics: both trained seeds are below clean-Qwen, and both retain AoA 0.0.
2. Cap-length / row-geometry tuning: the best official160 point gains only +0.0093 no-AoA equal7 over clean-Qwen and trades away Supplement, Entity, and Reading; cap120 geometry is below clean-Qwen.
3. Representation pair-agreement as a stand-alone objective: the no-training representation margin on clean qwen compliance and validity pairs is nearly identical between clean and official models and is lower for clean than the shuffled-trained control, so it mostly measures text relatedness/exposure rather than an unspent learning signal.

Tail checkpoint averaging remains under no-AoA evaluation as a single-model consolidation test. It should only receive full nine-column measurement if one averaged model clearly exceeds clean-Qwen equal7 while not damaging Supplement/Entity/Reading.

## What a new paid training wave must explain

The clean-Qwen model has the best admissible complete coordinate, Overall 41.3443, but it remains below the visible 41.8 leader. A new training wave should not merely increase one already-strong column. It should supply a causal route for raising the weaker region

- W = EWoK, COMPS, GlobalPIQA,

while retaining the clean model's important strengths

- R = Supplement, Entity, SuperGLUE, Reading,

and keeping BLiMP from collapsing. The test must use matched initialization, tokenizer, word exposure, row-length/topology where possible, batch/LR-time, checkpoint policy, and no-AoA checkpoint selection.

## Candidate family 1: natural entity-relation evidence clusters

Mechanism: find small clusters of official-training sentences from the same document or local source neighborhood that share a high-confidence anchor string (proper name, noun phrase, number, or explicit entity mention) but provide complementary facts or relations. Pack 2-3 complementary official sentences in the same 256-token window at a low dose, about 2-4% of the 10M pool, without using Qwen unless needed for a separate comparison.

Why it fits the observed gap: EWoK, COMPS, and GlobalPIQA need relation and attribute stability; this mechanism gives the model real text-internal evidence for entity/property/event binding without replacing large parts of the official corpus. It differs from failed pair-agreement because the useful object is not a seen original-rewrite pair but naturally occurring complementary evidence in the corpus.

Minimal test before any 100M wave:

- Build a metadata-only extractor and sample 1k clusters.
- Manually/automatically measure anchor precision, sentence completeness, duplicate rate, source concentration, word budget, and whether each cluster has non-identical relation/property words.
- Compare four small corpora for a 10-20M exposure screen: true cluster, anchor-only cross-document shuffle, original-repeat, and same sentence inventory left in original position.
- The true cluster must improve W or a fixed training-internal held-out sentence loss more than the shuffle/repeat controls while retaining R-related no-AoA columns.

## Candidate family 2: spaced second-view reactivation

Mechanism: keep selected original/Qwen rewrite texts but separate the two views by a controlled token lag instead of same-window adjacency. The lag is defined in trainer-visible token stream, with low rewrite dose around 3-5%. This tests whether delayed reactivation consolidates reusable relations better than immediate co-presence.

Why it fits the observed gap: same-window correspondence helped but produced tradeoffs; spacing could force a persistent parameter update rather than transient attention. It is still distinct from failed whole-pass developmental ordering because the object is local recurrence timing for matched content, not global corpus order.

Minimal test before any 100M wave:

- Build a no-training materialization preflight with exact text inventory and lag distribution.
- Compare true spaced rewrite, shuffled spaced rewrite in source/length/entity buckets, spaced original-repeat, immediate true pair, and random-lag true rewrite.
- Run only 10-20M exposure first, then screen no-AoA. Continue only if a nonzero structured lag improves W without giving up the clean model's Supplement/Entity/Reading pattern.

## Candidate family 3: evidence-visible MLM masking inside existing pairs

Mechanism: keep the clean-Qwen text inventory and low pair dose fixed, but alter which tokens are masked inside pair windows. When one side masks a relation-bearing token, entity, number, or predicate-neighborhood token, the other side keeps corresponding evidence visible. Mask count and token-class exposure must remain matched to independent WWM and shuffled-position controls. This remains MLM, not an additional agreement loss.

Why it fits the observed gap: standard WWM may not put prediction pressure on relation-bearing spans when cross-view evidence is visible. Unlike representation pair-agreement, this asks whether gradient falls on the useful relation-bearing positions.

Minimal test before any 100M wave:

- Before training, compute batch-level gradient-alignment on clean-Qwen early checkpoints for evidence-visible masking versus shuffled-position masking, independent WWM, original-repeat masking, and bilateral masking.
- Use only training text and Qwen-pair metadata; no downstream task labels or AoA information.
- Continue to a 10-20M paired-seed screen only if evidence-visible masking aligns better with same-document held-out sentences than all controls after normalizing gradient norm.

## Recommended next owner

Families 1 and 3 are the proposed candidates, with implementation risk and expected information value as selection criteria: they address the clean model's W deficit more directly than another Qwen exposure or cap-length change. The corresponding metadata extractor or gradient test would support a matched comparison distinguishing the proposed mechanism from anchor frequency, repetition, shuffled correspondence, or ordinary WWM.
