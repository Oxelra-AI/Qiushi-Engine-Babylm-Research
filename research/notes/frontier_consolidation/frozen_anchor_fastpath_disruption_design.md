# frozen anchor fastpath disruption design frozen-anchor fast-path disruption design

## Why the lead frozen anchor fastpath route first pair needed repair

lead frozen anchor fastpath route's proposed A2/A3 first pair remains the right route at the mechanism level: keep the verified `chck_82M` function frozen and let late information enter only through a reversible private path, then judge the private-ON model by item retention and acquisition. But the implementation details needed correction before using both H100s.

1. `frozen82_private_tail_trainer.py --mode neutral_only` does not train on main MLM. It returns no auxiliary views and then optimizes only deterministic private-ON vs private-OFF KL. At zero-output private initialization this is nearly zero, so it is not coherent broad replay acquisition. A new replay trainer is needed.
2. Raw token-order scrambling is an excessively artificial control. It can make the frozen slow model process non-language, so a coherent advantage could reduce to local fluency or input naturalness rather than cross-span dependency learning.

## Repaired first contrast

New script: `experiments/archive/frontier_consolidation/scripts/frozen82_fastpath_replay_trainer.py`.

Shared factors for A2/A3:
- endpoint: verified scale1.75 `chck_82M` slow function;
- all non-private tensors frozen;
- fresh zero-output private adapter after each layer, bottleneck 128, scale 1.0;
- private-only AdamW, lr 0.001, 455-step 100M-style tail horizon, warmup 0.06;
- same legal suffix after `skip_rows=530944` / `initial_consumed_words=82,012,495`;
- row-level cap `max_tail_charged_words=3,992,918`, matching the truthful shuffled86 charged suffix as closely as possible. With main-only rows this consumes 3,992,800 words (118 below shuffled86 because the next 160-word row would exceed the cap);
- same WWM mask probability and RNG seed;
- objective: private-ON main MLM CE + deterministic private-ON/private-OFF KL on the same masked batch.

A2 `coherent_replay`:
- normal legal suffix text and WWM targets.

A3 `spanbreak_replay`:
- preserves each row's token multiset, word-group multiset, attention length, row/source order, charged word counts, WWM target-word-group IDs, and therefore the target/mask token multiset within each row;
- builds punctuation/length-delimited spans, keeps token order inside each span, then permutes spans within the row;
- breaks cross-span/cross-sentence discourse dependencies while preserving short-range plausible fragments much better than raw token scrambling.

Raw scrambling is deliberately not used in the main first pair. It can remain a lower-bound stress input only if this repaired contrast is ambiguous.

## Success reading fixed before scores

The first pair can only justify continuing if the private-ON function retains most decisions that the anchor got right in the known declining families while adding reproducible new decisions elsewhere. Private-OFF equality is only a safety property; merely beating an artificial disrupted input is not a scientific success.

For A2/A3 versus anchor and truthful shuffled86:
- compare cheap7 and SuperGLUE only after both endpoints train successfully;
- compute paired official item transitions versus `chck_82M`, especially EWoK physical/material/interaction/spatial groups and high-operation Entity groups;
- report BLiMP/COMPS gains separately so they cannot hide relation/state losses;
- if coherent A2 and spanbreak A3 are similar to each other and to shuffled86 in family turnover, stop as private-tail redistribution;
- if A2 clearly preserves anchor-correct fragile decisions better than A3 while adding new correct decisions, then consider Arm 4 retention KL with fixed coefficient and repeated SuperGLUE seeds.

## GPU-work admission

This 4M pair is the minimum reliable mature-state test of the frozen-anchor route. It decides whether the route continues to retention machinery or stops before additional tail variants. It uses the two currently free H100s in parallel and spends only about 22% of the remaining legal suffix budget in each branch, starting from an already submitted immutable anchor.
