# address bootstrap and role query — Address-route bootstrapping and the remaining role-query problem

## Purpose

The strategist note corrected the format sensitivity boundary interpretation: exact-chance prefix and inline arms are not scientific measurements unless optimization access is resolved. address bootstrap and role query therefore used the fitted `Entry TAG:` address reader as a lever. The central question was whether direct address learning can make new context renderings usable, and whether that then allows role words in the context (`Focal background entry TAG`, `Focal update entry TAG`) to support queries that do not name the tag.

The probe used only small pretrained bridge runs on the existing ATP temporal-change construction; no Strict-Small 100M training or official evaluation was started.

## Scripts and construction

Created `training/scripts/prefix_warmstart_probe.py`.

- Builds the canonical tag-only train set and a matched transformed-context train set with the same 80 base stable worlds, `base_train_wording=true`, 16 sparse changed focal worlds, 16 sparse stable focal worlds, and held/train changed evaluations.
- Adds `--shared_address_ns` so tag-only and transformed-context rows can use the same arbitrary record tags for each world/vi/vl, changing only the context rendering.
- Seeds model construction before training after finding that the entailment head is randomly initialized; early unseeded address bootstrap and role query results are useful only as exploration.

Created `training/scripts/role_query_bootstrap_probe.py`.

- Runs a sequential chain: tag-only direct address learning -> inline-role context with direct-tag queries -> inline-role context with role-phrase queries.
- Compares against scratch role-query training.
- Evaluates direct tag queries, role queries, held role paraphrases, and role-swap contexts.

Dry builds passed: 5,632 rows for each train set; balanced labels; max token lengths under 200; no truncation.

## Direct address surfaces

The original format sensitivity boundary exact-chance reading is not stable. Re-running or rebuilding related controls showed three regimes.

### Minimal prefix is easy enough under the repaired path

`The entry TAG:` is not a true interface incompatibility.

- `data/badns_prefix1_warmstart_seed27000/`: using the exact format sensitivity boundary prefix namespace, tag-only pretraining reached best train 1.0; before continuation the tag model already read the prefixed context perfectly on held changed/stable direct queries; after continuation the prefix train set reached best 0.9996 with sparse changed focal 0.9922 and held direct readouts all 1.0.
- `data/badns_prefix1_scratch_seed27001/`: same namespace and scratch direct learning also reached best 0.9998 with sparse changed focal 0.9961.

Thus format sensitivity boundary `prefix1_direct` at exact chance was a bad optimization run, not a general data-efficient learning law.

### Inline role context is path-sensitive for changed focal rows

When the context is `Focal background entry TAG:` / `Focal update entry TAG:` and the query still names the tag, stable rows are easy but changed focal rows can fall into a shallow basin.

- `data/seeded_badns_inline_scratch_seed26900/`: scratch direct learning with the format sensitivity boundary inline namespace fit base and stable rows, but sparse changed focal stayed near chance (best overall 0.9776; sparse changed 0.5156; held focal_after 0.5 while secondary stayed 1.0).
- `data/badns_inline_scratch_seed27002/`: same namespace, another seeded scratch run also fit base/stable but sparse changed focal remained 0.5273.
- `data/badns_inline_scratch_seed27003/`: another run did not reliably fit even base/stable (best 0.8601), again with sparse changed focal near chance.
- A different namespace/control (`data/inline_scratch_control_seed27300/`) could fit inline direct from scratch, so the obstruction is not the English words themselves. The hard part is the interaction among arbitrary address tokens, initialized head, sparse contradictory changed records, and the transformed surface.

### A learned address route rescues the hard inline direct surface

The same difficult inline namespace becomes learnable after direct tag-address learning.

- `data/seeded_badns_inline_warm_seed26900/`: tag-only direct pretraining reached best 0.9996 with sparse changed 0.9922; continuation on inline direct reached best 0.9998 with sparse changed 0.9961 and held inline direct readouts all 1.0.
- `data/badns_inline_warmstart_seed27000/` and `data/shared_inline_warmstart_seed27000/` show the same rescue pattern under earlier exploratory settings.

The supported interpretation is not a broad first-token rule. It is a path-dependent bootstrapping effect: once an address coordinate route exists, the model can adapt it to a nearby surface carrying the same record tags and the same labels. Without that route, the sparse changed focal records can be inaccessible for some address-token/format basins even while base and stable records are learned.

## Role-query composition remains unsolved

The route-relevant question is not whether direct tag lookup survives role words in the context. It does. The harder operation is whether a query such as “the focal update record” can select the entry whose context contains `Focal update entry TAG:` without naming `TAG`.

Canonical seeded role-query experiment:

- Warm chain: `data/seeded_role_query_bootstrap_warm_seed26900/`.
  - Tag-only direct phase: best 0.9996; sparse changed focal 0.9922.
  - Inline direct phase: best 1.0; sparse changed focal 1.0.
  - Role-query phase: base and stable rows fit, but sparse changed focal stayed at 0.5313. Held role-query readouts did not give reliable focal before/after selection; secondary rows remained 1.0. Direct/tag retrieval was also degraded after role-query training: tag sparse changed fit fell to 0.5234, and held direct changed/stable readout fell to 0.312 in the direct-retention block.
- Scratch role-query comparator: `data/seeded_role_query_bootstrap_scratch_seed26911/`.
  - Scratch role-query training stayed at exact chance across base, stable, and sparse changed rows.

Earlier unseeded exploratory runs sometimes found better role-query fits, but they were confounded by unseeded model-head initialization. The reproducible seeded comparison says: an address route can bootstrap direct retrieval in role-bearing contexts, but a sparse role-query objective can still fail on contradictory changed records and can disturb the already learned direct address route.

## Updated scientific reading

The address bootstrap and role query evidence changes the current principle fragment.

1. Supplied address coordinates are not merely brittle lexical tricks: after formation, they transfer across nearby context renderings and can rescue a hard direct-surface acquisition path.
2. The difficult part is not adding role words to the record sentence. Direct tag lookup works through them once the address route is formed.
3. The unresolved operation is semantic selector composition: mapping a role phrase in the query to one of multiple role-labeled address records, especially when focal before and focal after contradict each other. Sparse role-query training can fit stable records while failing changed focal rows, and can overwrite or destabilize direct retrieval.
4. The broader data-efficient learning principle is therefore closer to **coordinate bootstrapping under sparse interference** than to generic format sensitivity: small learners can reuse a formed coordinate route to make nearby experiences learnable, but installing a new selector over that route needs either cleaner supervision, a separated head/objective, or a representation that protects record retrieval while the selector is learned.

This remains a bridge result, not a complete general theory. It does not establish temporal semantics, natural event-derived updating, architecture-general transfer, or a Strict-Small gain. It does identify a stronger next object than format sensitivity boundary: separately measure and train `M(role -> address)` and `R(address -> state)`, rather than forcing role-query state labels to learn both while also preserving the direct address route.

## Next construction that would most strengthen the route

Build a low-cost map-and-retrieve decomposition:

- Keep the fitted direct address route `R(k, v)` as a reusable object.
- Train or probe a separate role-to-address map `M(r, k)` using paired positives and negatives such as “The focal update entry is TAG_X” versus other tags in the same context. This teaches record selection without asking the same binary state head to relearn before/after facts.
- Compose the two components in evaluation: select the role-addressed tag, then use the direct tag route for the state query. Test held worlds, held tag names, role swaps, and held role paraphrases.
- Add direct-tag retention rows or freeze/adapter separation only as mechanisms to preserve the address route; do not treat replayed state labels as evidence for untouched-state conservation.

If `M` is learned and composition works while direct `R` remains stable, the research has a concrete bootstrapping principle for how sparse data can install reusable selectors over previously formed coordinates. If `M` fails independently, the bottleneck is role-to-address mapping itself rather than changed-state readout. If `M` works but composition fails, the interference is in combining selector and state reader rather than in either component alone.

## Relation to the Broader Experimental Evidence

frontier_consolidation’s MAX view/repeat/breadth/permuted-companion decomposition is directly relevant. address bootstrap and role query suggests that the structure of repeated or companion experience matters through whether it preserves an addressable correspondence, not merely through non-duplication. Their permuted companion arm is well aligned with this question: aligned view versus permuted companion separates source-conditioned correspondence from compact-text distribution and generic independent-token replacement. The most useful peer readout will compare Entity and binding-sensitive rows under V-R, V-P, and P-R rather than reducing the result to a single mean score.
