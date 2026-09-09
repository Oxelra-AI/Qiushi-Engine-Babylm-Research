# role equivariance world pair route — Role-equivariant paired-world source route

## Scientific Motivation

earlier analysis–249 established that same-world source/bridge paraphrase substrates cannot supply the missing signal for latent occurrence-role assignment: retention collapsed (11/42 families), and the dominant failure was teacher disagreement on same-world counterfactual negatives. The scientific strategist reframed the missing object: the learner needs **independently encoded paired worlds** where an asymmetric relation is genuinely reassigned — R(A,B) true in context C1 and false in C2, R(B,A) true in C2 and false in C1 — with untouched facts persisting and labels grounded in explicit source evidence, using approved teachers only to verify language realization.

This analysis tested, at CPU cost only, whether such source-attested paired worlds exist at scale. No BabyLM training, evaluation, upload, or submission was performed.

## Evidence produced

Scripts:
- `scripts/world_pair_source_feasibility.py`
- `scripts/structured_source_inventory.py`
- `scripts/event_pair_source_inventory.py`

Data:
- `data/world_pair_source_feasibility/world_pair_feasibility_summary.json`
- `data/world_pair_source_feasibility/laliga_reversed_win_seed_pairs.jsonl` (149 seed pairs)
- `data/world_pair_source_feasibility/laliga_reversed_win_world_pairs.jsonl` (356 capped instances)
- `data/world_pair_source_feasibility/structured_source_inventory.json`
- `data/world_pair_source_feasibility/event_pair_source_inventory.json`

independent_review: `data/external/independent_review01_generator1_integration.md`, `.../independent_review01_verifier1_integration.md`.

## Source comparison (reversed-outcome unordered pairs)

Deterministic winner/loser reversals between the same participants across independent event records:

- Jeff Sackmann ATP/WTA tennis archive: **35,715** reversed unordered player pairs (named players, dates, tournaments, surfaces). CC BY-NC-SA 4.0 (noncommercial; research probe use fine).
- BWF badminton (105,147 matches): **6,162** reversed unordered pairs, 17,208 entities, 5 disciplines. Includes many high-multiplicity rivalries (e.g. TAI Tzu Ying vs Ratchanok INTANON, 14/18).
- LaLiga (Zenodo `10.5281/zenodo.18861500`, CC-BY-4.0): **149** clean four-cell flip pairs, 29 teams. Small but fully open.
- Premier League single season (football-data.co.uk `E0.csv`): 43 reversed pairs; the football-data source spans 32 seasons and 22+ divisions, so full extraction scales to thousands.

Static knowledge-graph sources (research-S/M/L, FB15k-rr, wikidata-authors) do **not** provide same-pair role-reversed positives: their reciprocal pairs are dominated by symmetric relations (member-of, diplomatic-relations). They are useful only for relation vocabulary and hard negatives, not paired-world training substrate.

## Scientific interpretation

The paired-world object is real and **broad and source-attested at scale** — tens of thousands of deterministic four-cell role flips exist without any teacher-generated facts. This directly overcomes the sparsity and label-instability that closed the earlier analysis–249 same-world route.

But the strategist's and independent_review verifier's key caution stands: competitive outcomes are one relation family, and the surface score fields give a numeric shortcut ("final score X n, Y m") that a small model could solve by integer comparison rather than by binding the `defeated(A,B)` role to participants. So these sources are not yet a proven BabyLM learning substrate; they are a strong, broad candidate that must first survive a shortcut-resistance and language-realization check, and must be enriched with (a) non-trivial invariants true in both worlds and (b) multiple relation families beyond "defeated".

## Route decision and next work

Live route: build the role-equivariant paired-world substrate from source-attested event records, then test whether it induces latent occurrence-role assignment and transfers to natural EWoK/Entity conditional-reversal behavior.

Next work, in ascending cost, all before any BabyLM-scale training:
1. CPU source-construction pilot: verbalize a mixed set (tennis + BWF + LaLiga) into paired contexts with score text present vs score-ablated (role must be read from "defeated/lost to", not integers), add ≥1 symmetric invariant true in both worlds and ≥1 context-specific anchor, and produce deterministic four-cell labels.
2. Small approved-teacher language-realization check on a few hundred realized contexts: confirm cross-teacher agreement on phrasing (target ≥0.95) and confirm a bag-of-content-words classifier cannot separate C1 from C2 (shortcut test) once scores are ablated.
3. Only if 1–2 pass: broaden relation families (add non-sports asymmetric events, and consider a constrained-generation recipe seeded from KG asymmetric triples with teachers verifying realization) toward ≥5 relation algebras and ≥1,000 stable families with EWoK-domain coverage.
4. Only then: the earliest model test should be a small TinyMLM/DeBERTa role-equivariance probe with predeclared write/read permutation, actor-swap, query-swap, paraphrase, and untouched-entity controls, judged by the same EEBF affected/unaffected signature used at compositional update test design/245 and tied to the rawtoken bridge screen and route judgment EWoK bridge panel.

Do not: return to same-world paraphrase generation; train from the sparse v2 role substrate result and route 11-family core; train on LaLiga-only or score-visible sports data (shortcut risk); reopen compact-view or TinyMLM memory-interface variants. Retain the v2 role substrate result and route families and LaLiga/BWF/tennis pairs as probes/evaluation sets regardless.

Practical assets remain frozen and separate: coherent86 alpha0.75 Overall 42.1210247099666; protected chck_82M 41.942481167385985.
