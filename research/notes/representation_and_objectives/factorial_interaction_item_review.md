# factorial interaction item review — critical review of the HS/LS/HD/LD RoBERTa factorial terminal interaction

## Object under review

earlier analysis trained the four-arm contextual-anchor factorial (HS/LS/HD/LD) as an independent RoBERTa
masked-LM coordinate and produced the **first positive** cross-architecture selected-column interaction
`(HS-LS)-(HD-LD)` at `chck_100M`: cheap6_no_GlobalPIQA +0.9717, cheap5 +0.878, cheap7 +0.4736.
The strategist required that this be tested as broad, monotone, and item-distributed across STABLE
relational families before treating it as a transferable-principle candidate, and that the −2.5
GlobalPIQA move be held in view rather than hidden behind the GlobalPIQA-excluded aggregate.

Sources read (not summaries):
- `data/factorial_selected_eval_terminal/factorial_selected_interaction_summary.json`
- `data/factorial_item_interaction_terminal/factorial_item_interaction_summary.json`
- `notes/extractive_result_review_and_route.md`.

## Decisive finding: the aggregate is a composition artifact, not a relational-family signal

Item-level interaction `(HS_correct−LS_correct)−(HD_correct−LD_correct)` by column (mean pp):

| column | n | interaction pp | reading |
|---|---:|---:|---|
| BLiMP | 59,875 | **+2.29** | island/agreement block-flips (volatile) |
| COMPS | 91,028 | +1.56 | positive aggregate but largest subtask negative |
| Entity | 6,780 | +0.15 | ~zero |
| EWoK | 7,618 | **−1.14** | negative (predicted-positive family) |
| Supplement | 5,218 | **−1.23** | negative |

Item-weighted composites:
- **EWoK+Entity (the predicted relation/state mechanism target): −0.0053 pp** (n=14,398, sum −77) — flat/slightly negative.
- **EWoK+Entity+Supplement+COMPS (all stable relational families): +0.0116 pp** (n=110,644) — essentially zero.

The cheap6 +0.97 arises because the selected composite averages subtask columns with equal weight,
so a positive BLiMP-island movement plus COMPS aggregation with internal reversal inflate the mean.
When weighted by items, and specifically on the families where the compact-view advantage was supposed
to live, **the interaction vanishes**.

### The positive is BLiMP-island driven and symmetric with equally large negatives
Best-interaction subtasks are exactly the known block-flipping BLiMP islands:
`wh_island` +38.96, `left_branch_island_simple_question` +35.23, `only_npi_licensor_present` +31.86,
`coordinate_structure_constraint_complex_left_branch` +22.74, `irregular_past_participle_adjectives` +19.46.
Worst-interaction subtasks are equally large BLiMP items with the opposite sign:
`existential_there_quantifiers_2` −29.97, `wh_questions_subject_gap_long_distance` −20.54,
`irregular_past_participle_verbs` −18.79. This is the signature of block-correlated BLiMP volatility,
not a distributed semantic effect.

### COMPS positive reverses in its bulk
COMPS aggregate +1.56 pp, but its largest single subtask `wugs_dist_before` (n=13,896) is **−2.96 pp**.
The positive COMPS mean is carried by many smaller subtasks, not by the dominant one.

### GlobalPIQA held in view
GlobalPIQA_item_pool interaction −2.46 pp; nonparallel −6.0; parallel +0.97. The GlobalPIQA-excluded
aggregate is what produced the positive cheap6, so the composite positivity is conditional on removing
the family that moved most negatively.

## Convergence with independent DeBERTa extractive result

(`notes/232`) found the same failure mode from the other coordinate: extractive-vs-compact cheap7
positivity was carried by GlobalPIQA/BLiMP while **EWoK+Entity item intervals were negative and excluded
zero at 100M** (balanced item [−1.852, −0.395]; wide item [−1.747, −0.221]). Two architectures
(RoBERTa factorial, DeBERTa extractive), two coordinates, same verdict: the compact-over-repeat advantage
does **not** appear as a stable relational-family gain on the official-compatible surface.

## Scientific conclusion (this is the real answer, not a tuning failure)

The predeclared consequential test — a downstream masked-LM interaction `(HS-LS)-(HD-LD)` that is broad,
monotone, and distributed across stable relational families — is **negative**. The contextual-anchor
interaction is narrow (BLiMP islands), non-monotone (COMPS internal reversal), GlobalPIQA-negative, and
**flat-to-negative on EWoK+Entity**, the exact families where the compact-view mechanism was hypothesized.

Therefore, per the strategist's standing rule, this near-zero relational-family interaction is itself a
real scientific result about the mechanism: **contextual diversification around repeatedly supervised
source-shared anchors does not carry the compact-view advantage as an architecture-general downstream
principle.** The strong DeBERTa masked-denoising compact-over-repeat effect (+2.4164 equal7, earlier analysis)
remains a real, reproducible finding in that coordinate, and the cross-realization representation
(cross realization probe result) is real, but the transferable-principle bar is not met by this factorial.

## Route decision

- **Do NOT** extend to chck_60M/80M for this factorial: the terminal item structure already shows the
  aggregate is not the mechanism, so trajectory points would only describe non-monotonicity of a
  BLiMP-island artifact and cannot upgrade a flat relational-family interaction. Spending GPU on that
  would violate the expensive-work admission rule (it cannot change the continue/stop/redirect decision).
- **Do NOT** launch seed replication or new 100M arms on this construction.
- The compact-view mechanism line, as "downstream contextual-anchor interaction," is closed as a
  transferable principle. What survives: (1) the DeBERTa masked-denoising compact-over-repeat endpoint
  effect; (2) the cross-realization representation that tracks compact rewrite INPUT distribution in all
  arms including repeat-trained; (3) the local source-absent denoising channel dissociated from downstream
  competence.
- The scientifically honest next object, if the compact route continues, must explain WHY the strong
  DeBERTa masked-denoising effect does not transfer to RoBERTa selected downstream competence — i.e., a
  coordinate/architecture-dependence question — rather than another selection/derangement variant that
  local NLL alone would decide. This calls for rebuilding the proposed mechanism,
  coordinated with companion analysis's bridge dose-response (messages 288–289), which is testing the same
  transfer-failure question from the source-only continuity side.

## Practical endpoints (unchanged, frozen)
coherent86 α0.75 Overall 42.1210247099666; chck_82M 41.942481167385985; compact_view_reinvest 42.0867857191.
Leaderboard submission is outside this analysis.
