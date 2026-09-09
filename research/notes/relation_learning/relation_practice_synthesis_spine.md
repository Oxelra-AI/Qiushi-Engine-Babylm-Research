# Relation practice: synthesis of locality and source-use evidence

This historical synthesis separates the measured relation-locality effects from the behavioral and transfer questions that remained unresolved at the time. Suggested figures and follow-up comparisons below are plans, not additional results.

## Central scientific result now supported

The strongest current principle is not a symmetric story about repetition versus paraphrase. It is a relation-by-locality result under a fixed experience budget: the learner practices the relation that is locally available between spans, and that practiced relation changes how a later related source is used. Seeing the same content as separated exposures is not equivalent to seeing the relation in the same context window.

In the compact held-out rewrite probe, exact local recurrence installs a source-conditioned identity/content pull. This helps when the target is unchanged or exactly recoverable, but it weakens use of a related source for nonidentical targets. Across three original DeBERTa seeds, REPEAT versus CLEAN on tokenizer-nonoverlap compact rewrite targets has large negative source-conditioned gain values: `-0.7517`, `-1.0433`, and `-0.8503`. With the neutral ordinary-prefix anchor, the same effect is source-specific rather than an unrelated-neighbor artifact: R−C has three-seed Δ(T−N) `+0.8138±0.1729` and Δ(U−N) `-0.0679±0.0340`. Here positive Δ(T−N) means the true related source helps less than in CLEAN, while U is nearly neutral relative to N.

Local nonidentical restatement has the opposite compact-probe effect in DeBERTa. Across three original seeds, VIEW versus CLEAN has tokenizer-nonoverlap gain `+0.6837`, `+0.8938`, and `+0.8392`; with the neutral anchor V−C has Δ(T−N) `-0.7970±0.0758` and Δ(U−N) `+0.0085±0.0536`. Thus local restatement strengthens true-source-conditioned support for compact nonoverlap targets, while unrelated compact text behaves close to ordinary neutral text.

The causal locality result is now the spine. Row-split controls preserve selected source material, companion material, suffix/filler content, row-length sequence, and 100M-word budget, while removing within-window source/companion co-occurrence. In two DeBERTa seeds, splitting collapses the large compact nonidentical-use residuals:

| seed | R−C nonoverlap gain | RS−C nonoverlap gain | V−C nonoverlap gain | VS−C nonoverlap gain |
|---:|---:|---:|---:|---:|
| 43022 | `-0.7517` | `-0.0313` | `+0.6837` | `+0.0298` |
| 43122 | `-1.0433` | `-0.1898` | `+0.8938` | `-0.0355` |

At seed43122 the T/U terms show why this is not a gain-only artifact. REPEAT_SPLIT versus CLEAN improves both true-source and unrelated-source NLLs (`ΔT -0.3806`, `ΔU -0.5704`) rather than preserving original REPEAT's true-source-worse / unrelated-source-better reversal. VIEW_SPLIT versus CLEAN also improves both terms (`ΔT -0.8164`, `ΔU -0.8519`) rather than preserving local VIEW's source-specific positive residual. The large residuals therefore require local relation practice.

The fixed-budget bridge must be narrow. The compact neutral-prefix local-versus-split differences were about `+0.3027` for R−RS and `+0.2688` for V−VS, but deterministic-mask MLM loss on 6,992 ordinary held-out rows gave only `+0.0210` and `+0.0203`. Same-window pairing reallocates prediction work inside the compact companion/rewrite subdistribution; it is not a measured broad degradation of ordinary held-out language modeling at this scale.

## Asymmetric behavioral and mechanism readings

Entity behavior makes the REPEAT side more mature than the VIEW side. Across original DeBERTa seeds, REPEAT has a stable behavioral crossover: it beats CLEAN at zero relevant queried-entity updates (`+8.78`, `+9.09`, `+3.11`) and falls below CLEAN at deeper queried-state updates, with rel≥3 weighted deltas `-3.61`, `-1.30`, and `-4.50`. At seed43022, splitting local exact recurrence removes this behavioral face: R−C is `+8.78` at zero relevant updates, but RS−C is `-0.54`; rel≥3 moves from R−C `-3.61` to RS−C `-1.08`. This aligns the exact recurrence effect across compact T/U/N, source-output mass, natural copy, and Entity.

VIEW is different. Local VIEW has strong compact source-conditioned support, and VIEW_SPLIT removes that compact residual in two seeds. But seed43022 official Entity showed VIEW_SPLIT retaining much of VIEW's positive-update behavior while losing zero-update behavior: rel≥3 VS−C `+5.36` versus V−C `+5.28`, and rel_eq0 VS−C `-10.08`. Therefore the VIEW Entity behavior cannot be used as the same evidence object as the compact restatement residual. It may be a separate content/exposure effect, or it may become a two-sided grounded-versus-ungrounded change behavior if seed43122 split Entity repeats the pattern. That second-seed official Entity readout is already running at this stage.

The source-output mechanism is also asymmetric. The corrected relation practice principle 2×2 shows REPEAT raises true-source content-token probability mass by about 11–14 times more under the true related source than under an unrelated compact source, and suppresses target probability only in the true-source condition. VIEW also recognizes source content but raises target probability rather than suppressing it. accidental target context audit data-only checking shows token-nonoverlap targets are absent from the true source and neutral source, and appear in only `0.51%` of unrelated source slots, so the T/U/N effects are not hidden answer-token visibility. The remaining mechanism issue is concentration: whether the REPEAT output change is a diffuse recognized-source content pull or a precise copy-like allocation onto a small set of source tokens. That affects wording and possible figure design, but it does not overturn the locality spine.

## Evidence carriers to use directly

The core result is carried by these files:

| role in the argument | file(s) |
|---|---|
| original three-seed T/U/N relation direction | `research/notes/relation_learning/original_threeseed_neutral_anchor.md`, `experiments/archive/relation_learning/data/original_threeseed_neutral_anchor/rewrite_TUN_across_seed_contrasts.csv` |
| seed43022 split T/U/N and compact neutral-prefix price | `research/notes/relation_learning/neutral_anchor_rewrite_probe.md`, `experiments/archive/relation_learning/data/neutral_anchor_rewrite_probe` |
| two-seed DeBERTa split locality collapse | `research/notes/relation_learning/split_seed_replication.md`, `experiments/archive/relation_learning/data/split_seed_replication/rewrite_late_contrasts.csv` |
| ordinary held-out MLM narrowness of price | `research/notes/relation_learning/ordinary_heldout_price_probe.md`, `experiments/archive/relation_learning/data/ordinary_heldout_price_probe` |
| source-specific output mass 2×2 | `experiments/archive/relation_learning/data/source_specificity_misfire/specificity_late_terms.csv`, `experiments/archive/relation_learning/data/source_specificity_misfire/specificity_late_contrasts.csv` |
| accidental U/N answer visibility check | `research/notes/relation_learning/accidental_target_context_audit.md`, `experiments/archive/relation_learning/data/accidental_target_context_audit/target_context_occurrence_summary.csv` |
| Entity relevant-update behavior and seed43022 split localization | `research/notes/relation_learning/split_entity_official_integration.md`, `experiments/archive/relation_learning/data/split_entity_official_integration/entity_late_contrasts_by_group.csv`, plus original three-seed summary in `research/notes/relation_learning/mechanism_floor_and_behavioral_refinement.md` |
| natural-domain attempt that did not test restatement | `research/notes/relation_learning/childes_variation_probe_score.md` |

## Figures that would carry the science if prepared later

Figure A should show the compact relation-locality result, not a leaderboard table: true-source, unrelated-source, and neutral-source bars or point ranges for R, V, RS, VS, and C, with local-vs-split arrows. The main visual should emphasize Δ(T−N) and Δ(U−N), because gain U−T alone lacks an ordinary-prefix anchor. It should show original three-seed means for R/V/C and two split seeds for RS/VS/C.

Figure B should show locality collapse at the individual-seed level: R−C and V−C versus RS−C and VS−C nonoverlap gains for seeds 43022 and 43122, with T and U components shown adjacent. This figure makes the causal manipulation legible.

Figure C should show the exact-recurrence behavioral face: Entity deltas by relevant queried-entity updates. The mature version should wait for seed43122 official split Entity results; until then seed43022 split Entity can be shown only as first-seed behavioral localization, while the three original seeds carry the stable R−C crossover.

Figure D should show source-output mass as an output-level mechanism: target probability and true-source content mass in true-source and unrelated-source contexts for R−C and V−C. It should include or cite the accidental target context audit accidental-target result so readers know U/N are not usually hiding the answer.

Figure E, if needed, can show the compact-family narrowness of the fixed-budget price: compact N local-vs-split deltas near `+0.27` to `+0.30` versus ordinary held-out deltas near `+0.02`. This keeps the frontier_consolidation connection accurate.

Hash-mixed, aligned-Wikipedia natural restatement, causal-LM, and RoBERTa split results are not needed to establish the central DeBERTa relation-locality result. They decide how far the result travels and how to phrase transfer beyond the compact DeBERTa instrument.

## Live measurements at accidental target context audit close

The following are running or submitted and should be read before using their results:

| experiment | purpose | expected output |
|---|---|---|
| hash-mixed DeBERTa training | graded mixture or context-selected dual competence under same-window half exact / half rewrite relation practice | train result for `training/runs/full_p2c_c2p_abs_hash_mixed_dose2p64x_matched_rowholdout_deberta100M_seed43022`; then score compact T/U/N and natural-copy |
| seed43122 official split Entity | test whether seed43022 behavioral localization and VS change-pattern replicate | `data/split_entity_official_seed43122/`; needs integration with original seed43122 Entity rows |
| RoBERTa REPEAT_SPLIT training | second bidirectional MLM architecture test of exact-recurrence locality | `training/runs/roberta_repeat_split_dose2p64x_rowholdout_100M_seed43022`; then score T/U/N against existing RoBERTa C/R |
| seed43122 split neutral anchor | second-seed compact N price and T/U/N anchor for split arms | `data/split_seed43122_neutral_anchor/`, `notes/022_split_seed43122_neutral_anchor.md` |
| aligned-Wikipedia construction | fair natural restatement instrument, replacing the CHILDES topic-continuation probe | validated aligned-Wikipedia pairs and T/U/N probe records |

## Interpretation boundary

The evidence at this point was asymmetric: exact local recurrence had compact T/U/N, source-output mass, natural-copy, and Entity results; local restatement had strong compact T/U/N and split-locality results, but its behavioral and natural-domain transfer had not yet been established. These differences must remain explicit rather than presenting both mechanisms as equally verified.
