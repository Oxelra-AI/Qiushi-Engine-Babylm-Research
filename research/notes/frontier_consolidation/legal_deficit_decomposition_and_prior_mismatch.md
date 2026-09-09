# legal deficit decomposition and prior mismatch — What the legal tokenizer actually broke, and what it means for the route

CPU-only, evidence-only. Does not poll or infer the mature clean-vs-reinvest tasks
(`s50_t5/t11/t19`), does not launch GPU work, does not change corpus/tokenizer, and does
not tune any prior from evaluation data.

## Two analyses

### A. UID-level decomposition of the legal deficit (`legal_deficit_subtask_decomposition`)
Four endpoints, identical 100M reinvest stream (SHA `3dd19f...`), identical arch/seeds/
recipe, tokenizer-only differences:
- `old_ref_42033` — inherited GPT2-Strict tokenizer, NON-SUBMITTABLE (Overall 42.0331)
- `legal_step35` — same-pool 16k (best legal, Overall 41.2578)
- `legal_bytealpha` — byte-alphabet 16k (Overall 40.7040)
- `legal_a01_ss16k` — lead provenance guard and contingency routes independent legal 16k (partial: BLiMP/Supplement only)

Column averages (legal_step35 vs old_ref): BLiMP 65.87 vs 66.87, Supplement 61.17 vs
63.28, EWoK 50.39 vs 53.67.

Key facts:
- **Two independently built legal 16k tokenizers agree strongly on the deficit shape.**
  On Supplement UID deltas, Pearson(spatial repair route status, ss16k) = **0.82**, sign agreement **1.0**;
  on BLiMP, Pearson **0.65**. independent ss16k landed at BLiMP 66.33 / Supplement
  59.28 — essentially identical to byte-alphabet endpoint. The legal deficit is a
  **reproducible representation property of the Strict-Small-budget tokenizer**, not
  tokenizer noise or a single build defect.
- **BLiMP losses are syntactic binding / filler-gap, not relation content.** Biggest
  losses: `principle_A_reconstruction` −20.89, `wh_questions_object_gap` −17.34,
  `animate_subject_trans` −11.91, `tough_vs_raising_1` −10.97, `anaphor_gender_agreement`
  −10.71, `matrix_question_npi_licensor_present` −9.47. Biggest gains:
  `only_npi_licensor_present` +24.26, `left_branch_island_echo_question` +19.96,
  `tough_vs_raising_2` +12.39.
- **EWoK loss is dynamic + property, not spatial.** physical-dynamics −19.5,
  social-properties −13.75, material-dynamics −8.83, quantitative-properties −5.8; but
  spatial-relations **+3.06** and material-properties **+4.97**. Mean relation-domain Δ
  −3.16, mean property-domain Δ −3.50 — the loss is roughly uniform across the
  relation/property split rather than relation-selective.
- **Supplement loss is QA-congruence.** qa_congruence_easy −12.5, qa_congruence_tricky
  −3.64; turn_taking, subject_aux_inversion, hypernym all slightly *improved*. This
  confirms the deficit is not the spatial repair route status newline `<unk>` issue (turn_taking rose).

### B. Does the existing relation prior target the losses? (`static_prior_eval_alignment`)
Post-hoc diagnostic only: tokenize official eval strings with the spatial repair route status tokenizer, look
up the already-built corpus-only `relation_only_v1` / `relation_info_v1` weights, and
correlate per-UID mean prior pressure with the legal_step35−old_ref delta.

- Correlation is **positive**, i.e. the wrong direction for a repair prior: ALL Pearson
  **+0.145**, Spearman **+0.217**. Higher-prior subtasks tended to lose *less*. Lowest-
  pressure quartile mean Δ **−4.69**; highest-pressure quartile mean Δ **+0.24**.
- The two worst BLiMP losses (`principle_A_reconstruction`, `wh_questions_object_gap`)
  are in the **lowest** relation-prior quartile. Relation-weighted masking would not
  touch them.
- Even in EWoK, the prior does not track the losses cleanly: physical-dynamics (−19.5)
  is high-prior but spatial-relations (+3.06) is also high-prior and gained.

## Route implication (does not replace the mature-trajectory decision)

The prewritten three-way label included "selective relation weakness → single-variable
`relation_only_v1` static-prior screen." legal deficit decomposition and prior mismatch shows that **even if** the mature
clean-vs-reinvest deltas turn out to look relational, the *existing* relation prior is
mismatched to where the legal tokenizer actually loses (binding/filler-gap syntax,
QA-congruence, dynamics). Applying `relation_only_v1` because a label says "relation"
would spend an H100 100M run amplifying tokens that are not concentrated on the deficit.

This does not, by itself, select a route. It sharpens two things for when the mature
deltas arrive:
1. The static-prior branch should be launched only if the mature deltas specifically
   implicate the *amplified* relation/dynamic tokens **and** those align with the
   `relation_only_v1` pressure map — which legal deficit decomposition and prior mismatch shows is not automatic.
2. The reproducible, cross-tokenizer, syntax-and-QA-dominated character of the legal
   deficit is more consistent with a **representation/optimization** explanation (how the
   16k legal vocabulary fragments binding/filler-gap and QA strings) than with a learning-
   signal/relation-masking fix. This aligns with coordinating on legal-40k
   representation route rather than defaulting to a masking objective.

## Artifacts
- `experiments/archive/frontier_consolidation/scripts/legal_deficit_subtask_decomposition.py`
- `experiments/archive/frontier_consolidation/data/legal_deficit_subtask_decomposition`
- `experiments/archive/frontier_consolidation/scripts/static_prior_eval_alignment.py`
- `experiments/archive/frontier_consolidation/data/static_prior_eval_alignment`

## Unresolved Comparison
Clean 80M training, reinvest 70M/80M cheap evaluation and clean 20M/70M/80M cheap evaluation remain pending. The mature 70M/80M reinvest-minus-clean deltas are decisive; the deficit decomposition constrains their interpretation for the static-prior branch.
