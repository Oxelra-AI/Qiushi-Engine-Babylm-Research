# Counterfactual order diagnosis and the post-SCMLM route

## Scientific evidence examined

`plans/post_scmlm_route_rebuild.md` and its evidence:
`data/scmlm_r_fourarm_eval.json`,
`data/capacity_probe.json`,
`notes/r1_learning_response_result.md`,
`data/current_scoreboard_status.json`.

The earlier analysis script's `ARCHITECTURAL_LIMIT` label required a mechanistic check. The recorded diagnostic is `scripts/mirror_diagnosis.py`, with result `data/mirror_diagnosis.json`.

## Counterfactual order finding

For same-word-bag counterfactual pairs on the R1 ordered-dynamic chck_5M base:

- Passages A and B differ by mean 16.5 tokens (min 5, max 25): they are genuinely different token sequences, not near-duplicates.
- The final-layer hidden state at the masked query position has cosine similarity **0.99999999** between context A and context B.
- The same candidate answer scored in context A vs context B differs by mean-abs **2.1e-7** log-prob (max 9.5e-7): numerically zero.

**Interpretation.** The mirror failure (`m_A ≈ -m_B`, both-contexts-correct = 0) is not a head-capacity problem and not an optimization/loss-design problem. It is representational at a specific, diagnosable location: the ordinary bidirectional DeBERTa encoder maps the answer/query position to an essentially **operation-order-invariant** representation. The scoring head therefore receives no order signal, and no loss shaped at that position (WWM, answer-MLM, random-neg, or SCMLM-R ranking) can teach the state distinction. This is the proposed unified mechanistic explanation for the R1, SCMLM-R, and (partly) WESS-transfer failures considered here.

This is a genuine scientific result, not just another negative: it explains *why* many prior routes failed and it precisely bounds what any state route must do.

## Verdict on the pivot

The pivot in earlier analysis is **justified**, and this diagnosis makes it sharper:

1. **Stop final-answer state-counterfactual objectives.** Confirmed dead for a principled reason: the query position is order-blind under the ordinary MLM scoring interface. Do not run longer SCMLM-R, lambda/tau/gamma sweeps, or new final-answer ranking variants. They cannot work by construction.

2. **Route A (trajectory-supervised state) survives only in one specific form.** It is not enough to add prefix supervision through the same MLM head if that head is also order-blind at each query position. Any viable state route must force order into the per-position representation itself — e.g., an autoregressive/causal scoring interface (where position k only sees the prefix and therefore *must* encode order), or an explicit per-position recurrent state channel that is exported and used by the scoring function. This is a large intervention and should remain secondary.

3. **Route B (legal custom-corpus on protected DeBERTa WWM) is the right primary SOTA route** — with one caveat below. It targets the real Overall objective, both visible top phenotypes used custom 10M corpora, and it does not depend on solving the order-blindness problem.

4. **Route C (RecGPT-style recursive causal) gains importance from this diagnosis.** RecGPT's causal interface is exactly the kind of scoring function where the answer position sees only the prefix, so it is *not* structurally order-blind. That is a plausible reason a causal phenotype reaches Overall 41.53 with weak Entity but strong everything-else, and it means a causal/recurrent route is a legitimate parallel path to SOTA, not merely a reproduction.

## Concerns the next construction must address (do not skip)

- **Custom-corpus legality and word accounting.** Every word counts toward 10M; any ancillary model used to build/filter data counts its training words too. Use only legally accessible sources with recorded provenance; do not use the gated `go76dof/Fineweb_simplification_pairs`; do not use large-LLM generation unless closed-system accounting is satisfied. The construction step must produce an exact word ledger and source manifest before any training.
- **Do not repeat the GlobalPIQA-only trap.** Prior custom-data screens (WikiAuto/ASSET/structure-density) moved mainly GlobalPIQA and never Entity/EWoK. A custom-mix screen must be judged on whether it produces either (a) real Entity/EWoK movement, or (b) a broad RecGPT-like BLiMP/COMPS/Reading gain large enough to raise Overall — not GlobalPIQA alone.
- **Protect Supplement/Reading.** These are the protected model's two advantages over the leader (+3.84 Supplement, +2.20 Reading). A custom corpus heavy on synthetic/simplified style risks losing them. The screen must report Supplement and Reading every time, and a route that trades them away for a smaller Entity gain is not progress toward Overall.
- **Screen efficiency.** Use 10M–20M exposure matched screens with official-compatible columns before any 100M commitment. Keep the protected 8×480 backbone and baseline16k tokenizer fixed in the first screen so the data effect is isolated from architecture/tokenizer confounds already tested and closed.

## Recommended next action

Proposed design: **legal custom-corpus construction with exact word/source accounting**, then a matched official-only vs custom-mix screen on protected 8×480 DeBERTa WWM at 10–20M exposure, reporting all official-compatible columns with explicit attention to Entity/EWoK/Supplement/Reading (not GlobalPIQA alone). Route C (causal/recurrent scoring interface) remains a documented alternative motivated by the counterfactual order diagnosis, conditional on the custom-mix masked screen reproducing only the GlobalPIQA-only pattern.

The counterfactual order diagnosis provides mechanistic evidence about the entity-state failures and constrains the proposed follow-up routes. Its measurements concern the checkpoint and counterfactual family specified above.
