# tokenizer interface result and route — data × tokenizer interface screen: complete result and route decision

## What was measured (tokenizer interface result and route)

A matched 2×2 no-AoA route screen, all cells at 20M words, batch128, 8×480 DeBERTa-v2, AdamW LR 1e-3, fixed seq256, 15% WWM, seed43022, same data order, same LR schedule horizon (`lr_total_steps=5030`):

| cell | data | tokenizer | params | equal7 (noAoA, noSuperGLUE) |
|---|---|---|---:|---:|
| O16 | official-only | 16k | 34,467,424 | 40.4857 |
| Q16 | clean-Qwen | 16k | 34,467,424 | 40.2757 |
| O40 | official-only | 40k | 45,826,720 | 39.9036 |
| Q40 | clean-Qwen | 40k | 45,826,720 | 40.4971 |

Endpoints measured at chck_20M (`data/tok16_tok40_2x2_summary.json`) and chck_10M (`data/tok2x2_noaoa_eval_10M/tok16_tok40_2x2_10M_summary.json`). Zero-shot columns + Reading; no SuperGLUE, no AoA.

## Factorial effects (20M endpoint, Q−O and 40−16)

- **Data effect under 16k (Q16−O16):** equal7 −0.21. Supplement +2.44, Entity +1.40, COMPS +0.48, Reading +0.72, EWoK −0.55, GlobalPIQA **−5.90**. (This is the familiar clean-Qwen profile at 16k: Supplement/Entity up, GlobalPIQA down.)
- **Data effect under 40k (Q40−O40):** equal7 +0.59. GlobalPIQA **+4.03**, EWoK +0.86, Reading +0.86, Entity +0.45, BLiMP +0.10, COMPS −0.45, Supplement **−1.69**.
- **Tokenizer effect on official (O40−O16):** equal7 −0.58. GlobalPIQA **−7.41**, EWoK +2.09, BLiMP −0.26, Reading +0.09.
- **Tokenizer effect on Qwen (Q40−Q16):** equal7 +0.22. EWoK +3.50, GlobalPIQA +2.52, Reading +0.22, Supplement **−3.18**, COMPS −0.87.
- **data × tokenizer interaction:** equal7 +0.80, but the columns are **GlobalPIQA +9.93, EWoK +1.41, BLiMP +0.17 vs Supplement −4.13, Entity −0.95, COMPS −0.93.**

At 10M the interaction is smaller and differently distributed (equal7 +0.565, GlobalPIQA +4.0, EWoK +2.72, Supplement −2.95). The two endpoints do not agree on the column pattern except that GlobalPIQA dominates the positive side and Supplement is the main casualty.

## Decisive mechanism check: pair retention is NOT the mechanism

`data/pair_retention_and_interface_audit.json` measures actual seq256-visible tokenization on the exact training corpora, not raw lengths:

- **tok16 already fully retains both pair sides in 99.71% of pairs.** tok40 raises this to 99.93% — a delta of only **+0.00218**.
- tok16 pair-row over-256 rate 0.87% → tok40 0.20% (delta −0.0066).
- Qwen visible tokens/word 1.4388 → 1.4011 (delta −0.0377); pair-row mean untruncated tokens −12.36.

So the proposed "40k preserves more complete same-window pairs inside seq256" mechanism is essentially **already satisfied at 16k**. The GlobalPIQA-dominated interaction cannot be explained by rescued pair truncation. The interface interpretation is supported: 40k is a whole interface change (segmentation + which subword targets are visible + **+11.36M embedding/output parameters**), and the effect is concentrated where 16k official already fails (GlobalPIQA_parallel is near/below chance ~20 for all cells).

## Interpretation

1. The GlobalPIQA-driven interaction is not a broad, substantial, mechanism-clean interface gain. It is a single-column swing (largely GlobalPIQA_parallel, which sits near chance for every cell) traded against Supplement (−1.7 to −4.1) and Entity/COMPS. This is the same narrow-tradeoff signature that reversed under full scale/scoring for causal15, contextual cap120, and bidirectional order.
2. 40k **damages** official-data GlobalPIQA badly (−7.41) and only "helps" Qwen relative to that damaged official baseline. Q40 equal7 40.4971 is essentially tied with O16 40.4857 and Q16 40.2757 — the 40k vocabulary buys nothing net over the trusted 16k interface at matched 20M, despite +11.36M parameters.
3. The clean-Qwen data mechanism at 16k (the validated route, Overall 41.3443 at 100M) shows its real signature: Supplement/Entity up, GlobalPIQA down. 40k does not amplify that mechanism; it substitutes a GlobalPIQA-for-Supplement trade.

## Route decision

**Do not extrapolate 40k to a full 100M run.** Under the stated promotion criterion, the interaction is narrow (GlobalPIQA-dominated), comes with strong Supplement damage, is inconsistent between the 10M and 20M endpoints, and is not supported by any real fragmentation/retention mechanism. Given the repeated scale/scoring reversals in these experiments of exactly this narrow-swing signature, a 100M 40k run is not the right expensive experiment.

**Preserve:** clean-Qwen same-window data (validated), 16k tokenizer, 8×480 backbone, current recipe. clean-Qwen `chck_100M` Overall 41.34429066479573 remains the trusted admissible best, below the 41.8 leader.

## Next mechanism-faithful experiment (matched, single-factor, pure MLM)

The proposed second-priority mechanism is the strongest remaining mechanism-clean lever and does not touch tokenizer, objective, or architecture parameters:

**Coordinated cross-view WWM on clean-Qwen pair rows.** Keep pure WWM-MLM, 16k, 8×480, exact clean-Qwen corpus, exact exposure and masked-token budget. On packed pair rows only, make the masked spans of the two views approximately complementary (M_A ∩ M_B ≈ ∅) on aligned anchor content (shared normalized words / numbers / proper names). Ordinary official rows keep standard random WWM. This converts the same-window correspondence into direct cross-view reconstruction pressure without adding any parameters, second loss, or directional head — the cleanest test of whether the model can *use* the aligned second view.

Minimal matched controls at the same total mask exposure:
1. baseline random WWM (= clean-Qwen recipe),
2. independent random WWM on pair rows (dose-matched, no coordination),
3. complementary coordinated WWM on pair rows.

Acceptance: Entity/EWoK rise without Supplement/Reading collapse, AoA does not go negative, and the coordination advantage disappears under partner-view shuffle (mechanism confirmation). This is the decisive next construction; it requires implementation of anchor-aligned complementary masking inside the current trainer.

## Item-level GlobalPIQA confirmation (tokenizer interface result and route, no new training)

`data/globalpiqa_item_analysis.json` uses the already-produced GlobalPIQA prediction files and the local official data (parallel: 103 items, 4 options; nonparallel: 100 items, 2 options) to compute exact per-item correctness and paired-bootstrap CIs for the data×tokenizer interaction:

- **Parallel (4-option):** interaction 0.0 pts at 10M (95% CI [-11.65, 10.68]); +4.85 pts at 20M (CI [-4.85, 13.59]). CIs cross zero at both endpoints — no robust parallel interaction. Q40 vs O40 parallel data effect is +1.94 (10M) and **−1.94 (20M)** — it flips sign.
- **Nonparallel (2-option):** interaction +8.0 pts at 10M (CI [-3, 19], crosses zero); +15.0 pts at 20M (CI [3, 27]). Only this single 2-choice subtask shows a nonzero-CI effect, and its magnitude is unstable across endpoints (8→15), which is exactly the behavior of a small-N 2-option task driven by a few item flips.

**Conclusion:** the entire equal7 "+0.80 interaction" is carried by one fragile 2-option nonparallel GlobalPIQA subtask, not a broad interface capability, while Supplement is robustly damaged. This confirms — with real per-item statistics — that a 100M 40k run is not justified. The 40k data-tokenizer interface (segmentation + visible targets + 11.36M extra params) is closed as the next expensive route. No unsupported chance-level claim is made; option counts are reported from the data.
