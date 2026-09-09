# Coordinated cross-view complementary WWM: construction specification

## Trusted state
- Trusted admissible best: clean-Qwen 8×480 seed43022 `chck_100M` Overall **41.34429066479573** (AoA 0), below visible leader 41.8. No official SOTA established.
- Closed as SOTA routes: tail/mask allocation, mixed causal batch substitution, MLM-primary MNTP auxiliary (execution state Overall 39.7256, AoA −13.41), and now the 40k data-tokenizer interface (tokenizer interface result and route).

## Why 40k is closed (tokenizer interface result and route evidence)
- Matched 2×2 (20M, b128, 8×480, seed43022): O16 40.4857, Q16 40.2757, O40 39.9036, Q40 40.4971 (`data/tok16_tok40_2x2_summary.json`).
- Pair-retention audit (`data/pair_retention_and_interface_audit.json`): 16k already retains both pair sides in 99.71% of pairs; 40k only 99.93% (+0.0022). Fragmentation/truncation is NOT the mechanism. 40k adds +11,359,296 params for no net gain over the trusted 16k interface.
- Item-level GlobalPIQA (`data/globalpiqa_item_analysis.json`): the +0.80 equal7 interaction is carried entirely by one 2-option nonparallel subtask (interaction +15 pts, CI [3,27] at 20M but 10M only +8 CI [-3,19]); parallel interaction CI crosses zero at both endpoints; Q40−O40 parallel data effect flips sign 10M→20M. Robust Supplement damage (−1.7 to −4.1). Not a broad interface gain.

## The decisive next experiment (single factor, pure MLM, zero added parameters)
Convert the validated same-window correspondence into direct cross-view reconstruction pressure without changing tokenizer, objective, or architecture.

Keep everything identical to clean-Qwen: 16k tokenizer, 8×480 DeBERTa-v2, AdamW LR 1e-3, fixed seq256, exact clean-Qwen corpus (`data/qwen_clean_aligned/training_corpora/qwen_aligned_100M.jsonl`), 100M exposure, seed43022, checkpoint ladder, pure WWM-MLM, 15% mask, 80/10/10 replacement.

**Intervention (pair rows only):** on packed Qwen pair rows, define anchor mapping A = {(a_i, b_i)} from within-row aligned content (normalized identical words / numbers / proper names shared between an original segment and its rewrite segment — use `qwen_pair_packed_rows_meta.jsonl` pair_ids + `selected_pairs.jsonl` original/rewrite text; segment spans reconstructable exactly as in `pair_retention_and_interface_audit.py::build_pair_segments`). For each anchor pair, with symmetric probability mask the anchor in view A and keep it visible in view B, or the reverse. Fill remaining mask budget with ordinary WWM. Ordinary official rows keep standard random WWM, unchanged.

**Matched controls (all at identical total masked words AND masked subword tokens, same corpus/order/exposure/steps/LR/seed):**
1. baseline random WWM (= clean-Qwen recipe).
2. anchor-targeted INDEPENDENT WWM: same anchor candidate set and same marginal masking distribution as the treatment, but the two views sampled independently (this isolates coordination, not anchor-targeting).
3. anchor-targeted COMPLEMENTARY WWM: the only added factor is cross-view complementarity.

**Must control / report:** masked words and masked subword tokens matched across arms; anchor/non-anchor mask ratio matched; per-view prediction count and direction balance; anchor coverage (fraction of pair-row mask budget under coordination); span-length and lexical-frequency distributions; ordinary rows untouched. Report exact-string-anchor vs proper-name vs number vs content-word strata to prevent a copy shortcut (visible identical string in partner view could let MLM copy rather than form a transferable invariant).

**Mechanism confirmation:** training-time partner-view permutation control (same lengths, mask counts, anchor distribution, only partner shuffled) — a real coordination advantage must vanish under shuffle. Optionally frequency/token-length-matched pseudo-anchor control to separate semantic alignment from mask geometry.

**Acceptance:** Entity/EWoK rise without Supplement/Reading collapse; AoA does not go negative; coordination advantage disappears under partner shuffle. First a no-AoA screen at 20M/b128 on the three arms; promote to a true-100M complete nine-column run only if coordinated beats both controls on target columns without Supplement/Reading damage.

## Reusable assets built in tokenizer interface result and route
- `scripts/summarize_tok2x2_endpoint.py`, `scripts/launch_tok2x2_endpoint_noaoa_eval.sh` — generic endpoint 2×2 screen (works for any chck_XM).
- `scripts/pair_retention_and_interface_audit.py` — exact seq256 pair-side retention + segment-span reconstruction (reuse `build_pair_segments`, `encode_offsets`).
- `scripts/globalpiqa_item_analysis.py` — item-level paired/bootstrap analysis from existing predictions.
- clean-Qwen recipe launchers: `scripts/launch_tok16_b128_20M_matched.sh` (matched 16k b128 template).

Do not reopen tail/mask, geometry, cluster, agreement, SWA, ordering, mix25, MNTP dose sweeps, or the unisolated 12×384/40k/LAMB bundle.
