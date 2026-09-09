# FineWeb Mechanism-Scale Construction and Directed-Event Route Closure

## Route B Directed-Event Yield: CLOSED

Gate 1 measurement found critically low yield:
- 38,167 FineWeb sources searched
- 1,572 transitive events extracted
- **0 same-verb reversed pairs, 2 any-verb reversed pairs, 1 within-doc reversed pair**
- Encyclopedic text almost never contains the same entity pair in reversed agent-patient roles
- Gate 1 assessment: CLOSE (yield < 50 reversed pairs)

Scientific boundary: "without such a reservoir, continue improving the faithful compact-view substrate rather than spending 100M exposure on the auxiliary objective."

Route B and the interaction-isolating EWoK auxiliary objective are closed. All effort returns to the compact-view data substrate.

## Full Qwen3.5-9B Generation: Complete

- 26,015 prompts generated with original Qwen3.5-9B teacher (consistent with 12,152 existing rewrites)
- 1,280,695 tokens, 1605.4 seconds, 797.76 tok/s
- SHA: `cc49cbcbbb6a457fd363a94041d20d579554ae8f450cffbd95283e7d7587cfc5`

## Full Preservation Filter: 22,820 Usable Pairs

Applied full ewok interaction synthesis preservation standard to all 38,167 rows (12,152 existing + 26,015 new):
- **Overall usable: 22,820 (59.8%)**
- **Usable pair words: 813,005** (1.92× the current 423k validated mechanism)
- Existing companion analysis: 8,386/12,152 usable (69.0%)
- New full26k: 14,434/26,015 usable (55.5%)

Top hard failures: missing_entities 4,618, modality 4,248, comparison_direction 4,075, causal_relation 3,231, polarity 1,519

Target was 1,494,110 pair words (54.4% achieved). The 813k is a meaningful mechanism-scale test despite the shortfall.

## Materialized Arms: Both Exact 10M Words

Compact_view and source_repeat arms at `data/fw_full_arms/`:
- 64,183 rows each, exactly 10,000,000 words each
- Official preserved: 8,343,200 words (52,145 rows)
- FineWeb pair words: 813,005 (5,785 rows)
- Retained Qwen: 843,775 words (6,252 rows)
- Neutral topup: 20 words (1 row)
- Both arms use the identical retained source set; only the FineWeb companion differs (compact rewrite vs source prefix-repeat)

## Shared 16k BPE Tokenizer: Ready

Trained on 9,681,149 common words (official + Qwen + FW source, NOT companion text):
- Vocab: 16,384
- Tokens/word: 1.4965 (vs base legal16k 1.4981)
- Zero UNK, zero missing bytes
- Compliant: pool within 10M budget, no external text
- Dir: `data/shared_tokenizer/shared_16k_tokenizer/`

## Mechanism Comparison

18,682 accepted rewrites scored under simplified full ewok interaction synthesis filter:
- 15,589 pass (83.4%) — higher than full ewok interaction synthesis's detailed filter on 12,152 rows (69.0%)
- Main failures: entity_loss 7.8%, causal_direction_lost 5.7%, polarity_lost 3.7%
- Modality_lost only a soft flag (13.3%)

## Pending: SGCR Evaluation Resume

`s97_t12_tool1` remains unresolved. This is the MOST consequential pending evidence:
- If SGCR is near frontier → protect endpoint, launch seed reproduction
- If SGCR is weak → close SGCR, the FW mechanism comparison becomes THE primary route
- In either case, FW mechanism comparison is ready to launch

## Launch-Ready Assets for FW Mechanism Comparison

1. compact_view arm: `data/fw_full_arms/fw_preserved_compact_view_10M.jsonl`
2. source_repeat arm: `data/fw_full_arms/fw_preserved_source_repeat_10M.jsonl`
3. shared tokenizer: `data/shared_tokenizer/shared_16k_tokenizer/`
4. DeBERTa-v2 8×480 architecture (established configuration)
5. AdamW, fixed WWM 0.15, 10 epochs, batch 256

Training should NOT start until the SGCR evaluation is interpreted.
