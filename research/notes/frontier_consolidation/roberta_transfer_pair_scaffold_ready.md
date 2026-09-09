# roberta transfer pair scaffold ready RoBERTa compact-vs-repeat transfer scaffold ready

## compact order result and roberta transfer decision supersession

This roberta transfer pair scaffold ready 40M scaffold is superseded by `notes/compact_order_result_and_roberta_transfer_decision.md` and `data/roberta_full100m_transfer_scaffold/`. The compact order result and roberta transfer decision scientific correction is that the natural compact-vs-repeat RoBERTa transfer question is not logically gated on ordered-vs-scrambled succeeding, and 20M/40M exposure is insufficient to close a known late-emerging effect. compact order result and roberta transfer decision therefore verified the original 100M compact/repeat streams and launched a full-100M RoBERTa transfer pair with 10M checkpoints. Treat the remainder of this roberta transfer pair scaffold ready note as historical engineering context for the smaller 40M scaffold, not as the active launch plan.

## Status

This note records CPU/file-only preparation for an earlier possible model-coordinate transfer experiment. It did **not** start H100 training, selected official-compatible evaluation, model upload, SuperGLUE/AoA work, or leaderboard submission.

## Why this scaffold exists

The current strongest positive causal evidence is still the compact-view/reinvestment corpus effect under DeBERTa-v2 masked LM. The next scientific question, after the ordered/scrambled result is delivered, is whether the compact-data marginal transfers to a different bidirectional MLM encoder or is specific to DeBERTa-v2's disentangled-attention coordinate. A paired RoBERTa/BERT-style test under ordinary WWM is the smallest architecture-transfer test that avoids reopening the failed explicit source-absent target-prioritization family.

This scaffold prepares the data and commands for the proposed transfer test; it does not establish a training result.

## Data pair materialized and audited

Output root:

- `experiments/archive/frontier_consolidation/data/roberta_transfer_pair_scaffold`

Key artifact:

- `roberta_transfer_pair_scaffold.json`
- `roberta_transfer_pair_scaffold.md`
- `repeat_compact_reinvest_40M.jsonl`
- `repeat40_preflight/preflight.json`
- `repeat40_smoke_forward/smoke_forward.json`

The repeat 40M stream was materialized deliberately in lead route assessment after source use probe ordered 4×10M row order, not copied from the legacy 100M stream, because the legacy 100M streams have a different row order and would not row-align with the already running compact ordered experiment.

Audited identities:

- compact 40M: `experiments/archive/frontier_consolidation/data/compact_order_factorial_pool_scaffold/compact_ordered_40M.jsonl`
  - SHA256 `48313173a4cd93e768f490c8a5eaaa850fc828fabc9cfb21ebad73651bef85dc`
  - 258,960 rows, 40,000,000 words
- repeat 40M: `experiments/archive/frontier_consolidation/data/roberta_transfer_pair_scaffold/repeat_compact_reinvest_40M.jsonl`
  - SHA256 `ed0eb4ca34e7057f0e7a401008eb4163112a557973516ac86930142b41a7dca5`
  - 258,960 rows, 40,000,000 words
- compact/repeat 10M source files:
  - compact SHA256 `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`
  - repeat SHA256 `0950c099e7227b8dc6940ea4bf749ce5604b28ff244ca3cf0bf9c102dab2a242`
- spatial repair route status legal tokenizer:
  - JSON SHA256 `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`
  - vocab 16,384

Pair invariants over all 258,960 rows:

- example_id mismatches: 0
- word-count mismatches: 0
- compact-vs-repeat changed rows: 12,020 different (3,005 pair rows × 4 passes)
- changed topup rows equal: 4 (1 × 4 passes)
- filler text mismatches: 0
- filler source-label differences are only lead route assessment after source use probe `passN::` prefixing; text/example_id/word invariants hold

Full one-pass tokenizer audit over the legal spatial repair route status tokenizer, seq cap 256:

- compact total active tokens: 14,294,893
- repeat total active tokens: 14,270,893
- compact-minus-repeat active tokens: +24,000 per 10M pass
- compact-minus-repeat candidate tokens: +24,000 per 10M pass
- compact-minus-repeat word groups: -196 per 10M pass
- compact-minus-repeat truncated rows: +32 per 10M pass
- changed block only: +24,000 active tokens, -196 groups, +32 truncated rows, +0.056668 active tokens/word
- filler: zero active-token/candidate/group/truncation differences
- estimated 40M difference by 4× one pass: +96,000 active tokens, -784 word groups, +128 truncated rows

This difference must be recorded in any future interpretation: compact text creates slightly more BPE active tokens for the same legal word budget, concentrated entirely in the changed block, while filler is exactly matched.

## Engineering checks completed without training

RoBERTa transfer trainer repeat-arm preflight:

- `repeat40_preflight/preflight.json`
- OK true
- repeat JSONL SHA256 `ed0eb4ca34e7057f0e7a401008eb4163112a557973516ac86930142b41a7dca5`
- 258,960 rows, 40,000,000 words
- legal tokenizer SHA256 `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`
- RoBERTa 8×480 parameter count 30,528,064
- intended checkpoints `chck_20M`, `chck_40M`

Repeat-arm CPU smoke-forward:

- `repeat40_smoke_forward/smoke_forward.json`
- finite loss true, finite logits true
- CPU loss 9.775191307067871
- 56 masked tokens out of 368 candidates on 4 rows
- no training started

The compact-arm trainer preflight was already validated in earlier analysis with the compact ordered 40M JSONL and the same legal tokenizer/recipe.

## Correct future command state

The future command manifest in `roberta_transfer_pair_scaffold.json` now uses the actual `eval_custom_checkpoint.py` interface. The intended commands are recorded there and should not be treated as launched.

Training commands, not launched:

- compact ordered RoBERTa on GPU0 to `training/runs/roberta_compact_reinvest_40M_seed43022`
- repeat compact RoBERTa on GPU1 to `training/runs/roberta_repeat_compact_reinvest_40M_seed43022`

Selected scoring commands, not launched:

- compact chck_20M GPU0
- repeat chck_20M GPU1
- compact chck_40M GPU0
- repeat chck_40M GPU1

The adapter scale sweep plan wrapper evaluates one endpoint per command; two waves are appropriate.

## Scientific launch gate

Do **not** launch this RoBERTa transfer pair merely because the scaffold is ready. It is authorized only if the pending compact ordered-vs-scrambled result, channel probe, and exposure-split support the mechanism on stable families:

- selected ordered-minus-scrambled movement on cheap6 without GlobalPIQA / cheap5 without GlobalPIQA and Reading, not volatile columns alone;
- source_absent_content ordered advantage over retained_content and function_other controls in the channel probe;
- CUDA mask-count reconstruction validates the selected-count split;
- no engineering/integrity mismatch in training logs.

If those conditions hold, this scaffold supports the smallest next architecture-transfer test under ordinary WWM. If they do not hold, do not train RoBERTa; remove fluent compact order from the load-bearing explanation and route toward separating content density, tail coverage, and diversity reinvestment.

## Boundaries

- No explicit source-absent mask/loss weighting is implied by this scaffold.
- No leaderboard submission is implied or authorized.
- No selected evaluation has been started.
- The legacy repeat 100M stream is not the paired control for this future screen because its row order differs from the lead route assessment after source use probe ordered 4×10M stream.
