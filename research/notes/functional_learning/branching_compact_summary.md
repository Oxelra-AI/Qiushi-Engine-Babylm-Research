# branching experiment compact trajectory summary

## Acquisition epochs
- seed 42: epoch 300
- seed 43: epoch 500
- seed 100: epoch 400

## Full-objective recovery (switch_qfirst_full)
All 3 seeds: initial disruption then reliable recovery.
- seed 42: nadir top4=0.363 at bep=25, recovery by bep=100, final top4=0.998
- seed 43: nadir top4=0.232 at bep=200, recovery by bep=400, final top4=1.000
- seed 100: nadir top4=0.225 at bep=50, recovery by bep=275, final top4=1.000
Compare: query first binding compact summary fresh qf_full was 1/3 perfect, 1/3 partial, 1/3 bag-level at 500ep.

## Context-only collapse (switch_qfirst_ctx_only)
All 3 seeds: binding destroyed within 25 epochs.
Final top4: 0.275, 0.246, 0.260. Final B-swap: 0.29, 0.06, 0.04.

## Prospective marking (qfirst_block_qctx_ans)
Bound checkpoint evaluated with blocked mask:
- seed 42: std_top4=0.990, blk_top4=0.242
- seed 43: std_top4=0.998, blk_top4=0.256
- seed 100: std_top4=1.000, blk_top4=0.271
Immediate binding loss under blocked evaluation = prospective marking.

Late recovery under blocked training:
- seed 42: transition at ~bep=425-450, final blk_top4=1.000
- seed 43: no recovery, final blk_top4=0.271
- seed 100: transition at ~bep=475, final blk_top4=0.904

## Transfer to original order (transfer_orig_ans_only vs control)
Per-seed transfer at bep=500:
- seed 42: transfer orig_top4=0.238, control=0.250 → NO transfer
- seed 43: transfer orig_top4=0.459, control=0.240 → CLEAR transfer (B-swap +2.50 vs +0.006)
- seed 100: transfer orig_top4=0.262, control=0.254 → NO transfer
Result: 1/3 positive transfer, 2/3 no transfer, 3/3 control no binding.
