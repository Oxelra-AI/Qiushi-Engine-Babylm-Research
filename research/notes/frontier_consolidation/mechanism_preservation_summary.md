# mechanism preservation summary — paired-view mechanism preservation under leader-style curriculum

## Central finding (real clean-Qwen 10M + exact 40k tokenizer)

The necessary distinction is that near-unity Qwen *word* visibility under repaired pilot eval row chunking does
not establish that the *mechanism* (each original+rewrite pair jointly inside one attention
window) survives. I measured this directly.

Complete original+rewrite **joint** visibility (all whitespace words of both views inside one
training example/window), over 37,594 Qwen pairs, leader 40k tokenizer:

| arm | complete joint | partial joint | mean examples touching a pair |
|---|---:|---:|---:|
| inherited fixed-seq256 row-atomic | 0.9993 | 0.9999 | 1.00 |
| repaired pilot eval row-chunked stagewise (seq64/128/256) | 0.5720 | 0.9871 | 1.51 |
| pairfit stagewise (token-fit atomic pairs) — actual materialized | 1.0000 | 1.0000 | 1.00 |

Row-chunk stage breakdown (40k): stage1 seq64 complete joint = 0.084, stage2 seq128 = 0.461,
stage3 seq256 = 0.999. The collapse is concentrated in the early short-sequence stages, exactly
where the leader-style curriculum spends its first two thirds of word budget.

Evidence:
- `data/pair_joint_visibility/pair_joint_visibility_audit.json`
- `data/pairfit_contract/pairfit_joint_visibility_contract.json`
- `notes/pair_joint_visibility_audit.md`, `notes/pairfit_joint_visibility_contract.md`

## Consequence for the in-flight run

The repaired row-chunked stagewise pilot (10M, 12×384/40k/LAMB/wwm→token) is now
scientifically a **confounded** arm: a weak or ambiguous fast score would conflate the leader
curriculum effect with destruction of the paired-view signal, and even a positive score would be
mechanistically ambiguous. It remains useful only as a low-hidden-word training-dynamics control,
not as evidence about whether curriculum helps the clean-Qwen data principle.

## Repaired Comparator

`data/qwen10_stagewise_pairfit/` — pair-boundary-aware 10M stagewise corpus:
- every Qwen original+rewrite pair is one atomic example
- assigned to shortest stage whose seq window contains it under the exact 40k tokenizer:
  tokens ≤64 → stage1 (seq64), ≤128 → stage2 (seq128), ≤256 → stage3 (seq256)
- pair stage counts 27,144 / 10,259 / 191; pair words 998,731 / 640,856 / 17,213
- non-Qwen words fill exact remaining budgets in original order (chunks ≤32/64/160)
- all 37,594 pairs present exactly once, 1,656,800 pair words, **zero truncation**, complete
  joint visibility 1.0; total 10M words; trainer loader accepts all three stage files at exact
  word totals.

Same optimizer/architecture/masking/tokenizer geometry as the in-flight run; only the data arm
differs. Launch-ready: `scripts/launch_pairfit_stagewise.sh <gpu>` (reuses
`stagewise_trainer_amp.py`), target run dir
`training/runs/qwen10_stagewise_pairfit_40k_12x384_lamb_bf16_seed44011`.

## Next actions
1. Train the pairfit arm when computational resources are available.
2. When the row-chunked pilot completes, evaluate its `chck_10M` and the pairfit `chck_10M` with the same
   fast official-compatible tasks (`fast_eval_repaired_pilots.py`), plus the inherited
   fixed-seq256 clean-Qwen result as the anchor.
3. Decide the training-dynamics question from the **triangle** {fixed-seq anchor, row-chunk
   confounded control, pairfit mechanism-preserving arm}. If pairfit ≥ anchor on EWoK/Entity/
   GlobalPIQA while holding Supplement/Reading/SuperGLUE, scale it; otherwise the gap to 41.8 is
   not curriculum and we return to information-efficient second-view construction (denser relations)
   and the AoA checkpoint-ladder opportunity.
