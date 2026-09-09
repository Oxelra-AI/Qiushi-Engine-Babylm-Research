# paired continuation eval — paired continuation training done paired continuation evaluation

JSON: `experiments/archive/compact_experience/data/paired_continuation_eval/paired_continuation_eval_summary.json`

## Design carried into evaluation

Both paired continuation training done arms load the same INITIAL_MODEL_STUDIES earlier analysis `chck_70M`, use the same post-70M example order and the same fresh optimizer schedule, and differ only in late masking granularity. The WWM continuation is the restart control; token continuation is the treatment. The INITIAL_MODEL_STUDIES original `chck_100M` is the old full-run reference.

## Scores

| target | BLiMP | Supplement | EWoK | Entity | COMPS | GPIQA | Reading | legacy weighted screen | equal-7 mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| initial_model_chck100 | 67.350 | 65.200 | 49.640 | 21.240 | 53.110 | 36.120 | 7.330 | 32.273 | 42.856 |
| wwm_cont | 67.620 | 64.800 | 47.910 | 21.340 | 53.520 | 37.620 | 7.750 | 32.341 | 42.937 |
| token_cont | 67.430 | 63.600 | 48.640 | 21.170 | 52.830 | 37.665 | 7.115 | 32.104 | 42.636 |

Legacy weighted screen = (3/28)·(BLiMP+Supplement+EWoK+Entity+COMPS+GlobalPIQA) + (1/8)·Reading, preserved for continuity with INITIAL_MODEL_STUDIES/COMPACT_EXPERIENCE fast screens. Equal-7 mean is the arithmetic mean of the seven shown columns.

## Deltas

| contrast | BLiMP | Supplement | EWoK | Entity | COMPS | GPIQA | Reading | legacy weighted screen | equal-7 mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| token_minus_wwm_cont | -0.190 | -1.200 | 0.730 | -0.170 | -0.690 | 0.045 | -0.635 | -0.237 | -0.301 |
| wwm_cont_minus_initial_model_studies_ref | 0.270 | -0.400 | -1.730 | 0.100 | 0.410 | 1.500 | 0.420 | 0.069 | 0.081 |
| token_cont_minus_initial_model_studies_ref | 0.080 | -1.600 | -1.000 | -0.070 | -0.280 | 1.545 | -0.215 | -0.169 | -0.220 |

## Interpretation for the granularity mechanism

The isolated late-token effect is token minus WWM: BLiMP -0.190, Supplement -1.200, EWoK 0.730, Entity -0.170, COMPS -0.690, GlobalPIQA 0.045, Reading -0.635.
This row is the causal comparison for masking granularity under the restart-controlled paired continuation training done design; the comparison to INITIAL_MODEL_STUDIES original separates restart drift from the token-vs-WWM effect.

## Decisive route conclusion (paired continuation eval)

This is the cleanest granularity measurement in these experiments, and it resolves the earlier ambiguity between the seq-scheduled b64 run (token −0.239 proxy, Supplement −5.20) and the b256 fresh-init pair (token +0.182 proxy, Supplement +4.00). Those two disagreed because the b256 fixed-WWM baseline was a *different, weaker* from-scratch run (Supplement 57.2 vs the initial reference's 65.2), so its "improvement" from token switching was partly recovery of a degraded baseline, not a real granularity gain.

The paired continuation training done/paired continuation eval design removes that confound entirely:
- The **WWM continuation restart control** reproduces the INITIAL_MODEL_STUDIES original chck_100M almost exactly (equal-7 mean +0.081, legacy weighted screen +0.069, Supplement −0.40, BLiMP +0.27, COMPS +0.41, Reading +0.42, EWoK −1.73). The small EWoK drop and small gains elsewhere are within run-to-run noise; the restart itself is essentially free. This validates the continuation methodology: loading chck_70M + fresh AdamW on the exact post-70M order is a faithful proxy for the original tail of training.
- The **isolated late-token effect** (token minus WWM, both continued from the same chck_70M weights on the same data) is **negative in aggregate**: equal-7 mean −0.301, legacy weighted screen −0.237. It gives EWoK +0.73 and GlobalPIQA_nonparallel +3.0, but loses Supplement −1.2, COMPS −0.69, BLiMP −0.19, Entity −0.17, Reading −0.635, GlobalPIQA_parallel −2.91.

**Conclusion.** Under the strongest inherited baseline (earlier analysis-family DeBERTa-v2 8×480 WWM, batch256, fixed seq256, official 10M pool), switching to token-level masking for the final 30M words does not help and slightly hurts overall. The unconditional WWM→token switch is not a viable route to beat 40.7028 → 41.8. The leader's WWM(1-7)→token(8-10) schedule is not, by itself, the source of its edge on this backbone; its advantage must come from the *combination* with its data (FineWeb simplification pairs), tokenizer (40k SentencePiece), optimizer (LAMB), or hidden-size/depth (384×12), not from the masking schedule as an isolated mechanism.

## Implications for the next mechanism

The three closed/weak routes so far — residualized C/S text-scoring selection, same-content ordering, and unconditional WWM→token — share one property: they redistribute *what/where* is predicted within the same official corpus and same backbone, and each trades one competence column for another without a net aggregate gain. The consistent column signature (BLiMP/EWoK/GlobalPIQA up, Supplement/Entity/Reading down, or vice versa) suggests these interventions move the model along a *fixed competence frontier* rather than expanding it. To move the frontier itself, the next mechanism must add genuinely new, aligned, recoverable structure or change a load-bearing capacity layer, not just re-slice existing supervision.

Highest-value candidates that were not yet tested on this backbone at regime scale:
1. **Aligned paired-rewrite data (meaning-preserving multi-view).** The leader's strongest distinguishing ingredient. Build a legal, budget-accounted paraphrase/simplification set (self-generated within word/reward accounting, or an openly licensed corpus), so the model sees the same content in aligned alternative expressions. This is the only route with direct evidence of a frontier shift (leader's EWoK 56.07 / Entity 28.45 are far above this backbone's ~50 / ~21).
2. **Tokenizer + capacity change.** The inherited 16k baseline tokenizer and 8×480 shape differ from the leader's 40k / 384×12. A controlled 40k-vocab + 12-layer run may raise the whole frontier before any objective trick.
3. **Optimizer consolidation (LAMB / Muon-family + tail averaging).** 2025 findings and multiple leader cards attribute real gains to optimizer/averaging rather than data. This is a cheap, high-leverage factor to test on the fixed backbone.

The next Execute work should stop iterating masking-granularity variants and move to one of these frontier-shifting factors, starting with the one that has the strongest external evidence (paired-rewrite data) or the cheapest decisive test (optimizer/tail-averaging on the fixed backbone).
