# preservation admission standard preservation evidence synthesis

preservation admission standard changed the preservation route from an unfinished promising profile into a measured two-training-seed recipe on the repaired coordinate. The evidence now separates three layers: broad mechanism-consistent Entity improvement, a small reproducible GlobalPIQA item set, and acquisition-order safety from raw AoA refits.

## Evidence files

- Fixed interpretation note: `research/notes/relation_learning/preservation_admission_standard.md`.
- GlobalPIQA item comparison: `experiments/archive/relation_learning/data/globalpiqa_item_table`.
- Official-semantics raw AoA refits: `experiments/archive/relation_learning/data/raw_aoa_refits_official`.
- Earlier loose AoA script/output: `experiments/archive/relation_learning/data/raw_aoa_refits`; keep as a superseded draft because it used a looser child-curve fit and initially missed coherent86. Use the official-semantics output above for the scientific reading.

## Clean-preservation completion

The completed evaluation confirms that `clean_pres_lambda1_eval_seed62065` SuperGLUE completed cleanly on the repaired AutoModel path. The same-coordinate table at `research/documents/functional_learning/data/same_coordinate_all_candidates_after_clean65_superglue/same_coordinate_all_candidates.md` gives:

- coherent86 faithful repaired reference: Overall 42.023967991315104.
- dense_seed62064: Overall 42.14909111936738.
- dense_seed62065: Overall 42.168422702303516.
- clean_pres_lambda1_eval_seed62064: Overall 42.246412332209445, delta +0.2224443408943415 versus coherent86.
- clean_pres_lambda1_eval_seed62065: Overall 42.23173113265801, delta +0.20776314134290885 versus coherent86.

The clean-preservation training seeds are tightly matched: seed65 minus seed64 is -0.014681 Overall, with components BLiMP -0.04, Supplement +0.01, EWoK -0.10, Entity +0.05, COMPS -0.02, SuperGLUE -0.02713, GlobalPIQA 0, Reading -0.005, AoA 0.

The SuperGLUE profile at `research/documents/functional_learning/data/superglue_all_completed_profile/superglue_all_completed_profile.md` shows clean64/clean65 repaired SuperGLUE 69.047711/69.020580. Exact acquisition-only `(M,S)` seed62064 SuperGLUE is 68.887842, so preservation adds +0.159869/+0.132739 SuperGLUE relative to that direct parent on the completed SuperGLUE axis.

## GlobalPIQA item support

The item comparison reproduces the macro values exactly from predictions and gold files: coherent86 has 30/103 parallel and 48/100 nonparallel; all four dense/preservation endpoints have 31/103 and 50/100. Thus GlobalPIQA rises from 38.5631 to 40.0485 by five shared changed items across all four endpoints: four gains and one loss, net +3 items.

The shared changed items are:

1. `parallel_ex000000_eng_latn`: air-filled sealed plastic bag under load; all candidates switch from "air increases" to the correct "air stays the same".
2. `parallel_ex000071_eng_latn`: candle wick length; all candidates switch from "shorter" to the correct "longer than wax height".
3. `parallel_ex000094_eng_latn`: cookies/housework time; all candidates switch from correct "Drying the laundry" to wrong "Folding the laundry".
4. `group0123_ex000032_eng_latn_0_v1`: polymer clay sprinkles; all candidates switch from twenty minutes to the correct one minute.
5. `group0123_ex000084_eng_latn_0_v1`: protecting a counter from iron heat; all candidates switch from wet towel to the correct dry towel.

This supports a reproducible recipe effect, but its scope is small. It should not be weighed like the broader Entity movement.

## Raw AoA safety

The official-semantics refit uses the same structure as the BabyLM AoA evaluator: bounded child sigmoid at 0.5, mean surprisal per word and checkpoint, subword length by prefix subtraction, model AoA as log10(step+1) at halfway from random surprisal to minimum surprisal, then raw Pearson before p>0.1 clipping. Results:

- coherent86: r -0.040198, p 0.545925, n 228.
- dense_seed62064: r -0.033354, p 0.622688, n 220.
- dense_seed62065: r -0.029103, p 0.668426, n 219.
- clean_pres_seed62064: r -0.026870, p 0.690512, n 222.
- clean_pres_seed62065: r -0.041864, p 0.534926, n 222.
- densemask_sparselabel_seed62064: r -0.032574, p 0.630856, n 220.

No endpoint approaches a harmful significant negative relation; all store official AoA zero under p>0.1 clipping. The raw values differ slightly from the earlier earlier analysis local coherent86 number because the present refit uses batched 18-step full files and official-like refitting over those files; the sign and safety reading are unchanged.

## GPU and task state

- Dense62065 second SuperGLUE seed evaluation was cancelled after only BoolQ at 67.951070 had completed and MultiRC had started; it is not a complete result.
- The duplicate clean65 SuperGLUE evaluation was cancelled/failed after only BoolQ at 67.522936. It must not be used as clean65 evidence; the completed clean65 payload is the valid source.
- Clean-preservation seed62064 with downstream fine-tuning seed44 was running, with output `experiments/archive/relation_learning/data/clean_pres62064_ftseed44_superglue`.
- The 30M enrichment calibration arm was still running. At this stage it had 5.14M/30M words, effective mask rate near 0.15, loss near 5.40, and pace roughly 18--21 seconds per optimizer step; the full run had 759 steps.

## Scientific meaning

The practical route is now much stronger than dense alone. Dense `(M,M)` improved Overall but depended heavily on the five GlobalPIQA item changes and paid BLiMP/Supplement/EWoK costs. Clean preservation keeps the same Entity and GlobalPIQA movement while reducing those costs and also improves SuperGLUE relative to exact `(M,S)` acquisition-only. The active scientific reading is that denser corrupted-context evidence teaches a useful source-conditioned behavior, while preservation limits destructive drift of inherited short-input lexical behavior. The remaining work is not to declare a final endpoint immediately: complete downstream-seed replication on the preservation endpoints, then use the raw AoA and item-level evidence to decide whether the short/no-special preservation synthesis should be trained as the next fixed two-seed experiment to further reduce the remaining short-input costs.
