# b256 fixedseq pair eval raw — route update after the b256/fixed-seq WWM→token pair

Evidence files:

- Pair fast eval wrapper: `experiments/archive/compact_experience/data/b256_pair_eval/b256_pair_100M_eval_summary.json`
- Pair trajectory eval: `experiments/archive/compact_experience/data/b256_pair_eval/b256_pair_trajectory_eval_summary.json`
- Mechanism note: `research/notes/compact_experience/b256_fixedseq_pair_mechanism.md`
- earlier analysis-style direct recheck: `experiments/archive/compact_experience/data/b256_pair_eval/b256_pair_revision_322style_direct_recheck.json`
- earlier analysis alignment comparison: `experiments/archive/compact_experience/data/b256_pair_eval/alignment_comparison.json`

## What changed in b256 fixedseq pair eval raw

The b256/fixed-seq comparison was incomplete in wave 2, but wave 1 completed the matched pair. Both completed arms are valid 100M-exposure checkpoints:

- `experiments/archive/compact_experience/training/runs/wwm_fixed_100M_b256_seq256_seed43/hf_model/chck_100M`
- `experiments/archive/compact_experience/training/runs/wwm_to_token_100M_b256_seq256_seed43/hf_model/chck_100M`

They share official corpus, DeBERTa-v2 8×480, baseline16k tokenizer, fixed seq256, batch256, seed43, 100M word exposure, and identical training through the 70M switch point. This makes them a clean within-run-pair test of the late WWM→token intervention under the COMPACT_EXPERIENCE curriculum trainer.

The fast pair result differs from the earlier seq-length-scheduled b64 run:

| regime | WWM→token minus fixed proxy | main positive columns | main negative columns |
|---|---:|---|---|
| seq-scheduled b64 masking curriculum 4m eval/011 | -0.239 | BLiMP +0.94, EWoK +1.10, GlobalPIQA mean +3.41 | Supplement -5.20, Entity -1.90, Reading -0.46 |
| fixed seq256 b256 b256 fixedseq pair eval raw | +0.182 | BLiMP +0.27, Supplement +4.00 | EWoK -0.46, Entity -1.17, COMPS -0.54, Reading -0.345 |

The trajectory confirms causality within each pair: at 70M, before the switch, the arms are effectively identical; after token-level masking begins, task allocation changes. The stable evidence is not that global switching is good or bad, but that late token-level prediction pressure changes the distribution of competence across columns and often pressures Reading/Entity-like continuity.

## Direct recheck and baseline alignment

Because the COMPACT_EXPERIENCE fixed-WWM run scored far lower on Supplement than the old INITIAL_MODEL_STUDIES earlier analysis fixed-WWM coordinate, I adapted the exact INITIAL_MODEL_STUDIES earlier analysis direct evaluator to the COMPACT_EXPERIENCE pair. The direct recheck reproduces the b256 fixedseq pair eval raw wrapper scores exactly:

| checkpoint | BLiMP | Supplement | EWoK | Entity | COMPS | Reading | mean6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| COMPACT_EXPERIENCE b256 fixed chck_100M | 67.36 | 57.20 | 50.91 | 24.95 | 52.72 | 8.230 | 43.562 |
| COMPACT_EXPERIENCE b256 WWM→token chck_100M | 67.63 | 61.20 | 50.45 | 23.78 | 52.18 | 7.885 | 43.854 |
| WWM→token - fixed | +0.27 | +4.00 | -0.46 | -1.17 | -0.54 | -0.345 | +0.292 |

The INITIAL_MODEL_STUDIES earlier analysis direct recheck for the original earlier analysis fixed-WWM `chck_100M` was:

| checkpoint | BLiMP | Supplement | EWoK | Entity | COMPS | Reading |
|---|---:|---:|---:|---:|---:|---:|
| INITIAL_MODEL_STUDIES earlier analysis fixed chck_100M | 67.34 | 65.20 | 49.64 | 21.24 | 53.11 | 7.330 |

The Supplement mismatch is therefore not caused by the b256 fixedseq pair eval raw wrapper or by parent/revision loading. It remains when using exact checkpoint paths, no revision name, batch size 64, and the same fast task paths as INITIAL_MODEL_STUDIES earlier analysis. The mismatch is large enough that the COMPACT_EXPERIENCE b256 pair cannot be treated as a reproduction of the INITIAL_MODEL_STUDIES 40.7028 coordinate.

The alignment comparison found that high-level source word counts, epoch summaries, model config, and tokenizer SHA match. But the COMPACT_EXPERIENCE manifest does not store the exact consumed example ID list, and the trainers differ in an important implementation detail. The original INITIAL_MODEL_STUDIES `babylm_masked_train_fullcycle.py` does not reset model initialization with `args.seed` before `build_model` unless `--extra_init_seed` is set; the COMPACT_EXPERIENCE curriculum trainer explicitly resets all RNGs with `args.seed` before model construction. Thus the COMPACT_EXPERIENCE fixed-WWM run is a seed43-initialized rerun, not the old earlier analysis initialization. This is a plausible source of the different Supplement/Entity/Reading surface.

## Scientific interpretation now

1. The COMPACT_EXPERIENCE b256 pair is a valid within-pair granularity experiment, but not a true aligned earlier analysis reproduction.
2. The earlier conclusion that simple WWM→token is negative was too strong because the fixed-seq/b256 pair gives a small positive fast proxy. The opposite conclusion that WWM→token should be scaled is also unsupported because the signal is smaller than the baseline displacement and omits full official columns.
3. The useful invariant is post-70M redistribution: token-level masking can improve lexical/sentence judgments while pressuring entity, reading, and sometimes broader relational columns. The route should seek an additive multi-scale mechanism rather than a blind global switch.
4. Before building conditional granularity, the strongest baseline-aligned comparison should use the old earlier analysis trajectory itself, not a new from-scratch COMPACT_EXPERIENCE rerun with different initialization.

## Strongest next experiment

Construct a true earlier analysis-aligned late-token continuation:

- Start from the existing INITIAL_MODEL_STUDIES earlier analysis fixed-WWM checkpoint at `hf_model/chck_70M`.
- Continue only the last ~30M word exposures with token-level masking, using the exact original earlier analysis data order after the 70M checkpoint boundary and the same optimizer/LR schedule phase as the original 100M run.
- Compare the resulting `earlier analysis-chck70M + token-to-100M` checkpoint against the existing INITIAL_MODEL_STUDIES earlier analysis fixed-WWM `chck_100M` through the same direct fast columns and then full official-compatible columns if the fast split is meaningful.

This avoids the unobservable original initialization problem because the pre-switch weights are exactly the old earlier analysis weights. It also preserves causal interpretability: the only intended change after 70M is WWM versus token-level prediction. Implementation needs care:

- identify the exact `chck_70M` actual exposure and optimizer step count from INITIAL_MODEL_STUDIES metrics;
- reconstruct the post-70M example subsequence from INITIAL_MODEL_STUDIES `example_order_manifest.json` (`consumed_example_ids_in_order` is present there);
- load the model from `chck_70M`, initialize AdamW/scheduler so LR resumes the earlier analysis cosine schedule from the correct global step, and train until the original 100M exposure point;
- log exact consumed IDs, mask mode, LR, steps, and checkpoint paths.

If this continuation reproduces the COMPACT_EXPERIENCE b256 direction (Supplement/BLiMP gain with Entity/Reading cost), then conditional multi-scale granularity becomes a serious next mechanism. If it matches the seq-scheduled negative direction or damages the old earlier analysis Supplement, global late token switching should be treated as unreliable, and the next route should use either function-conditioned granularity, competence-targeted WWM replay, or paired-rewrite data rather than a hard global switch.

## Candidate mechanism after the aligned continuation

The proposed mechanism principle is: finite-experience learning should add prediction constraints at multiple length-scales rather than replacing whole-word/long-range constraints with token/local constraints. If the aligned continuation shows a real token benefit, the next mechanism should be a controlled additive granularity experiment:

- **function-conditioned granularity:** token-level masking on locally recoverable subword/morphological content, WWM on repeated entities, operator/function words, relation/cross-clause positions, and long-range anchors;
- **dual-granularity late updates:** a matched predicted-position budget combining WWM and token targets, with controls for supervision amount;
- **competence-targeted WWM replay:** late token pressure on ordinary batches while reserving a small WWM stream for entity-chain and clause-rich examples.

These should not be mixed with paired-rewrite data until the pure granularity mechanism is isolated.
