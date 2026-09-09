# reopen structured context margin route reopened route: legal coherent-context margin MLM

## Research state entering this route

The protected public endpoint remains scale1.75 `chck_82M` (local Overall 41.942481167385985, public rank 1). Coherent86 and private-scale alpha endpoints are useful numerical candidates, but private scale endpoint vs mechanism synthesis closed their mechanism reading: amplitude control changes mostly COMPS/BLiMP decisions, the alpha0.5 GlobalPIQA edge is only a few examples, and anchor-confidence magnitude does not separate private-residual acquisition from erosion. Given these results, the private-scale family should not be extended with retention KL, confidence modulation, or finer alpha sweeps.

The validated scientific substrate that still matters is the legal compact semantic-view + source-diversity corpus with the function-preserving scale1.75 residual adapter. Its failure mode is not scalar MLM underfitting: the scale1.75 trajectory improves corpus MLM loss to 100M while official relation/state/common-sense competence peaks at 82M and declines. The next mechanism must therefore create a distinct learning pressure for context-dependent structure, not just more lexical/surface MLM convergence.

## Candidate mechanism

Train from scratch on the same legal 100M stream and same scale1.75 adapter architecture, but add a small target-shared coherence-margin loss for masked tokens:

- Construct the usual WWM masked input from a coherent legal row.
- Build a disrupted context for the same row by preserving mask positions, target labels, sequence length, token multiset in unmasked context spans, and short-range spans, while permuting longer-range span order.
- Use the same masked target labels for both contexts.
- Let `nll_coh` be per-target NLL in the coherent context and `nll_bad` in the disrupted context. Add `lambda * softplus(margin + nll_coh - nll_bad)` to ordinary coherent MLM loss.

This pressure is in the same pseudo/log-likelihood channel used by the official zero-shot tasks: it asks the model to make masked target predictions depend on coherent context rather than only local lexical priors. It differs from closed RTD/source-correspondence/private-scale lines because the official MLM head itself is trained, the negative view shares the exact target labels, and the contrast is corpus-internal rather than source-pair correspondence or endpoint amplitude scaling.

## Legal accounting

The disrupted context is an additional training view derived from the legal row and must be charged as words exposed. A run with one coherent and one disrupted forward for a row charges `2 * batch_words`. The first pilots must therefore use a total legal exposure cap that includes both views. This handicaps the method in coherent-row count, so any improvement over the matched standard scale1.75 reference is strong evidence; an early collapse stops the route.

## First GPU test and what it decides

Minimum reliable real-training test after CPU/mechanical smoke:

- Arm A: `cohmargin` from scratch, scale1.75 adapter, legal tokenizer/pool/stream, seed43/train_rng_seed43022, WWM 0.15, total charged exposure 4M or 8M including disrupted views.
- Reference: existing scale1.75 standard trajectory at matched charged exposure where available, or a short matched no-margin run if no comparable checkpoint/evaluation exists.
- Optional Arm B if a second H100 is free: identical code with `lambda=0` but still performing the disrupted forward and charging the negative view, to isolate the cost of halving coherent-row exposure and extra dropout/compute from the margin gradient.

The first test decides only route viability:

- Continue to a 20M paired screen only if the margin arm is not obviously destructive on cheap7 and shows a relation/state-relevant movement (EWoK/Entity/GlobalPIQA/Supplement) not reducible to COMPS/BLiMP churn.
- Stop if it loses broadly relative to the matched reference or repeats the private-scale pattern: aggregate movement from GlobalPIQA few-item flips or COMPS/BLiMP churn while EWoK/Entity stay flat or worsen.
- Do not launch retention KL, endpoint tails, confidence gates, or finer alpha sweeps from this evidence.

## Implementation risk to smoke before GPU training

The load-bearing engineering risk is the disrupted-context constructor. The smoke must verify:

1. target labels and mask positions are identical between coherent and disrupted contexts;
2. special tokens/padding positions remain unchanged;
3. unmasked-context token multiset is preserved per row;
4. non-private model parameters and adapter scale match the scale1.75 reference architecture;
5. word-exposure accounting counts both coherent and disrupted views.

Only after these checks should any GPU training be launched.

## reopen structured context margin route execution status (pilot launched)

- Trainer implemented and CPU-smoke validated: `scripts/coherence_margin_trainer.py`. Smoke `data/coherence_margin_smoke/coherence_margin_smoke.json` shows zero target/mask/special-position mismatches, zero context-multiset mismatches, 64-226 moved context tokens/row, param_count 35,463,008, adapter_params 995,584 (exact scale1.75 adapter).
- Pilot launched on GPU0: 4M charged (view_charge 2.0 => ~2M coherent words), scale1.75 adapter, margin_lambda 0.10, margin 0.20, disrupt_span_tokens 8, lr_total_steps 1265 (full 100M charged horizon geometry), seed43/train_rng_seed43022. Output run dir `training/runs/cohmargin4M_scale1p75_seed43022`, final endpoint `hf_model/final`.

## Matched reference and decision rule

The matched standard reference is the existing scale1.75 standard `chck_4M` checkpoint:
`training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_4M`.
It exists with the adapter modeling source, so no new training is needed for the reference. Note the pilot is handicapped: at 4M charged it sees only ~2M coherent words vs the reference's 4M coherent words. Any non-destructive relation/state-relevant movement is therefore meaningful; a broad loss is decisive to stop.

Matched cheap7 eval command for the pilot final endpoint (run when a GPU is free, isolated roots):

```
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py \
  --arm reinvest \
  --run-dir experiments/archive/frontier_consolidation/training/runs/cohmargin4M_scale1p75_seed43022 \
  --target cohmargin4M_scale1p75_seed43022 \
  --endpoint final \
  --out-root experiments/archive/frontier_consolidation/data/cohmargin4M_eval \
  --collate-root experiments/archive/frontier_consolidation/data/cohmargin4M_collate \
  --gpu <free_gpu> \
  --columns BLiMP Supplement EWoK Entity COMPS GlobalPIQA_parallel GlobalPIQA_nonparallel Reading
```

Matched reference cheap7 (scale1.75 standard chck_4M) if not already scored:

```
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py \
  --arm reinvest \
  --run-dir experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder \
  --target scale1p75_standard_chck4M_ref \
  --endpoint chck_4M \
  --out-root experiments/archive/frontier_consolidation/data/scale1p75_chck4M_ref_eval \
  --collate-root experiments/archive/frontier_consolidation/data/scale1p75_chck4M_ref_collate \
  --gpu <free_gpu> \
  --columns BLiMP Supplement EWoK Entity COMPS GlobalPIQA_parallel GlobalPIQA_nonparallel Reading
```

Decision after both cheap7 vectors exist:
- Continue to a matched 20M-charged paired screen only if the coherence-margin arm is not broadly destructive and shows relation/state-relevant movement (EWoK/Entity/GlobalPIQA/Supplement) not reducible to COMPS/BLiMP churn.
- Stop if it loses broadly vs the reference, or if any gain is only GlobalPIQA few-item flips or COMPS/BLiMP churn while EWoK/Entity stay flat/worse.
- Do not launch retention KL, endpoint tails, confidence gates, or finer alpha sweeps.

## Endpoint Arithmetic and Validation
- alpha0.5 SuperGLUE completed (SuperGLUE 69.78975515726182); the wrapper crash was only a missing cheap7-key read. Repaired arithmetic summary `data/private_scale_superglue_summary/coherent86_private_scale_0p5_sg_superglue_summary.json`: cheap7 44.17785714285714, Overall(AoA0) 42.11497279525131 (+0.1724916278653268 vs chck82). Truthful local carrier materialized (valid) `data/truthful_private_scale_carriers/coherent86_alpha0p5/`. No upload or submission. Per private scale endpoint vs mechanism synthesis, alpha0.5's edge is few-example GlobalPIQA redistribution, not a mechanism.
