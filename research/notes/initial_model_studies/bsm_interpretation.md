# bsm interpretation — BSM pretest interpretation: binding is a signal-density problem

## The decisive result

The base (frozen 100M) DeBERTa has near-zero entity-conditioned binding:
- both_correct: 2.5%, same-value preference: 96.3%
- This confirms clbh binding repair result: the pretrained model overwhelmingly prefers one value regardless of query entity.

After just **75 gradient steps of plain MLM** on 200 binding-pair texts (no binding-switch margin, no added parameters, no architecture change):
- Train both_correct: **100%**, same-value preference: 0%
- Heldout entities: **96.9%**
- Heldout values: **97.1%**
- Heldout templates: **100%**
- Heldout order flips: **100%**
- Mean margins: +11 to +12 logits

The BSM margin loss adds nothing over plain MLM — because plain MLM already saturates when training examples are concentrated binding patterns.

The random-correspondence control (same BSM structure but randomized entity-value labels) reaches only 76.5% on train and 64-81% on held-out — confirming that the correct entity-value correspondence, not just exposure to the text pattern, drives the effect.

## What this means

**The bottleneck is NOT representation capacity or architecture.** The ordinary DeBERTa encoder + MLM head can trivially learn entity-conditioned value selection. It transfers to held-out entities, values, templates, and order flips in under 100 steps.

**The bottleneck IS training signal density.** In 100M words of official pretraining text, entity-value binding situations are rare and buried among frequent unrelated text. The model never forms the binding habit because the gradient signal from scattered binding examples is overwhelmed by the majority of non-binding MLM targets. When binding examples are concentrated (100% of training distribution), the model learns immediately and transfers.

**This eliminates the relation-architecture route as the primary direction.** Soft latent slots (LEBS), relation-factorized attention, and entity-conditioned output factorization (ECOF) would all add complexity to solve a problem that plain MLM already solves when it sees enough binding examples. The real question is how to get sufficient binding signal density during compliant pretraining on official text.

## What this does NOT establish

1. Whether the binding effect survives when mixed with 99% non-binding text at realistic pretraining proportions.
2. Whether sufficient binding examples can be mined from official BabyLM text.
3. Whether the effect transfers to official Entity Tracking, EWoK, or GlobalPIQA evaluation items.
4. Whether binding-concentrated fine-tuning damages Supplement, Reading, or BLiMP.
5. The minimum density/proportion of binding examples needed for the effect to emerge and persist.

## Route consequence

The next experiment should NOT be another architecture variant. It should test:

1. **Binding signal density threshold**: at what mixing ratio (binding:non-binding examples) does the model still learn entity-conditioned switching? E.g. 50%, 20%, 10%, 5%, 1%.

2. **Mining from official text**: can entity-value binding situations be identified in BNC/CHILDES/Wiki/Gutenberg using lightweight heuristics (entity mention + property predication + later re-mention)?

3. **Persistence under continued pretraining**: if binding is taught early and then the model continues on normal official text, does it persist or decay?

4. **Transfer to official evaluation**: does binding learned from generated/mined examples improve Entity Tracking, EWoK, or GlobalPIQA on the official evaluator?

The strongest immediate next step is a **density-threshold experiment**: take the same 200 binding pairs, mix with varying proportions of official text chunks, fine-tune from the same 100M checkpoint, and measure at what ratio binding still emerges. This directly determines whether a compliant pretraining strategy is feasible.

## Evidence files

- Script: `scripts/binding_switch_margin_pretest.py`
- Results JSON: `data/bsm_pretest_results.json`
- Results note: `notes/bsm_pretest_results.md`
- Plan that led here: `plans/route_decision_binding_switch_objective.md`
