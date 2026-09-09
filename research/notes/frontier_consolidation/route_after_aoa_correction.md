# aoa developmental signal audit — route after correcting the AoA intervention

## What changed

The proposed AoA curriculum would use official CDI/AoA words in the mask sampler and epoch-specific source mixtures. The developmental-signal audit identified these as invalid training inputs; that training was not launched.

Two CPU-only audits were run instead:

- `experiments/archive/frontier_consolidation/data/aoa_developmental_audit/aoa_developmental_signal_audit.json`
- `experiments/archive/frontier_consolidation/data/aoa_developmental_audit/official_aoa_curvefit_audit.json`

These establish that official AoA identities must remain posthoc analysis objects, not pretraining signals. The official scorer is a fitted model-AoA/child-AoA trajectory correlation over checkpoint surprisals. Existing clean-Qwen and prior legal first-pass source-order models all have negative, non-significant fitted correlations and threshold to AoA 0.0.

## Why not train another AoA-only curriculum now

The prior legal source-block first-pass intervention used only source labels and corpus-internal statistics. It changed the first 10M presentation order while preserving the same 10M corpus and nine later original passes. It produced AoA=0.0 for both seeds and did not improve the broad surface:

- seed43022 Overall 41.3443 -> 40.6475 (Δ -0.6968)
- seed43122 Overall 40.6501 -> 40.5656 (Δ -0.0844)
- two-seed mean Δ -0.3906

Thus pure presentation-order developmental timing is not a supported SOTA lever in the already-tested form. The aoa analysis and curriculum design arithmetic that AoA=+5 would add +0.56 Overall is arithmetically true but scientifically insufficient, because the intervention can and did move other columns.

## Stronger near-term route

The next high-value evidence is broad task-surface evidence, not an AoA-only run:

1. Let representation_and_objectives's already-running semantic-view treatment/control training and no-AoA evaluation finish. It directly tests whether meaning-preserving second views add transferable value beyond same-source repetition.
2. If semantic views are positive across multiple task families without the familiar GlobalPIQA/Reading cancellation, train the already-audited clean-Qwen + semantic-view hybrid contrast.
3. If semantic views are weak or as an orthogonal factor, use the cleaned seq256-safe cached FineWeb candidate (`experiments/archive/representation_and_objectives/training/data/cached_fineweb_seqsafe96_candidate`) as a source-breadth contrast, but only with its matched official length-control and full broad evaluation.
4. Consider a tokenizer-aware exposure/packing control over the exact same corpus if we need a low-confound efficiency test; the seqsafe96 audit shows the replacement block is visible, but the long 160-word official tail still hides ~181k words per pass under seq256.

AoA should be measured passively for SOTA candidates with checkpoint ladders, not optimized by using the official target words or their age labels inside training.
