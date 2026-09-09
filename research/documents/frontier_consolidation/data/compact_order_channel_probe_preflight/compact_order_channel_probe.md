# compact order evaluation and channel readout plan compact ordered-vs-scrambled source-absent channel probe

Fixed ordered compact-side denoising readout on the compact order experiment design arms. The main channel is source_absent_content. Negative ordered_minus_scrambled means ordered training gives lower NLL on the same ordered source+compact masked event, i.e. preserves the source-absent compact target channel better than scrambled training.

Events: 60 from experiments/archive/frontier_consolidation/data/factorial_view_candidate_audit/compact_candidate_pairs.jsonl with selected counts {'retained_content': 20, 'source_absent_content': 20, 'function_other': 20}.

## Realized loader/WWM small differences from lead route assessment after source use probe
compact-minus-scrambled changed block: Δ active tokens 0.0, Δ candidate groups 8.0, Δ view active tokens 165.0, Δ CPU-proxy masked tokens 557.0, Δ CPU-proxy masked view/source 472.0/183.0.

## Missing checkpoints
- experiments/archive/frontier_consolidation/training/runs/compact_order_scrambled_40M_seed43022/hf_model/chck_20M

JSON: `experiments/archive/frontier_consolidation/data/compact_order_channel_probe_preflight/compact_order_channel_probe.json`
