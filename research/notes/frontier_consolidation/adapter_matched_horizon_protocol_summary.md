# Matched-horizon adapter test

The earlier adapter runs, which scored approximately 37 points, were interpreted as failures of the compressed training schedule rather than decisive evidence against an additional residual branch. The adapter starts with zero output. Those runs used a standalone 20M-word cosine schedule that reduced the learning rate to zero, whereas the compact-view reference at 20M words is an intermediate checkpoint on a 100M-word schedule.

The corrected comparison uses the same compact-view training stream and permitted 16K tokenizer as the reference: an enabled 128-dimensional adapter, gradient checkpointing, and the reference AdamW and whole-word masking settings. It sets `lr_total_steps=2529` and stops at exactly 20,008,711 words. A matched disabled adapter is required as the implementation control. Measurements include Cheap7, adapter-branch activity, and base-model parameter displacement.

The result was pending when this protocol was written. The comparison tests whether the apparent adapter failure persists after matching the learning-rate horizon. Curriculum, seed, or model-family changes cannot resolve that confound on their own. See the subsequent [matched-horizon results](adapter_matched_horizon_results_summary.md).
