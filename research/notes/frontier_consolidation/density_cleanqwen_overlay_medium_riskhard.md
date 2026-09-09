# density cleanqwen overlay highprecision repaired FineWeb density contrast on clean-Qwen base

This row-holdout construction fixes the earlier word-stream overlay risk. It preserves all inherited `qwen_pair_packed` rows and all non-heldout base rows as coherent rows, removes 1,357 intact 160-word official/non-Qwen rows (217,120 words), and uses only that held-out stream for clean lengthmatching/topup.

Changed-block budget: 423,520 words. Preserved inherited Qwen pair words: 1,656,800. Common filler words: 9,576,480.

Metadata: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json`
