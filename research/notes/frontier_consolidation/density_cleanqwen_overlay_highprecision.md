# density cleanqwen overlay highprecision FineWeb density contrast on clean-Qwen base

This construction preserves all inherited `qwen_pair_packed` rows from the clean-Qwen 10M corpus, removes an exactly word-matched slice from the remaining official/non-Qwen words, and inserts the source-aligned FineWeb near/compact density blocks. It is designed to map any density effect closer to the 41.34 inherited anchor than the official-only filler construction.

Changed-block budget: 217,120 words. Preserved inherited Qwen pair words: 1,656,800. Common filler words: 9,782,880.

Metadata: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_highprecision/density_cleanqwen_overlay_metadata.json`
