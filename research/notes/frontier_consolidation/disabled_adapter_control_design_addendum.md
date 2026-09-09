# Disabled-Adapter Matched Control

Status: proposed control configuration; the command record alone does not establish execution or results.

The disabled path was designed to control for gradient checkpointing, the custom model wrapper and additional unused parameters, while retaining the original 100M learning-rate horizon. It set bottleneck 128 and adapter enabled to 0, using the compact-view-reinvestment stream and its compliant 16k tokenizer.

The scientific recipe was hidden size 480, 8 layers, 8 heads, FFN multiplier 4; data seed 43, initialization seed 43022 and training seed 43023; batch 256; sequence and maximum sequence length 256; learning rate 0.001, warmup 0.06 and weight decay 0.01; fixed WWM with start/end probability 0.15. Maximum exposure was 20,008,711 words, checkpoint interval 20,000,000 words and learning-rate horizon 2,529 updates.

This matched early-exposure control was intended to distinguish an active residual contribution from wrapper or execution-setting effects. A shortened learning-rate horizon or a stock model run with different execution settings would not be the same control. No efficacy claim follows from preserving this configuration.
