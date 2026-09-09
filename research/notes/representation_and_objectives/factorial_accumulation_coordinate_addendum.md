# A Symmetric Coordinate for the RoBERTa Factorial

Status: first full-batch attempts failed before scientific training evidence; accumulation preflight and two-update check passed; factorial outcomes were pending.

After the local source-absent denoising channel failed to mediate the 100M downstream gain, the remaining hypothesis was contextual diversification around repeatedly supervised shared anchors. The HS/LS/HD/LD factorial targeted `(HS-LS)-(HD-LD)` on Supplement, EWoK, Entity and COMPS in an independent RoBERTa MLM coordinate, not local pseudolikelihood alone.

The first HS and LS runs failed on their first backward pass through memory exhaustion. A repaired trainer retained effective batch 256, learning-rate horizon 2529, model, tokenizer, seeds, stream order, optimizer and exposure. WWM masks were constructed once on CPU for each full effective batch; 16-row microbatches used active-label-weighted loss accumulation.

This was not bit-equivalent to the failed full-batch GPU-masking trainer. All four factorial cells therefore had to use the same repaired coordinate. Dropout/randomness changes could not be dismissed as exact full-batch parity.

Preflight counted 30,528,064 parameters and 99,999,910 words. The two-update check yielded losses 9.8136896 and 9.7987591, 16 microsteps per update, effective WWM rate approximately 0.150 and first-update learning rate 6.622516556291391e-06. The terminal `chck_100M` name was an alias for 99,999,910 actual words.

The planned completed-arm contract required 2,529 training/optimizer updates, effective batch 256, microbatch 16, seeds 43/43022/43023 and the same 20M-to-100M checkpoint set. Only coherent interaction on stable families would support this mechanism. A null or unstable interaction would not justify further same-premise 100M arms.
