# SGCR Requires Training-Level Zero-Sharing Parity

Status: completed parity and small-scale execution checks; full SGCR efficacy was unmeasured.

Support-Gated Compositional Residual (SGCR) was intended to change representation sharing without changing baseline optimization. Two implementation discrepancies initially violated that contract. Gradient clipping at norm 1.0 was missing and was restored over the deduplicated standard-plus-SGCR parameters. Component initialization also consumed the global random sequence after training-seed initialization; restoring the training seed after construction repaired dropout alignment.

On the first 78,887 words of the actual stream, standard 12x384 DeBERTa-v2 and SGCR with K=0 then matched after two updates. Both losses were [10.648108288969965, 10.644323993480894], masked counts were [8043, 8405], maximum state difference was 0.0 and maximum probe-logit difference was 0.0.

A target-architecture check recorded losses 10.64593318998416 and 10.64287666007213 with masked counts 8247 and 8199. Nonzero SGCR diagnostics after two updates were mean deviation 0.020724117755889893, maximum deviation 0.0376579575240612 and projection norm 0.15548622608184814. Baked checkpoints loaded as the standard model with 38,421,952 parameters and without SGCR-specific keys.

The prepared full contrast used K=50 and component dimension 64, legal-40k tokens decomposed through legal-16k components, 100M exposure, fixed WWM 0.15 and matched data/model/optimization settings. Exact K=0 parity was necessary for that interpretation, but did not establish that K=50 would improve downstream behavior. A later exact-prefix decomposition correction was still required and is documented separately.
