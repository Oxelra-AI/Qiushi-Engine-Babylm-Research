# Cross-View Causal Separation Requires Intrinsic Targets

Status: repaired data/preflight and visible arms completed; blocked-arm results and the interaction were still pending.

The original wrong-partner construction relabeled a donor rewrite against its receiving source. This changed the target mixture: a word intrinsically absent from its own source could become copied under a wrong pairing. Sequential masking also selected different rewrite targets across pairings. Both confounds could mimic a cross-view information effect.

The corrected pool had 12,155 compact pairs and 61,735 filler rows, exactly 10,000,000 words per epoch. Pair mass was 423,511 words in both own and wrong pairings; filler mass was 9,576,489. A derangement used every source and rewrite exactly once with no self-pairs. Intrinsic origin labels traveled with each rewrite: copied fraction 0.832 and source-absent content mass 19,512 words.

All arms loaded identical untrained DeBERTa-v2 8x480 weights. WWM was keyed by source/rewrite/filler identity and epoch, preserving target selection across pairings and visibility conditions. Embeddings received a 2D padding mask; the encoder received a 3D pairwise mask. Preflight found zero origin-label and mask/target mismatches across the 12,155 shared rewrite identities. Full-visible logits matched stock exactly; blocking changed logits by maximum absolute 0.2691181004047394. Initial loss was 9.759902 with 170 finite gradient tensors and 26 masked tokens.

A non-epoch pilot first failed because an unfillable remainder could loop indefinitely. The corrected stopping rule completed 499,999 words in 36 seconds. This correction did not establish model efficacy.

The completed visible-arm mean losses were:

| Stratum | Own visible | Wrong visible |
|---|---:|---:|
| Filler | 4.53250 | 4.52689 |
| Source | 5.85797 | 5.88872 |
| Copied rewrite | 6.28766 | 6.35177 |
| Source-absent content | 7.27849 | 7.29906 |
| Source-absent other | 5.55748 | 5.56618 |

The four-arm design fixed 20M exposure, 2,529-update learning-rate horizon and warmup 126. Its estimand was `I_partner(X) = (own_visible_X - own_blocked_X) - (wrong_visible_X - wrong_blocked_X)`. Negative interaction concentrated on source-absent content, stronger than copied-token interaction, would support the proposed channel. Visible-only differences do not establish it. The separate coherent-86M alpha-0.75 endpoint Overall 42.1210247099666 was not evidence for this mechanism.
