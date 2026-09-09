# targeted masking closure and lamb opening — Targeted masking closure and whole-learning-system reopening

## Targeted relational masking: synthesis does not hold

The route requires a benchmark-independent demonstration that a relational masking signal
isolates sequential state composition rather than visible lexical lookup, remains unsolved by
spatial repair route status at meaningful mass, and explains the Muon/PVDM failures against matched controls. This
synthesis does not hold, for three converging reasons:

### 1. Accumulated evidence: all targeted masking variants redistribute, never lift broadly

| Intervention | Target | cheap7 Δ vs spatial repair route status | Signature |
|---|---|---:|---|
| Innovation masking (earlier analysis) | source-absent novel forms | −1.053 at 80M | 6/7 columns worse |
| Word-mean MLM (earlier analysis) | rare/infrequent tokens | −0.246 at 80M | Supp −1.71, EWoK −1.81 |
| Minfreq50 (support error conditioned probe) | tokens ≥50 freq, init-matched | −0.168 at 80M | Supp −2.29, EWoK −1.44 |
| Muon update geometry (mature muon reversal and switch repair) | hidden-matrix orthogonalization | −0.161 at 80M | Entity −2.89, Reading −1.03 |
| Muon20→AdamW switch (mature muon reversal and switch repair) | early orthogonalization only | −0.144 at 80M | Entity −3.47 |

All share the redistribution pattern: one or two surfaces improve while others degrade. No
targeted intervention moved cheap7 or Overall above spatial repair route status at mature (80M+) exposure.

### 2. independent empirical confirmation

The corrected dual-view design: full-context pivot-substitution probe on FW compact/rowblock/
interleaved showed true-pivot preference saturated at 86.4–89.1% event correlation, range only
0.041 of event SD. A zero-extra-view masked-alt pilot was even more saturated at 94.1%.
Conclusion: "current relation objectives mostly hit local lexical/context compatibility already
learned by failing endpoints."

A model-free audit of 18,651 pairs found only 2 antonym/opposition pairs and 5 comparative pairs in the strict-overlap subset. This broad pool does not justify training unless full calibration establishes arm separation.

### 3. Mechanistic explanation

In a fixed-capacity model (34.5M params), hidden representations serve all prediction tasks
simultaneously. Upweighting any subset of positions (relational, rare, or compositional)
increases gradient signal for features serving that subset but reduces relative gradient for
features serving the remaining positions. With 10M-word vocabulary and 8-layer depth, the
representational budget is tightly constrained. The shared-capacity tradeoff is intrinsic, not
a failure of any particular masking design.

### Verdict

Targeted masking of any form—relational, innovation, frequency-based, or masked-alternative—
cannot break the redistribution tradeoff in this system. "do not manufacture
another narrow masking variant; reopen the broader whole-learning-system comparison."

---

## Next direction: LAMB optimizer (whole-learning-system change)

### Rationale

The 41.8 leader (`wwm_curriculum_simplification_40k`) uses LAMB lr 0.007, while all endpoints use AdamW lr 0.001. LAMB is the single most prominent unexplored whole-learning-system
variable. Unlike targeted masking, LAMB changes ALL gradient dynamics simultaneously through
per-layer adaptive trust ratios, potentially avoiding the redistribution pattern.

LAMB's trust ratio = ||w||₂ / ||update||₂ gives each parameter layer an effective LR
proportional to its weight norm, normalized by its update norm. This naturally handles the
heterogeneous parameter scales in DeBERTa-v2 (word embeddings, relative position encodings,
attention layers at different depths, FFN layers, norms). AdamW applies a uniform LR to all
parameters of the same group, which may under- or over-step parameters whose natural scales
differ.

### Difference from Muon

Muon failed because it orthogonalized hidden-matrix updates, rotating the subspace away from
compositional credit and creating a trajectory-locked tradeoff. LAMB preserves the Adam update
direction but scales the magnitude per-layer by the trust ratio. This is a fundamentally
different intervention: direction-preserving vs. direction-changing.

### Screen design

- Substrate: exact spatial repair route status legal compact-view-reinvest corpus, tokenizer, architecture, WWM,
  seeds, batch 256, seq256, cosine warmup 6%
- Only change: optimizer AdamW → LAMB
- LR to screen: 0.005 and 0.007 (the leader's range)
- Weight decay: 0.01 (unchanged)
- LAMB betas: (0.9, 0.999) per LAMB standard; base AdamW uses (0.9, 0.98)
- Bounded 20M screen first, continue to 80M+ only if 20M cheap7 > spatial repair route status 20M broadly
- Two arms in parallel on GPU 0 and GPU 1
