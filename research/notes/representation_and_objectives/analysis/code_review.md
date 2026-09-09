# corrected entity gate stats SGCR independent code review

This review is pinned by the source hashes in `runtime_probe.json`. The live
files changed during the audit; results refer to the final hashes recorded
there, not to transient intermediate revisions.

## Verdict

The current implementation is **not ready for a 100M-word launch**. Its model
hooking and bake/load mechanics now pass, its repaired semi-cold initialization
is gradient-live, and matched gradient clipping has been restored. However, the
decomposition builder does not create the intended legal16k decomposition. A
launch would test a pad-dominated gate-scaled residual rather than SGCR.

## Launch-blocking findings

### The decomposition tensor is 99.30% padding

`build_decomposition_map()` loads the raw legal16k `tokenizer.json`, whose
runtime settings fix padding and truncation at 256. It copies every `enc.ids`
entry without applying `attention_mask`. All 40,000 rows therefore have length
256; 10,168,071 of 10,240,000 stored slots (99.2976%) are padding. At d_comp=64,
the component lookup materializes 2,621,440,000 bytes (2.44 GiB) per effective
table computation in fp32. Input and decoder compute the table separately, so
the two component intermediates alone are about 4.88 GiB before other saved
autograd tensors.

Calling `no_padding()` is necessary but not sufficient. Isolated decode and
re-encode is context-sensitive for this byte-level tokenizer and disagrees with
the exact merge ancestry for 10,910/40,000 types (27.275%). For example, the
legal40k token `!` (id 5) must be component id 5, while isolated re-encoding
produces the leading-space token id 2893. Some isolated byte fragments also
decode through the Unicode replacement character.

The two tokenizer models provide a decisive repair: all 16,384 legal16k vocab
ids match the legal40k ids, and all 16,123 legal16k merges are an exact prefix of
the legal40k merges. Recursively undoing later legal40k merges yields a complete
exact map with lengths 1--7 and no empty rows. See `decomposition_audit.json`
and `build_prefix_decomposition_map()` in `candidate_sgcr_module.py`.

## Core mechanism and compatibility checks that pass

At the audited semi-cold initialization (random component codes, zero projection
and bias), `W_eff == W_std` bit-for-bit. K=0 and K=7 cold tests both had zero
embedding-table and logit differences from the standard model. Projection
weight gradient norm was 0.0245311 on the first backward pass; the component
table, necessarily zero-gradient while projection is zero, became live on the
second pass (norm 0.000421156). Thus the current one-factor-zero repair fixes the
original both-factors-zero dead path.

The hooked input and decoder each matched direct use of the same effective table
with maximum difference 0. The cleaned baked checkpoint contained no `_sgcr*`
model keys, loaded through ordinary
`transformers.DebertaV2ForMaskedLM.from_pretrained` with no missing or unexpected
keys, and had exact toy-batch logit parity (maximum difference 0). Saving also
restored the live hooked model exactly. The real legal40k tokenizer round-tripped
as `PreTrainedTokenizerFast` with vocab size 40,000.

The full live trainer completed a one-step, two-microbatch CPU integration run
over 16 words and produced a standard-loadable baked checkpoint. This verifies
that the repaired logging accessor is valid. The audited trainer also clips the
deduplicated standard-plus-SGCR parameter list at norm 1.0 before AdamW, matching
legal40k accum training completion. The integration does not validate the flawed decomposition or replace a
real training test.

## Attribution and secondary findings

The current uniform control uses the unweighted type mean rho. For K=50 this is
0.4558668, while the treatment's training-mass-weighted mean is 0.9357199. The
uniform residual multiplier is therefore 0.5441332 versus a mass-matched
0.0642801, an 8.465x difference. This control confounds routing with pathway
strength and optimization. Use the mass-weighted rho for the first uniform
control.

The random-decomposition control is described in notes but has no trainer flag
or implementation. A random map should permute exact decomposition rows within
length strata, excluding specials, rather than independently drawing component
ids. Uniform-real and support-random do not form a complete factorial without a
uniform-random cell, so interactions remain unidentified.

All special ids are now forced to rho=1, correctly preventing the corpus-zero but
masking-high-exposure `<mask>` token from becoming an unintended strong SGCR
route. Empty decompositions are also now forced standard, though the exact merge
map has none.

The separately saved component state has no implemented resume path, optimizer,
scheduler, counters, or RNG states. The base embedding could be approximately
reconstructed by subtracting the saved residual from baked `W_eff`, but this is
not bit-exact or implemented. Store the pre-bake base table (or a complete
unbake loader) and full training state if checkpoint preemption recovery is a
launch requirement.

The advertised warm initialization is incomplete: it computes an SVD projection
but leaves all component embeddings zero, contrary to the aligned-component
description. Do not select `--sgcr_init_mode warm` until implemented and tested.
