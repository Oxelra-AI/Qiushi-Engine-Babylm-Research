# context geometry and v5 design context geometry and the next v5 composition design

This note records new execution evidence from context geometry and v5 design and turns it into a concrete research design object for the remaining legal Strict-Small v5 work. It is not a final paper or a selected endpoint.

## Why this step changed the format line

format attribution controls showed that `coherent_unsplit_special` is not a practical lever: both seeds kept the corrected training invariants but repeated a column trade, especially BLiMP up and Supplement/EWoK/Reading down. context geometry and v5 design asked whether this trade is part of the same Stage-II substitution coordinate that earlier separated ALN, SHUF, SEP, DUP, VIEW, and REPEAT.

The old Stage-II context coordinate is:

\[
\text{context\_gain}=L(\text{sentence in isolation})-L(\text{same sentence inside its 160-word row}).
\]

A lower value means the checkpoint gains less from adjacent row context on the same masked target span. It does not directly score Supplement or EWoK items; it is a reusable natural-text coordinate for whether a model has shifted toward isolated-span prediction or adjacent-context-supported prediction.

## Context-gain evidence from the coherent-special arm

The scorer `experiments/archive/relation_learning/scripts/format_context_gain_coordinate.py` reuses the earlier analysis/earlier analysis Strict-complement sentence coordinate under trusted loading. It scored 651 selected sentence spans and applied the earlier analysis filtering rule, leaving 623 paired spans per endpoint. Outputs:

- `research/documents/relation_learning/data/format_context_gain_coordinate/format_context_gain_summary.md`
- `experiments/archive/relation_learning/data/format_context_gain_coordinate/format_context_gain_summary.json`
- `experiments/archive/relation_learning/data/format_context_gain_coordinate/source_contrasts.csv`
- `experiments/archive/relation_learning/data/format_context_gain_coordinate/model_identity.jsonl`

Trusted model identities were correct: `chck82_slow_scale1p75` loaded `AdapterDebertaV2ForMaskedLM` with 35,463,008 parameters and 995,584 adapter parameters; coherent86 and both coherent-special endpoints loaded `FrozenSlowPrivateDebertaV2ForMaskedLM` with 36,458,592 parameters and 1,991,168 adapter/private parameters.

Main results:

| endpoint | row loss | isolated loss | context gain |
|---|---:|---:|---:|
| chck82_slow_scale1p75 | 3.0055 | 4.0296 | 1.0241 |
| coherent86_alpha075 | 2.9921 | 4.0300 | 1.0378 |
| coherent_special_98097_alpha075 | 2.9873 | 3.8229 | 0.8356 |
| coherent_special_98098_alpha075 | 2.9869 | 3.8232 | 0.8363 |

Paired deltas versus coherent86:

| endpoint | d_row | d_iso | d_context_gain | se |
|---|---:|---:|---:|---:|
| coherent_special_98097_alpha075 | -0.0048 | -0.2071 | -0.2022 | 0.0285 |
| coherent_special_98098_alpha075 | -0.0052 | -0.2067 | -0.2015 | 0.0282 |

The change is strong, replicated, and broad across sources. Versus coherent86, delta context gain is negative in BNC spoken (-0.086/-0.094), CHILDES (-0.381/-0.329), Gutenberg (-0.200/-0.211), OpenSubtitles (-0.287/-0.310), SimpleWiki (-0.103/-0.105), and Switchboard (-0.118/-0.114). The row-context loss moves only slightly, while isolated-span loss improves by about 0.207 nats. Thus the special-token branch does not merely damage row-context prediction; it substantially improves isolated-span fit and thereby reduces the marginal value of adjacent context.

This is the Stage-II coordinate for the format attribution controls column face. The COMPACT_EXPERIENCE SHUF family previously had a related practical face: BLiMP/Reading improved while Supplement/Entity fell relative to ALN/OFF, and earlier analysis measured SHUF as lower context gain than OFF or far lower than ALN. The coherent-special arm reaches the same kind of substitution from the opposite end: not wrong correspondence in the trunk, but special-token/private-branch format exposure that shifts the model toward short isolated prediction. The result supports the fixed-budget interpretation in a directly practical form: under finite private capacity and finite experience, a format channel can buy isolated-span competence while losing context-supported competence.

## Geometry of the special-token channel

The fast format attribution controls short-format readout showed that private/slow KL was much larger on short isolated rows than on coherent rows. context geometry and v5 design refined this by binning token KL by row length and distance to the nearest special token. Scripts and outputs:

- `experiments/archive/relation_learning/scripts/special_geometry_kl_bins.py`
- `experiments/archive/relation_learning/data/special_geometry_kl_bins/special_geometry_kl_bins.md/json`
- `experiments/archive/relation_learning/scripts/summarize_special_geometry.py`
- `experiments/archive/relation_learning/data/special_geometry_interpretation/special_geometry_interpretation.md/json`

The same 32 matched examples from the format attribution controls fast readout were used. This is a bounded but direct measurement of where the KL lives.

| endpoint | form | KL mean | special mass | content dist=1 mass | content dist<=15 mass | far-content mass | short<=31 mass |
|---|---|---:|---:|---:|---:|---:|---:|
| coherent86_alpha075 | coherent_add_special | 0.00082465 | 6.9% | 2.4% | 13.6% | 79.6% | 0.0% |
| coherent86_alpha075 | short_add_special | 0.00183627 | 30.1% | 10.2% | 47.7% | 22.3% | 48.7% |
| coherent_special_98097_alpha075 | coherent_add_special | 0.00193546 | 11.4% | 38.0% | 45.8% | 42.8% | 0.0% |
| coherent_special_98097_alpha075 | short_add_special | 0.00974691 | 40.9% | 41.8% | 52.4% | 6.7% | 62.7% |
| coherent_special_98098_alpha075 | coherent_add_special | 0.00240540 | 12.1% | 43.5% | 50.0% | 37.9% | 0.0% |
| coherent_special_98098_alpha075 | short_add_special | 0.01199229 | 47.9% | 35.2% | 45.4% | 6.7% | 67.4% |

The coherent-special endpoints concentrate short-row KL at `<s>`, `</s>`, and the immediately adjacent content token. Far-content mass is only 6.7% on short rows for both seeds, whereas coherent86 has 22.3%. The largest KL tokens include both special tokens and first content tokens such as `ĠApril`, `ĠHow`, `ĠWhy`, `ĠWhether`, and first-token positions in long coherent rows. This supports a special-token geometry channel rather than a purely global private-branch drift: every token in a short evaluation row lies close to the special tokens under the relative-position geometry, whereas most content in a 256-token coherent row does not.

This matters for interpreting isolated and half-format arms. If they have large KL but flatter distance profiles, then seeing short geometry during training may have taught the branch a less concentrated special-token rule. If they repeat the same near-special concentration and lower context gain, the format line closes as a practical v5 component but still strengthens the general principle: input geometry controls which private computation is actually restrained and transferred.

## Consequence for the next composed candidate

A private branch is controlled at evaluation time only on input geometries that training or the preservation term actually covers. A coherent-row-only preservation term leaves short special-token geometry open. The next composed candidate, if one is justified by the pending faithful results, should explicitly include geometry as part of the training design rather than inheriting the coherent-only leash from earlier analysis.

The current word accounting matters. The AoA smoke test identifies coherent86 as `endpoint_86.005295M`, so a continuation from coherent86 has roughly 13,994,705 remaining legal words under the 100M Strict-Small accounting. If the final candidate is trained fresh from `chck_82M`, the available private continuation room is larger, but the final assembled endpoint still must carry actual word-count labels and measure AoA from its true trajectory. Any gradient preservation examples count as charged training words. No-gradient readouts do not train the model and should not be confused with charged training examples.

A scientifically informative composition should therefore separate four roles:

1. **Content-install rows.** These are the main rows that try to install the surviving training principle, likely dense target coverage if its faithful SuperGLUE and measured AoA preserve the current zero-shot/Reading gain. They must be legal Strict-Small material and directly trained in the final composed stream rather than added arithmetically from separate endpoints.
2. **Coherent preservation rows.** These protect the long-row context-supported surface that coherent86 uses for Supplement/EWoK/Reading-like behavior. The earlier analysis coherent-only leash was too narrow, but abandoning coherent rows entirely risks losing the context-supported part of v4.
3. **Short special-token preservation rows.** These should be ordinary legal corpus sentences or short rows with official special tokens, disjoint from any no-gradient readout. Their purpose is to prevent the branch from using an unconstrained near-special geometry shortcut while still allowing content-specific changes on the installed rows.
4. **No-gradient readouts.** At minimum: the coherent readout used in earlier analysis, a matched short special-token readout, and the Stage-II context-gain coordinate. These are not training examples; they reveal whether the branch moved along isolated-span versus context-supported axes.

A conservative allocation for a single two-seed composed experiment after pending results could be expressed as a mixture rather than fixed here: keep all charged words below the true remaining legal room; spend most words on the surviving content-install rows; reserve a substantial minority for preservation rows split between coherent and short special-token geometry; report both coherent and short readout KL at every run; and run the context-gain coordinate on the final endpoints. The exact split should be fixed before launching the composed seeds, after the no-special and embedding-only controls say whether the special-token damage belongs to the tokens, the earlier analysis trainer family, or broad private-branch freedom.

## How to read pending arms with the new coordinate

`coherent_unsplit_special` is already a negative training lever under the earlier analysis two-seed rule, but it is now a positive scientific result: it identifies a replicated shift from context gain to isolated-span fit and localizes the hidden KL to special-token geometry.

For `isolated_all` and `half_coherent_half_isolated`, the score columns should be read together with two additional measurements:

- short-format private/slow KL and geometry bins, using `special_geometry_kl_bins.py` once their alpha0.75 endpoints exist;
- context-gain deltas, using `format_context_gain_coordinate.py` with an endpoints JSON that includes chck82, coherent86, the two coherent-special seeds, the isolated seeds, and the half seeds.

Directional expectations to test rather than assume:

- `isolated_all`: likely lowers context gain further if short-row practice becomes the dominant private relation; it may improve isolated zero-shot-like columns but risks Supplement/EWoK/Reading-like surfaces.
- `half_coherent_half_isolated`: the key test is whether mixed geometry lets the branch condition on context presence, preserving context gain while reducing short-format KL concentration. If it only averages the two behaviors, it is unlikely to be a useful v5 component.
- no-special control: determines whether the Supplement/EWoK/context-gain loss is due mainly to special-token exposure or to the earlier analysis trainer family.
- embedding-row-only control: determines whether a narrow special-token correction can help without opening the full private-branch geometry channel. Because embeddings and decoder weights are tied, this is a two-row tied-embedding/output intervention, not pure input-only training.

## Reproduction assets

Reusable scripts:

- `experiments/archive/relation_learning/scripts/format_context_gain_coordinate.py`
- `experiments/archive/relation_learning/scripts/special_geometry_kl_bins.py`
- `experiments/archive/relation_learning/scripts/summarize_special_geometry.py`

Current evidence files:

- `research/documents/relation_learning/data/format_context_gain_coordinate/format_context_gain_summary.md`
- `research/documents/relation_learning/data/special_geometry_interpretation/special_geometry_interpretation.md`
- `experiments/archive/relation_learning/data/special_geometry_kl_bins/kl_by_distance.csv`
- `research/documents/relation_learning/data/coherent_special_sofar/coherent_special_sofar.md`

The central practical target remains unchanged: produce one legal, faithfully evaluated, directly trained v5 endpoint that exceeds faithful v4. The current contribution is not another format arm but a sharper control of the private branch's evaluation geometry so that a future composition, if dense target coverage or another lever survives, tests the right scientific question instead of inheriting an uncontrolled short-format trade.

## Reuse commands for pending endpoint geometry

Endpoint template for future runs:

- `experiments/archive/relation_learning/data/format_context_gain_coordinate/pending_format_endpoints_template.json`

After the listed endpoints are complete and their paths exist, score context gain with a new output directory, for example:

```text
python -B experiments/archive/relation_learning/scripts/format_context_gain_coordinate.py \
  --endpoints-json experiments/archive/relation_learning/data/format_context_gain_coordinate/pending_format_endpoints_template.json \
  --out-dir experiments/archive/relation_learning/data/format_context_gain_coordinate_all_delivered \
  --device cpu --batch-size 32 --torch-threads 8
```

For special-geometry KL bins, create a smaller endpoints JSON containing only endpoints that have `model.safetensors` and run:

```text
python -B experiments/archive/relation_learning/scripts/special_geometry_kl_bins.py \
  --endpoints-json experiments/archive/relation_learning/data/format_context_gain_coordinate/pending_format_endpoints_template.json \
  --out-dir experiments/archive/relation_learning/data/special_geometry_kl_bins_all_delivered \
  --device cpu --rows 32 --batch-size 16
```

If the full 256-row readout is needed after task slots free, use the format attribution controls full readout script with `--endpoints-json` pointing to a measured endpoint subset and write to a fresh directory so the format attribution controls fast evidence is not overwritten.
