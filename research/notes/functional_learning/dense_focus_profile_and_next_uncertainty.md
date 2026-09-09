# dense focus profile and next uncertainty dense-focus profile and next uncertainty

## Active frontier target
The practical target is still a lawful BabyLM Strict-Small v5 improvement over coherent86/v4 projected Overall(AoA0) `42.1210247099666`. The two official-compatible dense jobs remain authoritative for endpoint judgment:

- seed62064 official-compatible evaluation for `unchanged_dense_focus_train/.../update_0080`.
- seed62065 official-compatible evaluation for `dense_focus_rep_seed62065_train/.../update_0080`.

This note does not infer terminal official values from unfinished payloads.

## What the two-seed replication now establishes
Seed62065 reproduced seed62064 on the fast-screen and trained-material readouts:

- Fast cheap7: seed62064 `44.80857142857143`, seed62065 `44.777142857142856`, coherent86 fast `44.56428571428571`.
- Shared fast item movement: `108219` common items, seed agreement `0.997431`, shared gains `3777`, shared losses `3725`, gain Jaccard `0.9633`, loss Jaccard `0.9653`.
- Qwen view-source help: seed62064 `+0.1066317`, seed62065 `+0.1103736`.
- Common-target movement: seed62064 mean `+0.2078753`, seed62065 mean `+0.2056452`; no-source remains small around `+0.02`, while source-original/source-altered/held-source are large positive.

This makes dense unchanged-Qwen focus a reproducible learning effect under the two tested seeds, not a one-seed accident. It is still not a v5 result without official-compatible Overall.

## Completed official-column state for seed62064
`data/dense_profile_localization/dense_profile_localization.md` localizes the official seed62064 columns that were already known in dense focus official and mechanism state:

- BLiMP delta `-0.450450` (net `-272` items over `59875`).
- Supplement delta `-0.597876` (net `-5` items over `5218`).
- EWoK delta `-0.097138` (net `+13` items over `7618`; macro and micro signs differ because of subtask weighting).
- Entity delta `+1.044733` (net `+42` items over `6780`).

The completed-column delta sum is `-0.100731`. The unresolved non-AoA columns COMPS, GlobalPIQA, Reading, and SuperGLUE need total delta `>0.100731` (average `>0.025183`) for AoA0 projected Overall to exceed coherent86. The fast seed62064 proxy for COMPS+GlobalPIQA+Reading is `+1.646980`, which would tolerate SuperGLUE down to roughly `-1.546249` only if that fast proxy were exact; terminal official values must replace this arithmetic context.

## Where the known official movement sits
Entity official movement is concentrated in operation depth, not a uniform Entity uplift:

- `4_ops`: `+3.1489` (net `+37`).
- `5_ops`: `+2.7027` (net `+9`).
- `2_ops`: `+1.8899` (net `+23`).
- `3_ops`: `+1.6129` (net `+20`).
- `1_ops`: `+0.0785` (net `+1`).
- `0_ops`: `-3.1149` (net `-48`).

This shape is compatible with dense focus improving some operation-depth or context-use behavior while shifting simple/no-operation behavior in the opposite direction. A comparison with the official Entity strata depended on completed corrected format and relation runs, but the connection was not yet a law: those experiments varied operation-instance diversity; the sparse/dense experiment used the same Qwen instances and varies target coverage/input corruption.

BLiMP official losses are broadly distributed, with family-level deltas all negative in the dense focus profile and next uncertainty grouping: binding/anaphora `-0.7680`, other `-0.7431`, agreement/number `-0.5834`, argument structure/ellipsis `-0.4277`, extraction/islands `-0.3229`, quantifier/NPI/scope `-0.2372`. The losses therefore cannot be dismissed as one isolated subtask. They may reflect the cost of additional dense supervised view prediction, the cost of dense input corruption, broader private-adapter movement, or ordinary score redistribution.

## Dense versus sparse input/label geometry
`data/dense_input_label_profile/dense_input_label_profile.md` profiles the exact 80-update unchanged-Qwen prefix and validates against actual training summaries.

Legal prefix: `20475` rows, `3162742` words, `3831` Qwen rows, `11778` Qwen pair segments, no generated compaction or topups.

Sparse seed62064:
- labelled/masked groups `21479`; labelled/masked target tokens `28590`.
- selected candidate groups after cap `61256`, label group fraction of raw `0.16237`.
- copied-token fraction among labels `0.42483`.

Dense seed62064/seed62065:
- labelled/masked groups `132283`; labelled/masked tokens `176607`.
- label and mask group fraction of raw `1.0`.
- copied-token fraction `0.41711`.

Prepared dense-mask/sparse-label control:
- seed62064-like labels remain sparse: `21479` groups / `28590` tokens.
- masks become dense: `132283` groups / `176607` tokens.
- mask-only tokens `148017`; label/mask token ratio `0.16188`.

This control is meaningful if official evaluation preserves useful Entity/source-responsive gains but aggregate v5 status is blocked by localized costs: it tests whether the effective-input side of dense masking can be separated from the extra supervised target coverage and its costs. If full official gains largely disappear, the reason to spend H100 time on this control becomes weaker.

## Corrected input-corruption and target-ratio audit
The format screen exposed a real fault: offset-based grouping masked whole rows and produced about `1.47` targets/word. The corrected trainer uses tokenizer word-start tokens, asserts about 15% target ratio, zero private-on/off initial CE difference, eval-mode neutral KL, and separates the coherent KL leash from no-gradient coherent readout. This is directly aligned with the effective-input validity requirement: an experiment's surface purpose is not enough; the actual corrupted input and target ratio must be checked.

The tight-private relation result and corrected format experiments should be compared with the dense-acquisition results only at the level of a shared question: how finite experience becomes training evidence for reusable computations. The relation experiments vary diversity of operation instances; dense acquisition uses the same Qwen instances and varies dense masking/coverage. A law connecting them would require direct evidence about which aspect of available experience changes the learned computation.

## Planned interpretation of completed official results
1. Validate the final seed62064 payload and use the SuperGLUE-aware `official_transition_compare.py` against the coherent86 official zero-shot/Reading and SuperGLUE references.
2. Apply the same comparison to seed62065.
3. Compare score concordance, item movement, Entity depth strata, BLiMP/Supplement costs, COMPS/GlobalPIQA/Reading stability, and SuperGLUE movement across seeds.
4. A practical improvement claim requires both seeds to exceed coherent86 under the same complete evaluation coordinate.
5. If meaningful component gains are erased in the aggregate by other losses, the dense-mask/sparse-label contrast remains a proposed test of input-side corruption versus target coverage, not a coefficient sweep.
6. If apparent gains disappear under full official evaluation, dense acquisition remains a trained-material/fast-screen effect with weaker practical value; the evidence-formation hypothesis would need revision.

## Addendum: zero-shot/Reading current thresholds for final SuperGLUE
After `data/zero_reading_dense_synthesis/zero_reading_dense_synthesis.md`, the completed zero-shot/Reading official-sized columns give:

- seed62064 cheap7 computed delta `+0.2208926866129559`, so the sum over the seven cheap7 columns is `+1.5462488062906913`. Under AoA0 arithmetic, final SuperGLUE primary-metric delta can be as low as `-1.5462488062906913` and still tie coherent86; equivalently dense seed62064 needs final SuperGLUE mean above about `68.27297358340866` given coherent86 SuperGLUE `69.81922238969935`.
- seed62065 cheap7 computed delta `+0.2061247939166364`, so the sum over the seven cheap7 columns is `+1.4428735574164548`. It needs final SuperGLUE mean above about `68.37634883228289` to tie coherent86.

The current payload `SuperGLUE` field in both running dense jobs contains only completed BoolQ (`68.56269113149847`) with one task entry and zero primary-metric detail entries, as recorded in `data/dense_interim_payload_provenance/dense_interim_payload_provenance.md`; it must not be used as final SuperGLUE or Overall.
