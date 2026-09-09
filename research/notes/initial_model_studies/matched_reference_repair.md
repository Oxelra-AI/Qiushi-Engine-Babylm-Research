# matched reference repair — Matched official reference repair (before 4M trace)

## Why this was needed

The bsm 1m route control route decision required an official reference that isolates update-count and packing effects from binding content. The earlier analysis `official_short_targeted` reference did NOT do this: in the 100k smoke it had 6875 rows / 107 updates while the BSM arm had 1899 rows / 30 updates. Update count and packing were still confounded, so official-column changes could not be attributed to relation consistency.

## What was built

`scripts/materialize_bsm_matched_official_reference.py` takes a BSM corpus as a *schedule* and replaces each row with official text of the SAME word length and SAME training mode:

- BSM `binding` row → `official_targeted` row (official text, one metadata target answer);
- BSM `official` row → `official` row (official text, standard WWM).

This matches, against the coherent BSM arm:

- row count (identical);
- exact per-row word-length sequence (identical, verified `same_word_sequence=true`);
- row order and therefore batch composition;
- number of targeted rows (`targeted_rows_match_binding_rows=true`);
- total words and update count.

The only difference from a BSM arm is that the targeted rows contain official text with an arbitrary content-word target, rather than entity-value binding text with the answer target. So the reference carries update/packing/targeted-supervision structure WITHOUT consistent entity-value correspondence.

## Trainer repair

`train_legal_bsm_screen.py` previously applied targeted masking only when `row.kind == 'binding'`. It now treats any row with `kind in ('binding','official_targeted')` and a valid `mask_char_start` as targeted. This lets the matched reference reproduce the BSM per-batch target opportunity.

## Smoke verification (100k)

- coherent BSM: 1899 rows, 99847 words, 30 steps, targeted 36–50/batch, WWM 134–410/batch.
- matched reference: 1899 rows, 99847 words, 30 steps, targeted 45–64/batch, WWM 133–419/batch.

Row count, word exposure, and update count now match. Targeted-mask opportunity is in the same range (small differences arise because official targeted words tokenize to slightly more subwords than short binding answers; this is expected and does not reintroduce the 3× update confound).

## Consequence for the 4M trace

The four-arm developmental trace should use:

1. `official_standard` (normal long rows, standard WWM) — the usual baseline;
2. `official_bsm_matched_reference` (this repaired reference) — update/packing/targeted-supervision control;
3. `bsm_paired_coherent`;
4. `bsm_paired_swapped`.

Primary mechanism contrast: coherent − swapped (relation consistency under matched structure).
Content-vs-structure contrast: BSM arms − matched reference.
Do NOT use coherent − official_standard as the relation effect.

For the 4M run, regenerate the matched reference from the 4M coherent corpus (schedule must match the actual 4M arm, not the 100k smoke).
