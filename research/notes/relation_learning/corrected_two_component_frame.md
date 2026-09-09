# corrected two component frame corrected scientific frame: two-component relation-typed composition with asymmetric reach

Created: 2026-09-06

This note supersedes the earlier one-dimensional dose-curve reading. It is the paper-ready frame, verified against all existing data tables.

## The corrected principle

Under finite training budget, local sequence composition installs two separable components:

**1. Trust/lookback component.**
Any corresponding or identical in-window pairing (but NOT non-corresponding adjacency) installs trust in adjacent spans. This shows as:
- Zero-update Entity advantage: the model retrieves the unchanged stated fact from a recently read passage
- Distraction from unrelated neighbors: models that practiced correspondence or identity are hurt MORE by unrelated adjacent text (negative ΔA_U on Wikipedia)
- Small superseded-state cost at deeper Entity updates

This component does not require exact recurrence — corresponding restatement (ALN) installs it from scratch (ALN−OFF Entity rel_eq0 = +9.54 at seed43022, +21.03 at seed43122). Exact duplication adds an increment above the base installed by correspondence (DUP−OFF = +15.46). Practiced non-correspondence (SHUF) installs the opposite: −0.66 below OFF.

**2. Readout component.**
The relation type sets the readout precision on top of the trust component:
- **Identity readout**: exact recurrence trains a format-bound routine that benefits identical-surface retrieval. Its changed-form liability travels broadly across register, objective, and target class.
- **Form-robust readout**: corresponding restatement trains readout for the specific target class that was practiced. Its competence transfers for the practiced target relation even across register, but does NOT help unpracticed target relations.

## Evidence verification

### Competence follows practiced target relation across register

ALN−OFF (qwen block: 37.6K corresponding restatement pairs with content overlap ~0.63, practicing recurring-content-under-changed-form):

| Probe | Target class | ΔA_T seed43022 | ΔA_T seed43122 | Interpretation |
|-------|-------------|-----------------|-----------------|----------------|
| Compact (FineWeb register) | overlap | **+1.649** | **+1.699** | Crosses register for practiced target relation |
| Compact (FineWeb register) | nonoverlap | +0.059 | −0.193 | Does NOT help unpracticed target relation |
| Wikipedia (matching register) | overlap | **+3.598** | **+4.200** | Strong in matching register |
| Wikipedia (matching register) | nonoverlap | −0.593 | −0.881 | Does NOT help source-absent substitutions |

V−C (33.3K designed compact rewrites, practicing source-absent substitution in FineWeb register):

| Probe | Target class | ΔA_T (3-seed mean ± SE) | Interpretation |
|-------|-------------|--------------------------|----------------|
| Compact (matching register) | nonoverlap | **+0.710 ± 0.076** | Strong for practiced target relation in matching register |
| Compact (matching register) | overlap | +0.587 (also positive) | Moderate for source-recurring class |
| Wikipedia (different register) | overlap (source-recurring) | +0.238 ± 0.071 | Small cross-register transfer |
| Wikipedia (different register) | nonoverlap | +0.032 ± 0.040 | Near zero for unpracticed relation in different register |

### Liability from exact recurrence generalizes broadly

DUP−OFF (exact duplication, BabyLM register):

| Probe | Target class | ΔA_T | Interpretation |
|-------|-------------|------|----------------|
| Compact nonoverlap | changed-form | **−1.926** | Large cost in different register |
| Wikipedia nonoverlap | changed-form | **−1.184** | Cost in matching register |
| Wikipedia overlap | identity-form | +3.342 | Benefits when target recurs identically |
| Compact overlap | identity-form | +0.557 | Benefits even in different register |

R−C (designed exact recurrence):

| Context | Target class | Quantity | Interpretation |
|---------|-------------|----------|----------------|
| Compact nonoverlap | 3-seed ΔA_T | **−0.683 ± 0.173** | Stable cost |
| Wikipedia nonoverlap | ΔA_T | **−0.217 ± 0.029** | Cost crosses register |
| Causal GPT compact nonoverlap | ΔA_T | **−1.153** | Cost crosses objective |

### Trust component: unrelated-source deficit

| Contrast | Wikipedia ΔA_U (seed43022) | ΔA_U (seed43122) | Reading |
|----------|---------------------------|-------------------|---------|
| ALN−OFF | −0.323 | −0.527 | Correspondence installs trust → unrelated distraction |
| DUP−OFF | −0.288 | — | Identity also installs trust |
| SHUF−OFF | +0.115 | — | Non-correspondence does NOT |
| SEP−OFF | +0.108 | — | Separated does NOT |

### Trust component: Entity zero-update

| Arm/Contrast | Entity rel_eq0 accuracy (%) | ΔEntity vs OFF/C |
|-------------|------------------------------|-------------------|
| C (designed CLEAN, has qwen block) | 39.45 | — |
| R (CLEAN + 33K exact) | 48.24 | R−C = +8.79 |
| V (CLEAN + 33K rewrite) | 38.55 | V−C = −0.91 |
| ALN (COMPACT_EXPERIENCE, 37.6K corresponding) | 36.18 | ALN−OFF = +9.54 |
| OFF (COMPACT_EXPERIENCE, no qwen) | 26.64 | — |
| DUP (COMPACT_EXPERIENCE, 37.6K exact) | 42.11 | DUP−OFF = +15.46 |
| SHUF (COMPACT_EXPERIENCE, wrong correspondence) | 25.99 | SHUF−OFF = −0.66 |

The designed C already has substantial trust via its inherited qwen block; additional rewrites (V) don't further improve Entity zero-update, but additional exact pairs (R) do.

### Cooperation between components (HM vs HV)

Compact nonoverlap (the designed rewrite readout):
| Arm | A_T | ΔA_T vs C | Natural copy gain |
|-----|-----|-----------|-------------------|
| C | 1.527 | — | 3.899 |
| HV (16.7K rewrite only) | 1.544 | +0.018 | 3.878 |
| HM (16.7K rewrite + 16.6K exact) | 1.872 | +0.346 | 4.425 |
| V (33.3K rewrite) | 2.237 | +0.710 | 4.304 |
| R (33.3K exact) | 0.844 | −0.683 | 4.421 |

HALF_VIEW (16.7K rewrites alone) barely moves anything: HV−C = +0.018 on compact nonoverlap, near zero.
Hash-mixed (same 16.7K rewrites PLUS 16.6K exact copies) reaches HM−C = +0.346 — 49% of VIEW's effect.

The exact pairs supply the trust/lookback component (copy gain: HM near R), and the rewrite pairs' readout expresses on top of that. Without the trust component (HV), the rewrites alone aren't sufficient at 16.7K dose. This is cooperation between relations, not competition.

Entity HM/HV data is pending. Prediction: HM Entity rel_eq0 should be above C (exact pairs install trust), HV should be near C.

### SHUF as practiced non-correspondence (seed43022 only)

SHUF preserves local rewrite-register adjacency but breaks own-source correspondence:

| Probe | Target class | SHUF−OFF ΔA_T | Interpretation |
|-------|-------------|---------------|----------------|
| Compact nonoverlap | changed-form | −0.085 | Small |
| Compact overlap | source-recurring | **−0.702** | Practiced mismatch → discounting |
| Wikipedia overlap | source-recurring | **−0.994** | Discounting in matching register |
| Wikipedia nonoverlap | source-absent | +0.012 | No effect on this class |

SHUF−OFF ordinary-heldout Δloss = +0.013 (tiny broad-fit cost), so the source-recurring cost is not explained by generic degradation.

This is a third relation: wrong local restatement installs discounting, active suppression of source-recurring retrieval for the non-corresponding class.

## Implication for the field

The reach asymmetry predicts the field-level pattern in neighboring literature:
- **Near-duplicate structure is broadly harmful**: the liability from identity/recurrence (changed-form cost) generalizes across register, objective, and target class. This is why deduplication helps broadly.
- **Retrieval-packing and paraphrase augmentation help narrowly**: the competence from corresponding restatement transfers only for the practiced target relation. This is why ICLM's within-window composition helps and why paraphrase augmentation improves only related tasks.

## Seed count labels

| Quantity | Seeds | Arms/conditions per seed |
|----------|-------|--------------------------|
| ALN−OFF compact/Wikipedia overlap ΔA_T | **Two** (43022, 43122) | Two seeds confirmed |
| ALN−OFF compact/Wikipedia nonoverlap ΔA_T | **Two** (43022, 43122) | Two seeds confirmed |
| V−C compact/Wikipedia | **Three** (43022, 43122, 43222) | Designed family |
| R−C compact/Wikipedia | **Three** | Designed family |
| DUP, SHUF, SEP, HM, HV | **One** (43022 only) | Label as one-seed |
| Causal GPT R/RS | **One** (43022 only) | One-seed objective boundary |

## Pending checks (bg tasks)

1. Entity HM/HV: cooperation test
2. Wikipedia holdout near-duplicate check: confirm probe is clean against qwen block's 10.8K simple_wiki pairs
3. ALN−OFF compact ΔA_U: already in the data, record below

ALN−OFF compact ΔA_U:
- seed43022 nonoverlap: −0.046, overlap: −0.023
- seed43122 nonoverlap: −0.101, overlap: −0.127

These are small and mixed-sign in compact register, while Wikipedia ΔA_U is consistently negative. The trust/distraction effect is stronger in the matching register.

## Data paths

- COMPACT_EXPERIENCE compact/Wikipedia/Entity/copy: `data/paired_context_relation_design_probe/`
- COMPACT_EXPERIENCE seed43122 ALN/OFF: `data/paired_context_aln_off_seed43122_probe/`
- Half-view curve compact/Wikipedia/copy: `data/half_view_curve_probe/`
- Designed 3-seed compact/Wikipedia/Entity: `data/numerical_repair/`, `split_entity_official_integration/`
- Causal GPT: `data/causal_gpt_relation_boundary/`
- COMPACT_EXPERIENCE ordinary heldout: `data/paired_context_ordinary_heldout_loss/`
